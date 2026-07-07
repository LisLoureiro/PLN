"""
store.py
Persiste documentos e jobs de extração no PostgreSQL, e os itens
extraídos no ChromaDB para busca semântica futura.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from sqlalchemy import (
    Column, DateTime, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from custom_extractor import ExtractedItem

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    """Markdown normalizado — fonte da verdade do PDF, cacheado por hash."""
    __tablename__ = "documents"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    doc_hash    = Column(String(64), unique=True, nullable=False, index=True)
    source_file = Column(Text, default="")
    markdown    = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    char_count  = Column(Integer, default=0)


class ExtractionJobModel(Base):
    """Um job = um PDF + uma instrução de extração + os itens resultantes."""
    __tablename__ = "extraction_jobs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    job_id      = Column(String(64), unique=True, nullable=False, index=True)
    doc_hash    = Column(String(64), nullable=False, index=True)
    source_file = Column(Text, default="")
    instrucao   = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    total       = Column(Integer, default=0)
    items_json  = Column(Text, default="[]")
    
class Store:
    """Camada de persistência dupla: PostgreSQL (metadados) + ChromaDB (busca vetorial)."""

    CHROMA_PATH = "./chroma_db"

    def __init__(self):
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://extractor:extractor123@localhost:5432/extractor",
        )
        self._engine = create_engine(db_url, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)
        self._create_tables_safely()
        logger.info("[Store] PostgreSQL conectado.")

        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            _ef = DefaultEmbeddingFunction()
        except Exception:
            _ef = None

        self._chroma = chromadb.PersistentClient(path=self.CHROMA_PATH)
        self._items = self._chroma.get_or_create_collection("extracted_items", embedding_function=_ef)
        logger.info("[Store] ChromaDB pronto.")

    def _create_tables_safely(self, attempts: int = 3) -> None:
        """
        Cria as tabelas se não existirem. Tolerante a corrida: se dois
        processos (ex: workers do Gunicorn) tentarem criar as tabelas ao
        mesmo tempo, o "perdedor" recebe um erro de chave duplicada do
        catálogo do Postgres — nesse caso apenas verificamos se as tabelas
        já existem e seguimos em frente, em vez de derrubar o processo.
        """
        import time
        from sqlalchemy import inspect
        from sqlalchemy.exc import IntegrityError, OperationalError

        for attempt in range(1, attempts + 1):
            try:
                Base.metadata.create_all(self._engine)
                return
            except (IntegrityError, OperationalError) as e:
                inspector = inspect(self._engine)
                existing = set(inspector.get_table_names())
                expected = set(Base.metadata.tables.keys())
                if expected.issubset(existing):
                    logger.warning(
                        "[Store] Corrida na criação de tabelas (outro processo venceu) — "
                        "tabelas já existem, seguindo normalmente."
                    )
                    return
                if attempt == attempts:
                    raise
                logger.warning("[Store] Tentativa %d/%d de criar tabelas falhou (%s), tentando de novo…",
                                attempt, attempts, e)
                time.sleep(0.5 * attempt)

    # ──────────────────────────────────────────────
    # Documentos (Markdown — cache por hash do PDF)
    # ──────────────────────────────────────────────

    def get_document_by_hash(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        with self._Session() as session:
            row = session.query(DocumentModel).filter_by(doc_hash=doc_hash).first()
            if not row:
                return None
            return {
                "doc_hash": row.doc_hash,
                "source_file": row.source_file,
                "markdown": row.markdown,
                "char_count": row.char_count,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }

    def save_document(self, doc_hash: str, source_file: str, markdown: str) -> None:
        with self._Session() as session:
            existing = session.query(DocumentModel).filter_by(doc_hash=doc_hash).first()
            if existing:
                existing.markdown = markdown
                existing.source_file = source_file
                existing.char_count = len(markdown)
                existing.created_at = datetime.utcnow()
                logger.info("[Store] Documento %s atualizado (%d chars).", doc_hash[:8], len(markdown))
            else:
                session.add(DocumentModel(
                    doc_hash=doc_hash, source_file=source_file,
                    markdown=markdown, char_count=len(markdown),
                ))
                logger.info("[Store] Documento %s salvo (%d chars).", doc_hash[:8], len(markdown))
            session.commit()

    # ──────────────────────────────────────────────
    # Jobs de extração
    # ──────────────────────────────────────────────

    def save_job(
        self,
        job_id: str,
        doc_hash: str,
        source_file: str,
        instrucao: str,
        items: List[ExtractedItem],
    ) -> None:
        items_json = json.dumps([i.to_dict() for i in items], ensure_ascii=False)

        with self._Session() as session:
            existing = session.query(ExtractionJobModel).filter_by(job_id=job_id).first()
            if existing:
                existing.doc_hash = doc_hash
                existing.source_file = source_file
                existing.instrucao = instrucao
                existing.total = len(items)
                existing.items_json = items_json
                existing.created_at = datetime.utcnow()
            else:
                session.add(ExtractionJobModel(
                    job_id=job_id, doc_hash=doc_hash, source_file=source_file,
                    instrucao=instrucao, total=len(items), items_json=items_json,
                ))
            session.commit()
        logger.info("[Store] Job %s salvo no PostgreSQL (%d itens).", job_id, len(items))

        if items:
            ids, docs, metas = [], [], []
            for i, item in enumerate(items):
                ids.append(f"{job_id}_item_{i}")
                docs.append(item.resumo)
                metas.append({
                    "job_id": job_id,
                    "doc_hash": doc_hash,
                    "source_file": source_file,
                    "trecho_referencia": item.trecho_referencia,
                    "dados_json": json.dumps(item.dados, ensure_ascii=False),
                })
            self._items.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info("[Store] %d itens salvos no ChromaDB.", len(items))

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._Session() as session:
            rows = session.query(ExtractionJobModel).order_by(ExtractionJobModel.created_at.desc()).all()
            return [
                {
                    "job_id": r.job_id,
                    "source_file": r.source_file or "",
                    "instrucao": r.instrucao,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "total": r.total,
                }
                for r in rows
            ]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._Session() as session:
            row = session.query(ExtractionJobModel).filter_by(job_id=job_id).first()
            if not row:
                return None
            return {
                "job_id": row.job_id,
                "source_file": row.source_file or "",
                "instrucao": row.instrucao,
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "total": row.total,
                "items": json.loads(row.items_json or "[]"),
            }

    def delete_job(self, job_id: str) -> None:
        with self._Session() as session:
            session.query(ExtractionJobModel).filter_by(job_id=job_id).delete()
            session.commit()
        logger.info("[Store] Job %s removido do PostgreSQL.", job_id)

        try:
            result = self._items.get(where={"job_id": job_id}, include=["metadatas"])
            ids = result.get("ids", [])
            if ids:
                self._items.delete(ids=ids)
                logger.info("[Store] %d itens removidos do ChromaDB.", len(ids))
        except Exception as e:
            logger.warning("[Store] Erro ao limpar ChromaDB: %s", e)