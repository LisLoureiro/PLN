"""
store.py
Persiste documentos e jobs de extração no PostgreSQL.
Busca de itens feita via TF-IDF local (scikit-learn).
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    """Camada de persistência: PostgreSQL (documentos e jobs de extração)."""

    def __init__(self):
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://extractor:extractor123@localhost:5432/extractor",
        )
        self._engine = create_engine(db_url, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)
        self._create_tables_safely()
        logger.info("[Store] PostgreSQL conectado.")

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

    def search_items_semantic(self, query: str, top_k: int = 10, method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Busca itens extraídos usando embeddings semânticos.

        Args:
            query: Texto da busca
            top_k: Quantidade de resultados
            method: 'sentence-transformers', 'hybrid', ou 'word2vec'

        Returns:
            Lista de itens relevantes ordenados por similaridade
        """
        from semantic_embedding import create_semantic_embedding
        import numpy as np

        jobs = self.list_jobs()
        if not jobs:
            return []

        # Coleta todos os itens
        all_items = []
        for job in jobs:
            job_detail = self.get_job(job["job_id"])
            if job_detail:
                for item in job_detail.get("items", []):
                    all_items.append({
                        "job_id": job["job_id"],
                        "source_file": job.get("source_file", ""),
                        "instrucao": job.get("instrucao", ""),
                        "resumo": item.get("resumo", ""),
                        "trecho_referencia": item.get("trecho_referencia", ""),
                        "dados": item.get("dados", {}),
                    })

        if not all_items:
            return []

        # Cria embedding
        embedder = create_semantic_embedding(method=method)

        # Embeddings dos resumos
        summaries = [item["resumo"] for item in all_items]
        embeddings = np.array(embedder.embed(summaries))

        # Embedding da query
        query_emb = np.array(embedder.embed([query])[0])

        # Similaridade de cosseno
        similarities = np.dot(embeddings, query_emb)

        # Ordena
        ranked = sorted(zip(similarities, all_items), key=lambda x: x[0], reverse=True)

        # Filtra por threshold e retorna top_k
        threshold = 0.3
        results = [item for sim, item in ranked if sim > threshold][:top_k]

        logger.info("[Store] Semantic search (%s): %d resultados para '%s'", method, len(results), query[:50])

        return results

    def search_items_tfidf(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Busca itens extraídos usando TF-IDF sobre os resumos.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("Execute: pip install scikit-learn")

        jobs = self.list_jobs()
        if not jobs:
            return []

        # Coleta todos os resumos de todos os jobs
        all_items = []
        for job in jobs:
            job_detail = self.get_job(job["job_id"])
            if job_detail:
                for item in job_detail.get("items", []):
                    all_items.append({
                        "job_id": job["job_id"],
                        "source_file": job.get("source_file", ""),
                        "instrucao": job.get("instrucao", ""),
                        "resumo": item.get("resumo", ""),
                        "trecho_referencia": item.get("trecho_referencia", ""),
                        "dados": item.get("dados", {}),
                    })

        if not all_items:
            return []

        # TF-IDF sobre os resumos
        summaries = [item["resumo"] for item in all_items]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(summaries)

        # Busca
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]

        results = [all_items[i] for i in top_idx if sims[i] > 0]
        logger.info("[Store] TF-IDF search: %d resultados para query '%s'", len(results), query[:50])

        return results