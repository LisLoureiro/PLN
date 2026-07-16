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
    scan_id     = Column(String(64), nullable=True, index=True)  # Link para FullScanModel


class FullScanModel(Base):
    """Varredura completa: um documento scanneado contra múltiplos tipos de cláusula."""
    __tablename__ = "full_scans"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    scan_id         = Column(String(64), unique=True, nullable=False, index=True)
    doc_hash        = Column(String(64), nullable=False, index=True)
    status          = Column(String(20), nullable=False, default="running")  # running, completed, error
    total_types     = Column(Integer, nullable=False)
    processed_types = Column(Integer, default=0)
    results_json    = Column(Text, default="{}")  # Resultados consolidados por tipo
    created_at      = Column(DateTime, default=datetime.utcnow)
    completed_at    = Column(DateTime, nullable=True)
    error_message   = Column(Text, nullable=True)
    clause_types    = Column(Text, default="[]")  # Lista de tipos incluídos no scan
    
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
        summaries = [item.get("resumo", "") for item in all_items]
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

    # ──────────────────────────────────────────────
    # Full Scan CRUD
    # ──────────────────────────────────────────────

    def create_full_scan(
        self,
        scan_id: str,
        doc_hash: str,
        clause_types: List[str],
        total_types: int,
    ) -> None:
        """Cria um novo registro de full scan com status 'running'."""
        with self._Session() as session:
            existing = session.query(FullScanModel).filter_by(scan_id=scan_id).first()
            if existing:
                logger.warning("[Store] Full scan %s já existe, atualizando status para running.", scan_id)
                existing.status = "running"
                existing.processed_types = 0
                existing.results_json = "{}"
                existing.error_message = None
                existing.completed_at = None
            else:
                session.add(FullScanModel(
                    scan_id=scan_id,
                    doc_hash=doc_hash,
                    clause_types=json.dumps(clause_types, ensure_ascii=False),
                    total_types=total_types,
                    processed_types=0,
                    status="running",
                ))
                logger.info("[Store] Full scan %s criado (%d tipos).", scan_id, total_types)
            session.commit()

    def update_full_scan_progress(
        self,
        scan_id: str,
        processed_types: int,
        results: Dict[str, List[Dict]],
    ) -> None:
        """Atualiza progresso e resultados parciais do scan."""
        with self._Session() as session:
            scan = session.query(FullScanModel).filter_by(scan_id=scan_id).first()
            if not scan:
                logger.warning("[Store] Full scan %s não encontrado para atualizar progresso.", scan_id)
                return
            scan.processed_types = processed_types
            scan.results_json = json.dumps(results, ensure_ascii=False)
            session.commit()

    def complete_full_scan(
        self,
        scan_id: str,
        final_results: Dict[str, List[Dict]],
        error: Optional[str] = None,
    ) -> None:
        """Marca scan como completado (ou com erro)."""
        with self._Session() as session:
            scan = session.query(FullScanModel).filter_by(scan_id=scan_id).first()
            if not scan:
                logger.warning("[Store] Full scan %s não encontrado para completar.", scan_id)
                return
            if error:
                scan.status = "error"
                scan.error_message = error
                logger.error("[Store] Full scan %s terminou com erro: %s", scan_id, error)
            else:
                scan.status = "completed"
                scan.processed_types = scan.total_types
                logger.info("[Store] Full scan %s completado com sucesso (%d tipos).", scan_id, scan.total_types)
            scan.results_json = json.dumps(final_results, ensure_ascii=False)
            scan.completed_at = datetime.utcnow()
            session.commit()

    def get_full_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retorna dados completos de um full scan."""
        with self._Session() as session:
            row = session.query(FullScanModel).filter_by(scan_id=scan_id).first()
            if not row:
                return None
            return {
                "scan_id": row.scan_id,
                "doc_hash": row.doc_hash,
                "status": row.status,
                "total_types": row.total_types,
                "processed_types": row.processed_types,
                "results": json.loads(row.results_json or "{}"),
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                "error_message": row.error_message,
                "clause_types": json.loads(row.clause_types or "[]"),
            }

    def list_full_scans(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lista os full scans mais recentes."""
        with self._Session() as session:
            rows = session.query(FullScanModel).order_by(FullScanModel.created_at.desc()).limit(limit).all()
            return [
                {
                    "scan_id": r.scan_id,
                    "doc_hash": r.doc_hash,
                    "status": r.status,
                    "total_types": r.total_types,
                    "processed_types": r.processed_types,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in rows
            ]