"""
clause_library.py
Biblioteca de cláusulas aprovadas — Knowledge Management Interno.

Permite que advogados busquem precedentes de como cláusulas foram
redigidas em contratos anteriores da empresa.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Boolean, JSON,
    ForeignKey, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ApprovalStatus(Enum):
    """Status de aprovação de uma cláusula."""
    PENDING = "pending"           # Aguardando aprovação
    APPROVED = "approved"         # Aprovada pelo advogado
    REJECTED = "rejected"         # Rejeitada
    MODIFIED = "modified"         # Aprovada com modificações


class ClauseType(Enum):
    """Tipos de cláusulas comuns em contratos."""
    FORCE_MAJEURE = "forca_maior"
    TERMINATION = "rescisao"
    PENALTY = "multa"
    PRICE_ADJUSTMENT = "reajuste"
    CONFIDENTIALITY = "confidencialidade"
    LIABILITY = "responsabilidade"
    INTELLECTUAL_PROPERTY = "propriedade_intelectual"
    DISPUTE_RESOLUTION = "resolucao_disputas"
    PAYMENT = "pagamento"
    WARRANTY = "garantia"
    INDEMNITY = "indenizacao"
    CHANGE_OF_CONTROL = "mudanca_controle"
    ASSIGNMENT = "cessao"
    NON_COMPETE = "nao_concorrencia"
    SOLICITATION = "solicitacao"
    OTHER = "outros"


class ApprovedClause(Base):
    """
    Cláusula aprovada que faz parte da biblioteca de precedentes.

    Cada registro representa uma cláusula que foi extraída de um contrato
    e aprovada por um advogado como boa referência futura.
    """
    __tablename__ = "approved_clauses"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Metadados da cláusula
    clause_type = Column(String(50), nullable=False, index=True)  # ClauseType value
    title = Column(String(200))  # Título descritivo da cláusula

    # Conteúdo
    original_text = Column(Text, nullable=False)  # Texto original da cláusula
    standardized_text = Column(Text)  # Versão padronizada (se aprovada com mods)
    summary = Column(Text, nullable=False)  # Resumo do que a cláusula faz

    # Contexto
    job_id = Column(String(64), nullable=False, index=True)  # Job original
    doc_hash = Column(String(64), nullable=False, index=True)  # Documento original
    source_file = Column(Text)  # Nome do arquivo original

    # Tags e metadados adicionais
    tags = Column(JSON)  # Lista de tags personalizadas
    extra_metadata = Column(JSON)  # Metadados adicionais (setor, valor, etc.)

    # Aprovação
    approval_status = Column(String(20), default=ApprovalStatus.PENDING.value, index=True)
    approved_by = Column(String(100))  # Nome/email do aprovador
    approved_at = Column(DateTime)  # Data de aprovação
    notes = Column(Text)  # Notas do aprovador

    # Métricas de uso
    times_used = Column(Integer, default=0)  # Quantas vezes foi usada como referência
    last_used_at = Column(DateTime)  # Última vez que foi consultada

    # Rastreabilidade
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "clause_type": self.clause_type,
            "title": self.title or f"Cláusula {self.clause_type}",
            "original_text": self.original_text,
            "standardized_text": self.standardized_text,
            "summary": self.summary,
            "job_id": self.job_id,
            "source_file": self.source_file,
            "tags": self.tags or [],
            "extra_metadata": self.extra_metadata or {},
            "approval_status": self.approval_status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "notes": self.notes,
            "times_used": self.times_used,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ClauseLibrary:
    """
    Gerenciador da biblioteca de cláusulas aprovadas.

    Funcionalidades:
    - Adicionar cláusulas extraídas à biblioteca
    - Aprovar/rejeitar cláusulas
    - Buscar cláusulas similares por tipo
    - Buscar por TF-IDF (texto livre)
    - Gerar relatórios de uso
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Args:
            db_url: URL do banco PostgreSQL. Se None, usa a mesma configuração do Store.
        """
        if db_url is None:
            import os
            db_url = os.environ.get(
                "DATABASE_URL",
                "postgresql://extractor:extractor123@localhost:5432/extractor"
            )

        self._engine = create_engine(db_url, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)
        self._create_tables()

        logger.info("[ClauseLibrary] Inicializado com banco de dados.")

    def _create_tables(self):
        """Cria tabelas se não existirem."""
        Base.metadata.create_all(self._engine)

    # ─────────────────────────────────────────────────────────────────────
    # Adicionar cláusulas
    # ─────────────────────────────────────────────────────────────────────

    def add_from_extraction(
        self,
        job_id: str,
        doc_hash: str,
        source_file: str,
        items: List[Dict],
        auto_approve: bool = False,
    ) -> int:
        """
        Adiciona itens extraídos à biblioteca para aprovação.

        Args:
            job_id: ID do job de extração
            doc_hash: Hash do documento original
            source_file: Nome do arquivo
            items: Lista de itens extraídos (do job)
            auto_approve: Se True, aprova automaticamente (cuidado!)

        Returns:
            Número de cláusulas adicionadas
        """
        added = 0

        with self._Session() as session:
            for item in items:
                # Detecta tipo de cláusula automaticamente
                clause_type = self._detect_clause_type(item)

                clause = ApprovedClause(
                    clause_type=clause_type,
                    title=item.get("resumo", "")[:200],
                    original_text=item.get("trecho_referencia", ""),
                    summary=item.get("resumo", ""),
                    job_id=job_id,
                    doc_hash=doc_hash,
                    source_file=source_file,
                    tags=self._extract_tags(item),
                    extra_metadata=item.get("dados", {}),
                    approval_status=ApprovalStatus.APPROVED.value if auto_approve else ApprovalStatus.PENDING.value,
                )

                session.add(clause)
                added += 1

            session.commit()

        logger.info("[ClauseLibrary] %d cláusulas adicionadas do job %s", added, job_id)
        return added

    # ─────────────────────────────────────────────────────────────────────
    # Aprovação
    # ─────────────────────────────────────────────────────────────────────

    def approve_clause(
        self,
        clause_id: int,
        approved_by: str,
        notes: Optional[str] = None,
        standardized_text: Optional[str] = None,
    ) -> bool:
        """
        Aprova uma cláusula para a biblioteca.

        Args:
            clause_id: ID da cláusula
            approved_by: Quem aprovou
            notes: Notas do aprovador
            standardized_text: Texto padronizado (se houve modificações)

        Returns:
            True se aprovou com sucesso
        """
        with self._Session() as session:
            clause = session.query(ApprovedClause).filter_by(id=clause_id).first()
            if not clause:
                return False

            clause.approval_status = ApprovalStatus.APPROVED.value
            clause.approved_by = approved_by
            clause.approved_at = datetime.utcnow()
            clause.notes = notes
            if standardized_text:
                clause.standardized_text = standardized_text

            session.commit()

        logger.info("[ClauseLibrary] Cláusula %d aprovada por %s", clause_id, approved_by)
        return True

    def reject_clause(self, clause_id: int, rejected_by: str, notes: Optional[str] = None) -> bool:
        """Rejeita uma cláusula."""
        with self._Session() as session:
            clause = session.query(ApprovedClause).filter_by(id=clause_id).first()
            if not clause:
                return False

            clause.approval_status = ApprovalStatus.REJECTED.value
            clause.notes = notes

            session.commit()

        logger.info("[ClauseLibrary] Cláusula %d rejeitada por %s", clause_id, rejected_by)
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Busca
    # ─────────────────────────────────────────────────────────────────────

    def search_by_type(self, clause_type: str, approved_only: bool = True) -> List[Dict]:
        """
        Busca cláusulas por tipo.

        Args:
            clause_type: Tipo da cláusula (ex: 'forca_maior', 'multa')
            approved_only: Se True, retorna apenas aprovadas

        Returns:
            Lista de cláusulas
        """
        with self._Session() as session:
            query = session.query(ApprovedClause).filter_by(clause_type=clause_type)

            if approved_only:
                query = query.filter_by(approval_status=ApprovalStatus.APPROVED.value)

            results = query.order_by(ApprovedClause.times_used.desc()).all()

            # Atualiza métricas de uso
            for clause in results:
                clause.times_used += 1
                clause.last_used_at = datetime.utcnow()

            session.commit()

            return [clause.to_dict() for clause in results]

    def search_tfidf(self, query: str, clause_type: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """
        Busca cláusulas usando TF-IDF (busca semântica).

        Args:
            query: Texto da busca (ex: "como já redigimos cláusula de força maior")
            clause_type: Filtrar por tipo (opcional)
            top_k: Quantidade de resultados

        Returns:
            Lista de cláusulas ordenadas por relevância
        """
        from tfidf_utils import tfidf_rank
        import numpy as np

        with self._Session() as session:
            # Busca cláusulas aprovadas
            query_db = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.APPROVED.value
            )

            if clause_type:
                query_db = query_db.filter_by(clause_type=clause_type)

            clauses = query_db.all()

            if not clauses:
                return []

            # Prepara textos para busca
            texts = []
            for clause in clauses:
                # Combina título + resumo + tags para busca rica
                text = f"{clause.title or ''} {clause.summary or ''}"
                if clause.tags:
                    text += " " + " ".join(clause.tags)
                texts.append(text)

            # TF-IDF
            ranked = tfidf_rank(query, texts, top_k=len(texts))

            # Atualiza métricas
            for score, _ in ranked[:top_k]:
                idx = texts.index(ranked[ranked.index((score, _))][1]) if ranked else -1
                if idx >= 0 and idx < len(clauses):
                    clauses[idx].times_used += 1
                    clauses[idx].last_used_at = datetime.utcnow()

            session.commit()

            # Retorna resultados
            results = []
            for score, text in ranked[:top_k]:
                # Encontra a cláusula correspondente
                for clause in clauses:
                    clause_text = f"{clause.title or ''} {clause.summary or ''}"
                    if clause.tags:
                        clause_text += " " + " ".join(clause.tags)

                    if clause_text == text:
                        results.append({**clause.to_dict(), "relevance_score": float(score)})
                        break

            return results

    def search_similar(self, clause_id: int, top_k: int = 5) -> List[Dict]:
        """
        Busca cláusulas similares a uma específica.

        Útil para: "Quais outras formas já redigimos esta cláusula?"
        """
        with self._Session() as session:
            reference = session.query(ApprovedClause).filter_by(id=clause_id).first()
            if not reference:
                return []

            # Busca usando o resumo como query
            return self.search_tfidf(
                reference.summary,
                clause_type=reference.clause_type,
                top_k=top_k + 1  # +1 para excluir a própria
            )[:top_k]

    # ─────────────────────────────────────────────────────────────────────
    # Relatórios
    # ─────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Estatísticas da biblioteca."""
        with self._Session() as session:
            total = session.query(ApprovedClause).count()
            approved = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.APPROVED.value
            ).count()
            pending = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.PENDING.value
            ).count()

            # Por tipo
            by_type = {}
            for type_enum in ClauseType:
                count = session.query(ApprovedClause).filter_by(
                    clause_type=type_enum.value,
                    approval_status=ApprovalStatus.APPROVED.value
                ).count()
                if count > 0:
                    by_type[type_enum.value] = count

            # Mais usadas
            most_used = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.APPROVED.value
            ).order_by(ApprovedClause.times_used.desc()).limit(5).all()

            return {
                "total_clauses": total,
                "approved": approved,
                "pending": pending,
                "by_type": by_type,
                "most_used": [c.to_dict() for c in most_used],
            }

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _detect_clause_type(self, item: Dict) -> str:
        """Detecta automaticamente o tipo de cláusula."""
        resumo = item.get("resumo", "").lower()
        dados = item.get("dados", {})

        # Palavras-chave para cada tipo
        keywords = {
            ClauseType.FORCE_MAJEURE.value: ["força maior", "caso fortuito", "evento imprevisto"],
            ClauseType.TERMINATION.value: ["rescisão", "término", "encerramento", "cancelamento"],
            ClauseType.PENALTY.value: ["multa", "penalidade", "sanção", "mora"],
            ClauseType.PRICE_ADJUSTMENT.value: ["reajuste", "correção", "indexador", "inflação"],
            ClauseType.CONFIDENTIALITY.value: ["confidencial", "sigilo", "informação confidencial"],
            ClauseType.LIABILITY.value: ["responsabilidade", "limitação", "danos"],
            ClauseType.INTELLECTUAL_PROPERTY.value: ["propriedade intelectual", "direitos autorais"],
            ClauseType.DISPUTE_RESOLUTION.value: ["disputa", "controvérsia", "arbitragem", "judiciário"],
            ClauseType.PAYMENT_Terms.value: ["pagamento", "parcela", "mensalidade", "fatura"],
        }

        # Busca por palavras-chave
        for type_value, keys in keywords.items():
            if any(key in resumo for key in keys):
                return type_value

        # Padrão
        return ClauseType.OTHER.value

    def _extract_tags(self, item: Dict) -> List[str]:
        """Extrai tags do item extraído."""
        tags = []

        # Tira do resumo
        resumo = item.get("resumo", "")
        if "%" in resumo:
            tags.append("percentual")
        if "R$" in resumo or "reais" in resumo.lower():
            tags.append("monetario")
        if "diário" in resumo.lower() or "dia" in resumo.lower():
            tags.append("prazo_diario")
        if "mensal" in resumo.lower():
            tags.append("prazo_mensal")

        return tags


# ─────────────────────────────────────────────────────────────────────────
# API REST endpoints (para integrar com app.py)
# ─────────────────────────────────────────────────────────────────────────

def setup_clause_routes(app):
    """Configura rotas Flask para a biblioteca de cláusulas."""

    @app.route("/api/clauses", methods=["GET"])
    def list_clauses():
        """Lista cláusulas com filtros."""
        from flask import request

        clause_type = request.args.get("type")
        approved_only = request.args.get("approved", "true").lower() == "true"

        library = ClauseLibrary()

        if clause_type:
            results = library.search_by_type(clause_type, approved_only)
        else:
            # Lista todas aprovadas
            results = library.search_by_type("", approved_only)

        return {"results": results, "total": len(results)}

    @app.route("/api/clauses/search", methods=["POST"])
    def search_clauses():
        """Busca cláusulas por TF-IDF."""
        from flask import request

        data = request.json
        query = data.get("query", "")
        clause_type = data.get("type")
        top_k = data.get("top_k", 10)

        if not query:
            return {"error": "Query é obrigatória"}, 400

        library = ClauseLibrary()
        results = library.search_tfidf(query, clause_type, top_k)

        return {"results": results, "total": len(results)}

    @app.route("/api/clauses/<int:clause_id>/approve", methods=["POST"])
    def approve_clause(clause_id):
        """Aprova uma cláusula."""
        from flask import request

        data = request.json
        approved_by = data.get("approved_by")
        notes = data.get("notes")
        standardized_text = data.get("standardized_text")

        if not approved_by:
            return {"error": "approved_by é obrigatório"}, 400

        library = ClauseLibrary()
        success = library.approve_clause(clause_id, approved_by, notes, standardized_text)

        if not success:
            return {"error": "Cláusula não encontrada"}, 404

        return {"ok": True}

    @app.route("/api/clauses/<int:clause_id>/reject", methods=["POST"])
    def reject_clause(clause_id):
        """Rejeita uma cláusula."""
        from flask import request

        data = request.json
        rejected_by = data.get("rejected_by")
        notes = data.get("notes")

        if not rejected_by:
            return {"error": "rejected_by é obrigatório"}, 400

        library = ClauseLibrary()
        success = library.reject_clause(clause_id, rejected_by, notes)

        if not success:
            return {"error": "Cláusula não encontrada"}, 404

        return {"ok": True}

    @app.route("/api/clauses/stats", methods=["GET"])
    def clause_stats():
        """Estatísticas da biblioteca."""
        library = ClauseLibrary()
        return library.get_stats()

    @app.route("/api/jobs/<job_id>/add-to-library", methods=["POST"])
    def add_job_to_library(job_id):
        """Adiciona itens de um job à biblioteca."""
        from flask import request
        from store import Store

        data = request.json
        auto_approve = data.get("auto_approve", False)

        store = Store()
        job = store.get_job(job_id)

        if not job:
            return {"error": "Job não encontrado"}, 404

        library = ClauseLibrary()
        added = library.add_from_extraction(
            job_id=job_id,
            doc_hash="",  # TODO: pegar do job
            source_file=job.get("source_file", ""),
            items=job.get("items", []),
            auto_approve=auto_approve,
        )

        return {"ok": True, "added": added}
