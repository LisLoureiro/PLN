"""
vectorizer.py
Vetoriza chunks de texto e recupera os mais relevantes via TF-IDF.

Diferente da versão CPR-F original, aqui NÃO existe uma query padrão fixa:
a própria instrução do usuário (o texto livre descrevendo o que ele quer
extrair) é usada como query, garantindo que os trechos mais relevantes
para o pedido específico sejam selecionados — qualquer que seja o domínio
do documento (contrato, laudo, edital, etc.).
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("Execute: pip install scikit-learn")


class Vectorizer:
    """
    Indexa chunks com TF-IDF e recupera os mais relevantes por cosseno
    em relação a uma query (normalmente a instrução do usuário).
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self._chunks: List[str] = []
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = None

    def index_chunks(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("Lista de chunks vazia.")
        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform(chunks)
        logger.info("[Vectorizer] %d chunks indexados.", len(chunks))

    def search(self, query: str) -> List[str]:
        """
        Retorna os chunks mais relevantes para a query fornecida
        (tipicamente a instrução de extração do usuário).
        """
        if not query or not query.strip():
            raise ValueError("Query vazia — informe a instrução de extração.")
        if self._matrix is None:
            raise RuntimeError("Chame index_chunks() antes de search().")

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][: self.top_k]

        results = [self._chunks[i] for i in top_idx if sims[i] > 0]

        # Log dos scores dos top chunks e conteúdo dos chunks
        logger.info("[Vectorizer] Top 5 scores: %s", [(i, float(sims[i])) for i in top_idx[:5]])
        logger.info("[DEBUG] CONTEÚDO DOS CHUNKS SELECIONADOS:")
        for idx, chunk_idx in enumerate(top_idx[:5]):
            if chunk_idx < len(self._chunks):
                chunk_content = self._chunks[chunk_idx]
                logger.info("[DEBUG] Chunk #%d (score %.4f): %s", chunk_idx, float(sims[chunk_idx]), chunk_content[:200])
        logger.info("[DEBUG] TOTAL DE CARACTERES NOS CHUNKS SELECIONADOS: %d", sum(len(c) for c in [self._chunks[i] for i in top_idx if sims[i] > 0]))

        # Fallback: se a query não bateu com nada (documento muito diferente
        # do vocabulário da instrução), devolve os primeiros chunks para não
        # travar o pipeline — o Claude decide se há algo relevante ou não.
        if not results:
            logger.warning("[Vectorizer] Nenhum chunk com score > 0 — tentando busca expandida.")
            # Expande a query com termos relacionados para endereçamento
            expanded_query = self._expand_query(query)
            logger.info("[Vectorizer] Query expandida: '%s' → '%s'", query, expanded_query)
            q_vec_expanded = self._vectorizer.transform([expanded_query])
            sims_expanded = cosine_similarity(q_vec_expanded, self._matrix).flatten()
            top_idx_expanded = np.argsort(sims_expanded)[::-1][: self.top_k]
            results = [self._chunks[i] for i in top_idx_expanded if sims_expanded[i] > 0]
            logger.info("[Vectorizer] Top 5 scores expandidos: %s", [(i, float(sims_expanded[i])) for i in top_idx_expanded[:5]])

        if not results:
            logger.warning("[Vectorizer] Ainda sem resultados — usando fallback pelos primeiros chunks.")
            results = self._chunks[: min(self.top_k, len(self._chunks))]
            logger.info("[Vectorizer] Fallback: retornando primeiros %d chunks", len(results))

        logger.info("[Vectorizer] %d trechos relevantes (query: %d chars).", len(results), len(query))
        return results

    def _expand_query(self, query: str) -> str:
        """Expande a query com termos relacionados para melhorar a busca."""
        query_lower = query.lower()
        expansions = {
            "endereçamento": "endereço parte contratada domicílio sede local",
            "endereco": "endereço parte contratada domicílio sede local residência",
            "advogado": "advogado procurador representante legal escritório",
            "parte": "parte contratada contratante signatário",
            "pagamento": "pagamento valor preço remuneração custo honorário",
            "prazo": "prazo data prazo limite vencimento duração período",
            "honorários": "honorários sucumbência verba honorária honorários advocatícios advocatício percentual",
            "honorarios": "honorários sucumbência verba honorária honorários advocatícios advocatício percentual",
        }
        for termo, relacionados in expansions.items():
            if termo in query_lower:
                return f"{query} {relacionados}"
        return query
