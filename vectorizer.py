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

    def __init__(self, top_k: int = 25):
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

        # Fallback: se a query não bateu com nada (documento muito diferente
        # do vocabulário da instrução), devolve os primeiros chunks para não
        # travar o pipeline — o Claude decide se há algo relevante ou não.
        if not results:
            logger.warning("[Vectorizer] Nenhum chunk com score > 0 — usando fallback pelos primeiros chunks.")
            results = self._chunks[: min(self.top_k, len(self._chunks))]

        logger.info("[Vectorizer] %d trechos relevantes (query: %d chars).", len(results), len(query))
        return results
