"""
tfidf_utils.py
Módulo compartilhado para TF-IDF e busca de texto.
Usado tanto pelo app principal quanto pelo MCP server.
"""
import logging
import math
import re
from collections import defaultdict
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

_STOPWORDS_PT = {
    "a", "ao", "aos", "as", "da", "das", "de", "do", "dos", "e", "em", "na", "nas",
    "no", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "que", "se", "um", "uma", "com", "é", "ser", "ter", "foi", "são", "mais",
    "como", "mas", "seu", "sua", "seus", "suas", "não", "este", "esta", "esse",
    "essa", "estes", "estas", "esses", "essas", "qual", "quando", "onde", "cada",
    "já", "até", "também", "ainda", "sobre", "entre", "após", "antes", "durante",
}


def tokenize(text: str) -> List[str]:
    """Tokeniza texto em português, removendo stopwords."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t not in _STOPWORDS_PT and len(t) > 2]


def chunk_markdown(markdown: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Divide markdown em chunks com overlap."""
    text = markdown.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def tfidf_rank_pure(query: str, chunks: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
    """
    Implementação pura de TF-IDF sem dependências externas.
    Retorna lista de (score, chunk) ordenada por relevância.
    """
    if not chunks:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return [(0.0, c) for c in chunks[:top_k]]

    chunk_tfs = []
    doc_freq: defaultdict = defaultdict(int)

    # Calcula TF para cada chunk e DF para todos os documentos
    for chunk in chunks:
        tokens = tokenize(chunk)
        tf: defaultdict = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = sum(tf.values()) or 1
        for t in tf:
            tf[t] = tf[t] / total
        chunk_tfs.append(dict(tf))
        for t in set(tokens):
            doc_freq[t] += 1

    n_docs = len(chunks)

    def idf(term: str) -> float:
        df = doc_freq.get(term, 0)
        return math.log((n_docs + 1) / (df + 1)) + 1

    scored = []
    for i, chunk in enumerate(chunks):
        tf = chunk_tfs[i]
        score = sum(tf.get(t, 0.0) * idf(t) for t in query_tokens)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for s in scored[:top_k] if s[0] > 0]


def tfidf_rank_sklearn(query: str, chunks: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
    """
    TF-IDF usando scikit-learn (mais preciso, mas requer dependência).
    Retorna lista de (score, chunk) ordenada por relevância.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("[TF-IDF] scikit-learn não disponível, usando implementação pura")
        return tfidf_rank_pure(query, chunks, top_k)

    if not chunks:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(chunks)

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()

    scored = [(float(sims[i]), chunks[i]) for i in range(len(chunks))]
    scored.sort(key=lambda x: x[0], reverse=True)

    return [s for s in scored[:top_k] if s[0] > 0]


def tfidf_rank(query: str, chunks: List[str], top_k: int = 5, use_sklearn: bool = True) -> List[Tuple[float, str]]:
    """
    Função unificada de TF-IDF.
    Por padrão usa scikit-learn se disponível.
    """
    if use_sklearn:
        return tfidf_rank_sklearn(query, chunks, top_k)
    return tfidf_rank_pure(query, chunks, top_k)


def highlight_text(text: str, query: str) -> str:
    """Marca tokens da query no texto com **."""
    for token in tokenize(query):
        if len(token) > 3:
            text = re.sub(rf"\b({re.escape(token)})\b", r"**\1**", text, flags=re.IGNORECASE)
    return text
