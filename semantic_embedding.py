"""
semantic_embedding.py
Implementação local de embeddings semânticos sem usar APIs externas.

Opções disponíveis:
1. sentence-transformers (modelos pré-treinados locais)
2. TF-IDF híbrido com features semânticas
3. Word embeddings locais (Word2Vec-style)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SemanticEmbeddingST:
    """
    Embeddings semânticos usando sentence-transformers local.

    Modelos recomendados:
    - 'paraphrase-multilingual-MiniLM-L12-v2' (50MB, multilíngue)
    - 'distiluse-base-multilingual-cased-v2' (250MB, melhor precisão)
    - 'all-MiniLM-L6-v2' (23MB, inglês apenas)
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model_name = model_name
        self._model = None
        self._lazy_load()

    def _lazy_load(self):
        """Carrega o modelo apenas quando necessário (lazy loading)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Execute: pip install sentence-transformers\n"
                "Ou use: pip install sentence-transformers[flask]"
            )

        cache_dir = Path("./embedding_models")
        cache_dir.mkdir(exist_ok=True)

        try:
            logger.info(f"[SemanticEmbedding] Carregando modelo {self._model_name}...")
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=str(cache_dir),
            )
            logger.info(f"[SemanticEmbedding] Modelo carregado com sucesso.")
        except Exception as e:
            logger.error(f"[SemanticEmbedding] Erro ao carregar modelo: {e}")
            raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings semânticos para uma lista de textos.

        Args:
            texts: Lista de strings para embeddar

        Returns:
            Lista de vetores (cada vetor é uma lista de floats)
        """
        if not texts:
            return [[]]

        self._lazy_load()

        # Gera embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Normaliza L2 (vetores unitários)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        return embeddings.tolist()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Interface compatível com ChromaDB e outros frameworks."""
        return self.embed(texts)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade de cosseno entre dois textos.

        Returns:
            Valor entre 0 (nada similar) e 1 (idêntico)
        """
        emb1 = self.embed([text1])[0]
        emb2 = self.embed([text2])[0]

        dot_product = np.dot(emb1, emb2)
        return float(dot_product)


class HybridSemanticEmbedding:
    """
    Embedding híbrido que combina TF-IDF com características semânticas
    derivadas do próprio corpus (sem modelos pesados).

    Estratégias usadas:
    - TF-IDF (bag-of-words)
    - n-gramas de caracteres (captura padrões de sufixos/prefixos)
    - Features estatísticas (densidade de números, etc.)
    - Coocorrência de termos (similaridade de contexto)
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

        # TF-IDF de palavras
        self._word_tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            norm='l2',
        )

        # TF-IDF de caracteres (captura padrões morfológicos)
        self._char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            min_df=1,
            norm='l2',
        )

        self._fitted = False

    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extrai features adicionais."""
        features = []
        for text in texts:
            if not text:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            words = text.split()
            if not words:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            # Densidade de números
            num_digits = sum(c.isdigit() for c in text)
            digit_density = num_digits / len(text)

            # Comprimento médio das palavras
            avg_word_len = sum(len(w) for w in words) / len(words)

            # Razão de maiúsculas
            upper_ratio = sum(1 for c in text if c.isupper()) / len(text)

            # Densidade de pontuação jurídica
            legal_punct = text.count('§') + text.count('°') + text.count('º') + text.count('ª')
            punct_density = legal_punct / len(text) if text else 0

            features.append([digit_density, avg_word_len / 20, upper_ratio, punct_density])

        return np.array(features)

    def fit(self, texts: List[str]) -> None:
        """Fit nos dados de treino."""
        if not texts:
            raise ValueError("_texts vazios para fit")

        self._word_tfidf.fit(texts)
        self._char_tfidf.fit(texts)
        self._fitted = True

        logger.info("[HybridEmbedding] Fit realizado em %d textos.", len(texts))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings híbridos."""
        if not texts:
            return [[]]

        if not self._fitted:
            self.fit(texts)

        # TF-IDF de palavras
        word_vecs = self._word_tfidf.transform(texts).toarray()

        # TF-IDF de caracteres
        char_vecs = self._char_tfidf.transform(texts).toarray()

        # Features adicionais
        extra_features = self._extract_features(texts)

        # Combina tudo
        combined = np.hstack([word_vecs, char_vecs, extra_features])

        # Normaliza L2
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1
        combined = combined / norms

        return combined.tolist()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


class Word2VecStyleEmbedding:
    """
    Embeddings estilo Word2Vec treinados no próprio corpus.

    Simples mas eficaz para capturar relações de palavras
    em um domínio específico (ex: jurídico).
    """

    def __init__(self, vector_size: int = 100, window: int = 5):
        """
        Args:
            vector_size: Dimensão dos embeddings (padrão: 100)
            window: Janela de contexto para treino
        """
        self._vector_size = vector_size
        self._window = window
        self._model = None
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        """Treina o modelo no corpus fornecido."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("Execute: pip install gensim")

        # Tokeniza textos
        sentences = [text.lower().split() for text in texts]

        # Treina Word2Vec
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self._vector_size,
            window=self._window,
            min_count=1,
            workers=4,
        )

        self._fitted = True
        logger.info("[Word2VecEmbedding] Modelo treinado em %d textos.", len(texts))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings (média dos vetores das palavras)."""
        if not self._fitted:
            self.fit(texts)

        if not self._model:
            raise RuntimeError("Modelo não treinado")

        embeddings = []
        for text in texts:
            words = text.lower().split()
            if not words:
                embeddings.append([0.0] * self._vector_size)
                continue

            # Média dos vetores das palavras
            word_vecs = []
            for word in words:
                if word in self._model.wv:
                    word_vecs.append(self._model.wv[word])

            if word_vecs:
                vec = np.mean(word_vecs, axis=0)
            else:
                vec = np.zeros(self._vector_size)

            # Normaliza
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            embeddings.append(vec.tolist())

        return embeddings

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


def create_semantic_embedding(
    method: str = "sentence-transformers",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> object:
    """
    Factory para criar embeddings semânticos.

    Args:
        method: Método a usar ('sentence-transformers', 'hybrid', 'word2vec')
        model_name: Nome do modelo (para sentence-transformers)

    Returns:
        Instância de embedding semântico
    """
    if method == "sentence-transformers":
        return SemanticEmbeddingST(model_name=model_name)
    elif method == "hybrid":
        return HybridSemanticEmbedding()
    elif method == "word2vec":
        return Word2VecStyleEmbedding()
    else:
        raise ValueError(f"Método desconhecido: {method}")


# ─────────────────────────────────────────────────────────────
# Teste rápido
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Teste dos diferentes métodos
    texts = [
        "O inquilino deve pagar aluguel mensalmente",
        "Locatário obrigado a pagar mensalidade",
        "O carro é vermelho e rápido",
    ]

    print("=" * 60)
    print("Teste de Embeddings Semânticos Locais")
    print("=" * 60)

    # Método 1: sentence-transformers
    print("\n[1] sentence-transformers (multilingual)")
    try:
        st_emb = create_semantic_embedding("sentence-transformers")
        embeddings = st_emb.embed(texts)

        # Similaridade entre os dois primeiros (devem ser similares)
        sim = st_emb.similarity(texts[0], texts[1])
        print(f"Similaridade '{texts[0][:30]}...' vs '{texts[1][:30]}...': {sim:.3f}")

        sim2 = st_emb.similarity(texts[0], texts[2])
        print(f"Similaridade '{texts[0][:30]}...' vs '{texts[2][:30]}...': {sim2:.3f}")
    except ImportError as e:
        print(f"Não disponível: {e}")

    # Método 2: Hybrid
    print("\n[2] Hybrid (TF-IDF + features)")
    hybrid_emb = create_semantic_embedding("hybrid")
    embeddings = hybrid_emb.embed(texts)
    print(f"Dimensionalidade: {len(embeddings[0])}")

    # Método 3: Word2Vec
    print("\n[3] Word2Vec-style")
    try:
        w2v_emb = create_semantic_embedding("word2vec")
        embeddings = w2v_emb.embed(texts)
        print(f"Dimensionalidade: {len(embeddings[0])}")
    except ImportError as e:
        print(f"Não disponível: {e}")
