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

# Reaproveita o MESMO tokenizador (com stopwords em português) usado em
# tfidf_utils.py. Antes, este arquivo usava a tokenização padrão do
# scikit-learn (sem remover "de", "do", "para" etc.), o que dilui a
# precisão em instruções curtas — ex.: "REPETIÇÃO DE INDÉBITO" tem só 2
# palavras de conteúdo real; manter "de" no vocabulário só acrescenta
# ruído. Unificar aqui também resolve a duplicação de implementações de
# TF-IDF já apontada em OTIMIZACOES.md (item 5).
from tfidf_utils import tokenize as _tokenize_pt


class Vectorizer:
    """
    Indexa chunks com TF-IDF e recupera os mais relevantes por cosseno
    em relação a uma query (normalmente a instrução do usuário).
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self._chunks: List[str] = []
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize_pt,
            token_pattern=None,   # obrigatório ao usar tokenizer customizado (evita warning/conflito no sklearn)
            lowercase=False,      # _tokenize_pt já lowercasa internamente
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = None
        self._cached_chunks_hash = None  # Cache para evitar reindexação em full-scan

    def index_chunks(self, chunks: List[str]) -> None:
        """
        Indexa chunks com TF-IDF. Se os mesmos chunks forem indexados novamente
        (ex.: em full-scan com múltiplos tipos), reusa a matriz cacheada.
        """
        if not chunks:
            raise ValueError("Lista de chunks vazia.")

        # Calcula hash dos chunks para detectar reutilização
        chunks_hash = hash(tuple(chunks))

        if chunks_hash == self._cached_chunks_hash and self._matrix is not None:
            logger.info("[Vectorizer] Reusando matriz TF-IDF cacheada (%d chunks).", len(chunks))
            self._chunks = chunks  # Atualiza referência, mas mantém matriz
            return

        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform(chunks)
        self._cached_chunks_hash = chunks_hash
        logger.info("[Vectorizer] %d chunks indexados (matriz nova).", len(chunks))

    def search(self, query: str) -> List[str]:
        """
        Retorna os chunks mais relevantes para a query fornecida
        (tipicamente a instrução de extração do usuário).
        """
        if not query or not query.strip():
            raise ValueError("Query vazia — informe a instrução de extração.")
        if self._matrix is None:
            raise RuntimeError("Chame index_chunks() antes de search().")

        # A expansão é aplicada SEMPRE (não só quando o score é zero).
        # _expand_query só adiciona termos quando reconhece um termo-gatilho
        # na própria query, então é seguro: uma instrução sem nenhum termo
        # conhecido volta inalterada. Isso corrige o caso em que a busca
        # original já retornava ALGUM resultado (score > 0), mas não o
        # trecho certo — a expansão antes só entrava em ação quando a
        # busca dava ZERO resultados, o que raramente é o problema real.
        search_query = self._expand_query(query)
        if search_query != query:
            logger.info("[Vectorizer] Query expandida: '%s' → '%s'", query, search_query)

        q_vec = self._vectorizer.transform([search_query])
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

        if not results:
            logger.warning("[Vectorizer] Ainda sem resultados — usando fallback pelos primeiros chunks.")
            results = self._chunks[: min(self.top_k, len(self._chunks))]
            logger.info("[Vectorizer] Fallback: retornando primeiros %d chunks", len(results))

        logger.info("[Vectorizer] %d trechos relevantes (query: %d chars).", len(results), len(query))
        return results

    def _expand_query(self, query: str) -> str:
        """
        Expande a query com termos relacionados para melhorar a busca,
        quando um termo-gatilho conhecido aparece na instrução do usuário.
        Só adiciona termos quando encontra um gatilho — instruções sem
        nenhum termo reconhecido voltam inalteradas.
        """
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
            # ── Termos tributários/cíveis comuns em petições (adicionados
            # após instruções curtas como "REPETIÇÃO DE INDÉBITO" não
            # priorizarem o trecho certo — ver conversa sobre a extração
            # da ação de isenção de IR) ──
            "indébito": "repetição indébito restituição tributo pago indevidamente cobrança indevida art. 165 ctn súmula 546",
            "indebito": "repetição indébito restituição tributo pago indevidamente cobrança indevida art. 165 ctn súmula 546",
            "repetição": "indébito restituição tributo pago indevidamente cobrança indevida",
            "restituição": "repetição indébito tributo pago indevidamente devolução valores",
            "tutela": "tutela urgência evidência liminar fumus boni iuris periculum in mora art. 300 art. 311 cpc",
            "liminar": "tutela urgência evidência fumus boni iuris periculum in mora antecipação",
            "isenção": "isenção imposto renda doença grave moléstia lei 7.713 art. 6",
            "isencao": "isenção imposto renda doença grave moléstia lei 7.713 art. 6",
            "contraditório": "contraditório ampla defesa devido processo legal perícia laudo médico",
            "contraditorio": "contraditório ampla defesa devido processo legal perícia laudo médico",
            # ── Termos de Direito Civil material (Grupo 3) ──
            "dano moral": "abalo moral sofrimento psíquico dor ofensa honra imagem reputação vergonha humilhação art. 186 cc art. 5º v x cf",
            "dano_estético": "dano estético deformidade alteração aparência física defeito estético",
            "dano material": "dano material prejuízo econômico patrimonial lucros cessantes dano emergente",
            "lucros cessantes": "lucros cessantes rendimento perdido ganho que deixou de obter",
            "nexo causal": "nexo de causalidade link relação causa-efeito dano evento nexo",
            "nexo": "nexo causal causalidade relação causa efeito dano",
            "culpa": "culpa negligência imprudência imperícia dolo responsabilidade civil",
            "enriquecimento sem causa": "enriquecimento ilícito indevido pagamento indevido benefício sem causa art. 884 cc",
            "enriquecimento ilícito": "enriquecimento sem causa indevido pagamento ilícito art. 884 cc",
            "prescrição civil": "prazo extintivo perda direito ação decurso tempo prazo legal art. 205 cc",
            "decadência": "prazo decadencial perda direito prazo legal cc",
            "cláusula penal": "cláusula penal multa compensatória mora descumprimento obrigação",
            "multa": "multa pena sanção cláusula penal compensatória",
            "vício redibitório": "vício redibitório defeito oculto coisa viciada evicção art. 441 cc",
            "evicção": "evicção perda coisa terceiro direito posse",
            "obrigação de fazer": "obrigação de fazer não fazer dar coisa certa execução específica",
            "obrigação": "obrigação contratual legal dever jurídico prestação",
            "posse": "posse propriedade domínio usucapião arts. 1196 cc",
            "propriedade": "propriedade domínio direito real coisa art. 1228 cc",
            "usucapião": "usucapião prescrição aquisitiva posse mansa pacífica",
            "alimentos": "alimentos pensão alimentícia obrigação alimentar covenantor art. 1694 cc",
            "pensão": "pensão alimentícia alimentos obrigação de pagar",
            "herança": "herança sucessão testamento inventário partilha art. 1784 cc",
            "sucessão": "sucessão herdeiros legítimos testamentários partilha inventário",
        }
        for termo, relacionados in expansions.items():
            if termo in query_lower:
                return f"{query} {relacionados}"
        return query