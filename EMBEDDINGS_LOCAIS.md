# Implementação Local de Algoritmos

## O que foi feito

Substituímos APIs externas por implementações locais:

### 1. TF-IDF (`vectorizer.py`)
**Status:** ✓ Já era local
- Usa scikit-learn para vetorização TF-IDF
- Seleção de trechos relevantes via similaridade de cosseno
- Sem chamadas de API

### 2. Embeddings para ChromaDB (`store.py`)
**Status:** ✓ Atualizado para local
**Antes:**
```python
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
_ef = DefaultEmbeddingFunction()  # API externa
```

**Depois:**
```python
from local_embedding import create_local_embedding
self._embedding_fn = create_local_embedding()  # Implementação local TF-IDF
```

### 3. Arquivo Novo: `local_embedding.py`
Implementa duas opções de embedding local:

#### `LocalEmbeddingFunction` (padrão)
- TF-IDF com n-gramas (1,2)
- Normalização L2
- Vetores densos compatíveis com ChromaDB

#### `HybridEmbeddingFunction` (opcional)
- TF-IDF + features textuais:
  - Densidade de números
  - Comprimento médio das palavras
  - Razão de maiúsculas/minúsculas
  - Densidade de pontuação

### 4. MCP Server (`mcp_server.py`)
**Status:** ✓ Já era local
- Implementa TF-IDF manualmente na função `_tfidf_rank`
- Sem dependências de APIs de embedding

## Benefícios

✓ **Zero custos de API** - Nenhuma chamada paga para embeddings
✓ **Privacidade** - Dados não saem da máquina
✓ **Latência menor** - Processamento local
✓ **Offline** - Funciona sem internet

## Dependências

Todas as dependências já estavam no `requirements.txt`:
- scikit-learn (TF-IDF)
- numpy (operações vetoriais)
- chromadb (vetor store - agora com embedding local)

## Uso

O sistema funciona da mesma forma, apenas usando embeddings gerados localmente:

```python
# No store.py, embeddings são gerados automaticamente
embeddings = self._embedding_fn(docs)
self._items.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
```

## Trocar para Hybrid Embedding

Para usar a versão híbrida com features adicionais, edite `store.py`:

```python
from local_embedding import create_hybrid_embedding
self._embedding_fn = create_hybrid_embedding()
```
