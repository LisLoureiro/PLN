# Embeddings Semânticos Locais

## O que foi criado

Arquivo `semantic_embedding.py` com **3 métodos** de embeddings semânticos que funcionam 100% localmente:

## 1. Sentence-Transformers (Recomendado) ⭐

Usa modelos pré-treinados que rodam localmente.

```python
from semantic_embedding import create_semantic_embedding

embedder = create_semantic_embedding(
    method="sentence-transformers",
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 50MB, português
)

embeddings = embedder.embed([
    "O inquilino deve pagar aluguel",
    "Locatário obrigado a pagar mensalidade",
    "O carro é vermelho"
])

# Similaridade: 0.85 (inquilino/locatário são sinônimos)
# Similaridade: 0.12 (carro não tem relação)
```

**Modelos disponíveis:**
| Modelo | Tamanho | Idioma | Precisão |
|--------|---------|--------|----------|
| `all-MiniLM-L6-v2` | 23MB | Inglês | Boa |
| `paraphrase-multilingual-MiniLM-L12-v2` | 50MB | Multilíngue ✅ | Boa |
| `distiluse-base-multilingual-cased-v2` | 250MB | Multilíngue | Excelente |

**Instalação:**
```bash
pip install -r requirements-semantic.txt
```

## 2. Hybrid (Mais Leve) 🔧

Combina TF-IDF com features semânticas (sem modelos pesados).

```python
embedder = create_semantic_embedding(method="hybrid")
embeddings = embedder.embed(textos)
```

**Características:**
- ✅ Zero dependências extras (usa scikit-learn)
- ✅ Mais rápido que sentence-transformers
- ⚠️ Menos preciso que modelos pré-treinados

## 3. Word2Vec-Style (Treinável) 📊

Treina um modelo Word2Vec no próprio corpus.

```python
embedder = create_semantic_embedding(method="word2vec")
embedder.fit(corpus)  # Treina no seu texto
embeddings = embedder.embed(textos)
```

**Características:**
- Aprende o vocabulário do seu domínio (ex: jurídico)
- Requer `gensim`
- Bom para domínios específicos

## Integração com Store.py

```python
from store import Store

store = Store()

# Busca semântica em itens extraídos
results = store.search_items_semantic(
    query="multas e penalidades",
    method="sentence-transformers",  # ou "hybrid", "word2vec"
    top_k=10
)

# Retorna:
# [
#   {
#     "job_id": "ABC123",
#     "resumo": "Multa de 10% por rescisão...",
#     "dados": {"valor": "10%", ...}
#   },
#   ...
# ]
```

## Comparação

| Método | Instalação | Precisão | Velocidade | Tamanho |
|--------|------------|----------|------------|---------|
| sentence-transformers | pip | ⭐⭐⭐⭐⭐ | Média | 50-250MB |
| hybrid | Já tem | ⭐⭐⭐ | Rápido | 0MB |
| word2vec | pip gensim | ⭐⭐⭐ | Rápido | ~100MB |
| TF-IDF puro | Já tem | ⭐⭐ | Muito rápido | 0MB |

## Exemplo de Uso

```bash
# Teste rápido
python semantic_embedding.py

# Exemplo completo
python exemplo_semantic.py
```

## Diferença: TF-IDF vs Semântico

```
Query: "sanções por não pagar"

TF-IDF acha:
✓ "sanção por não pagamento"  ← mesma palavra
✗ "multa por inadimplemento"  ← palavra diferente

Semântico acha:
✓ "sanção por não pagamento"
✓ "multa por inadimplência"    ← mesmo sentido!
✓ "penalidade por atraso"      ← mesmo sentido!
```

## Quando usar cada um

**Use TF-IDF:**
- Busca por palavras específicas
- Precisa ser muito rápido
- Recursos limitados

**Use Semântico:**
- Busca por conceitos/sinônimos
- Usuário não sabe os termos exatos
- Melhor experiência de busca
