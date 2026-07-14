# Otimizações Possíveis - Reduzir Dependências

## 1. REMOVER ChromaDB ❌

**Problema:**
- ChromaDB é usado apenas para buscar itens extraídos
- O MCP server já implementa TF-IDF manualmente (`_tfidf_rank`)
- É uma dependência pesada (~10MB) para funcionalidade duplicada

**Solução:**
- Remover ChromaDB
- Usar PostgreSQL + TF-IDF local para tudo
- Já existe `_tfidf_rank` no MCP que pode ser reutilizado

## 2. SIMPLIFICAR PDF Normalization

**Atual:**
- `pdfplumber` (OK - local)
- Processamento página por página
- Detecção de headers/footers

**Melhoria:**
- Está bom, mas pode adicionar cache mais agressivo
- Já tem cache por hash (documentos), pode adicionar cache de chunks

## 3. Otimizar Modelo Claude

**Atual:**
```python
MODEL = "claude-opus-4-5-20251101"  # Nome antigo
MAX_TOKENS = 4096
```

**Melhoria:**
```python
MODEL = "claude-sonnet-4-6-20250314"  # Mais rápido, mais barato
# ou
MODEL = "claude-haiku-4-5-20251001"  # Para tarefas simples
```

## 4. REMOVER docker-compose dependencies

**Atual:**
- PostgreSQL (via docker)
- pgAdmin (via docker)
- Open WebUI (via docker)

**Melhoria:**
- PostgreSQL → SQLite (arquivo local, zero setup)
- pgAdmin → Remover (não essencial)
- Open WebUI → Manter opcional

## 5. Unificar TF-IDF

**Atual:**
- `vectorizer.py` usa scikit-learn
- `mcp_server.py` tem `_tfidf_rank` manual
- Dois implementações diferentes!

**Solução:**
- Usar só scikit-learn em todo lugar
- Remover implementação manual do MCP

## 6. Remover a2wsgi

**Atual:**
- `a2wsgi` para montar MCP em `/mcp`

**Solução:**
- Usar subdomínio ou porta separada
- Ou implementar middleware próprio

## 7. Simplificar Logging

**Atual:**
- ColorFormatter customizado
- Vários `sep()` calls

**Solução:**
- Usar logging padrão do Python
- Remover código visual não-essencial

## Resumo de Remoções

| Componente | Motivo | Substituto |
|------------|--------|-----------|
| ChromaDB | Redundante com TF-IDF + PostgreSQL | Remover |
| pgAdmin | Não essencial | Remover |
| docker-compose | Overhead | Setup local direto |
| `_tfidf_rank` manual | Duplicado com scikit-learn | Vectorizer |
| a2wsgi | Dependência extra | Middleware simples |

## Resumo Final

**APÓS OTIMIZAÇÃO:**

### Dependências Mínimas
```
anthropic    # Claude API ✅
pdfplumber   # PDF → Texto ✅
scikit-learn # TF-IDF ✅
sqlalchemy   # SQLite/PostgreSQL ✅
flask        # Web server ✅
starlette    # MCP ✅
```

### Arquitetura Simplificada
```
PDF → pdfplumber → Markdown → TF-IDF (scikit) → Claude → SQLite/Postgres
                                                              ↓
                                        TF-IDF para busca (scikit)
```

### Serviços
- Flask app (porta 5000)
- MCP server (porta 8000 ou /mcp)
- SQLite local (ou PostgreSQL se preferir)
