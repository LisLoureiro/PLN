# Extrator Custom de PDFs — IA

Interface web para extrair **qualquer informação que o usuário definir** de um
PDF, usando uma instrução em linguagem natural — sem schema fixo. Baseado na
arquitetura do projeto de obrigações de Agente Fiduciário (CPR-F), mas
generalizado: aqui é o próprio usuário quem delimita o que será extraído,
**antes** de qualquer coisa ser salva no banco.

## Como funciona

1. O usuário envia um PDF **e** escreve uma instrução livre, ex:
   > "Extraia todas as cláusulas de reajuste do aluguel, com o índice usado,
   > a periodicidade e o percentual aplicado."
2. O PDF é convertido em Markdown (com cache por hash — o mesmo arquivo não
   é reprocessado se enviado de novo, mesmo com instruções diferentes).
3. Os trechos mais relevantes **para aquela instrução específica** são
   selecionados via TF-IDF.
4. O Claude extrai os itens em JSON, com o formato de campos livre (definido
   implicitamente pela instrução), sempre com um `resumo` e uma
   `trecho_referencia` para rastreabilidade.
5. O resultado é salvo (PostgreSQL + ChromaDB) como um **job de extração**,
   ligado ao documento e à instrução usada.

## Arquitetura

```
app.py                  ← Servidor Flask + rotas API
├── pdf_normalizer.py   ← PDF → Markdown estruturado (genérico)
├── vectorizer.py        ← TF-IDF guiado pela instrução do usuário
├── custom_extractor.py  ← Claude AI → JSON de formato livre
├── store.py              ← Persistência PostgreSQL + ChromaDB
└── templates/
    └── index.html        ← Interface unificada: Landing + Extrator + Biblioteca

mcp_server.py            ← Servidor MCP/SSE genérico para Open WebUI
```

## Instalação

```bash
pip install -r requirements.txt
```

## Configuração

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export DATABASE_URL="postgresql://extractor:extractor123@localhost:5432/extractor"
```

## Execução

```bash
python app.py
```

Acesse: http://localhost:5000

## Interface Web

A aplicação possui uma interface unificada com três seções:

1. **Landing Page** - Apresentação do sistema com visão geral das funcionalidades
2. **Extrator de PDFs** - Upload de documentos e extração guiada por instruções
3. **Biblioteca de Cláusulas** - Busca e aprovação de precedentes jurídicos

**Design:**
- Tipografia elegante com Newsreader (serif) e Inter (sans)
- Paleta de cores navy/blue profissional
- Layout responsivo para diferentes dispositivos
- Navegação por abas integradas

## Docker (recomendado)

```bash
cp .env.example .env   # edite ANTHROPIC_API_KEY
docker compose up -d --build
```

| Serviço | URL |
|---|---|
| Flask (upload + instrução) | http://localhost:5000 |
| MCP Server (SSE) | http://localhost:8000/sse |
| Open WebUI | http://localhost:3000 |
| pgAdmin | http://localhost:5050 |

## API REST

| Método | Rota | Descrição |
|--------|------|-----------|
| POST | /api/upload | Upload de PDF + `instrucao` (form field) → extração |
| GET | /api/jobs | Lista todos os jobs de extração |
| GET | /api/jobs/:id | Detalhe de um job (instrução + itens) |
| DELETE | /api/jobs/:id | Remove um job |

### Exemplo `curl`

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@contrato.pdf" \
  -F "instrucao=Extraia todas as multas contratuais, com o valor, o evento que a gera e a cláusula correspondente."
```

Resposta:
```json
{
  "job_id": "A1B2C3D4",
  "instrucao": "Extraia todas as multas contratuais...",
  "total": 3,
  "from_cache": false,
  "markdown_chars": 48213,
  "items": [
    {
      "resumo": "Multa de 10% do valor do contrato por rescisão antecipada",
      "trecho_referencia": "Cláusula 9.2",
      "dados": {
        "valor": "10% do valor total do contrato",
        "evento_gatilho": "rescisão antecipada pelo locatário"
      }
    }
  ]
}
```

## Persistência

**PostgreSQL**:
- `documents` — Markdown normalizado, cacheado por **hash do PDF** (o mesmo
  arquivo pode alimentar vários jobs com instruções diferentes sem
  reprocessar o PDF)
- `extraction_jobs` — instrução usada + itens extraídos (JSON de formato
  livre) por job

**Busca de itens**: Implementada via TF-IDF local (scikit-learn) sobre os
resumos dos itens extraídos.

## Ferramentas MCP (Open WebUI)

| Ferramenta | Descrição |
|---|---|
| `listar_jobs` | Lista jobs de extração com paginação |
| `buscar_job` | Detalhe completo de um job (instrução + itens) |
| `pesquisar_itens` | Busca itens já extraídos por texto livre |
| `status_banco` | Estatísticas gerais |
| `pesquisar_no_documento` | RAG TF-IDF no Markdown do documento de um job |
| `perguntar_ao_documento` | RAG + Claude — responde perguntas sobre o documento original |
| `pesquisar_geral` | Busca cruzada: metadados + itens + documento |

## Diferenças em relação ao projeto original (CPR-F)

| | CPR-F (original) | Extrator Custom (este projeto) |
|---|---|---|
| O que é extraído | Fixo: obrigações do Agente Fiduciário | Livre: definido pelo usuário a cada upload |
| Categorias | 15 categorias fixas | Nenhuma — campos livres em `dados` |
| Query de busca (TF-IDF) | Fixa (termos de CPR-F) | A própria instrução do usuário |
| Cache de documento | Por `emission_id` informado manualmente | Por hash do arquivo (automático) |
| Reuso de schema | N/A | Não salvo (cada upload define do zero) |
