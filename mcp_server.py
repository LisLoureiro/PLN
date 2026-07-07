"""
mcp_server.py — Servidor MCP para Open WebUI
Protocolo MCP 2024-11-05 via SSE + OpenAPI REST para Open WebUI.

Ferramentas genéricas — não presumem nenhum domínio específico de
documento, já que o que é extraído depende da instrução de cada job.
"""

import asyncio
import json
import logging
import math
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import psycopg2
import psycopg2.extras
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://extractor:extractor123@localhost:5432/extractor",
)


def _conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _query(sql: str, params: tuple = ()) -> list:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def _scalar(sql: str, params: tuple = ()):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return list(row.values())[0] if row else None


# ─────────────────────────────────────────────────────────────
# RAG — TF-IDF puro (sem dependências externas)
# ─────────────────────────────────────────────────────────────

_STOPWORDS_PT = {
    "a", "ao", "aos", "as", "da", "das", "de", "do", "dos", "e", "em", "na", "nas",
    "no", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "que", "se", "um", "uma", "com", "é", "ser", "ter", "foi", "são", "mais",
    "como", "mas", "seu", "sua", "seus", "suas", "não", "este", "esta", "esse",
    "essa", "estes", "estas", "esses", "essas", "qual", "quando", "onde", "cada",
    "já", "até", "também", "ainda", "sobre", "entre", "após", "antes", "durante",
}


def _tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t not in _STOPWORDS_PT and len(t) > 2]


def _chunk_markdown(markdown: str, chunk_size: int = 600, overlap: int = 100) -> list:
    text = markdown.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _tfidf_rank(query: str, chunks: list, top_k: int = 5) -> list:
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [(0.0, c) for c in chunks[:top_k]]

    chunk_tfs = []
    doc_freq: dict = defaultdict(int)

    for chunk in chunks:
        tokens = _tokenize(chunk)
        tf: dict = defaultdict(float)
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


def _highlight(text: str, query: str) -> str:
    for token in _tokenize(query):
        if len(token) > 3:
            text = re.sub(rf"\b({re.escape(token)})\b", r"**\1**", text, flags=re.IGNORECASE)
    return text


def _fmt_date(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y %H:%M")
    return str(value)[:19]


# ─────────────────────────────────────────────────────────────
# Schema das ferramentas MCP
# ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "listar_jobs",
        "description": "Lista todos os jobs de extração já executados (cada um com sua instrução e total de itens).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limite": {"type": "integer", "default": 20, "description": "Máximo de registros (padrão 20, máx 100)."},
                "offset": {"type": "integer", "default": 0, "description": "Deslocamento para paginação."},
            },
        },
    },
    {
        "name": "buscar_job",
        "description": "Retorna detalhes completos de um job: instrução usada e todos os itens extraídos.",
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {
                "job_id": {"type": "string", "description": "ID do job (ex: 'A3F7B2C1')."},
            },
        },
    },
    {
        "name": "pesquisar_itens",
        "description": (
            "Busca itens já extraídos (de qualquer job) por texto livre — procura no resumo, "
            "na referência e nos dados extraídos. Útil para achar algo que já foi extraído antes "
            "sem saber em qual job está."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["termo"],
            "properties": {
                "termo": {"type": "string", "description": "Texto a buscar (case-insensitive)."},
                "job_id": {"type": "string", "description": "Opcional — restringe a um job."},
            },
        },
    },
    {
        "name": "status_banco",
        "description": "Estatísticas gerais do banco: documentos, jobs e itens extraídos.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pesquisar_no_documento",
        "description": (
            "RAG: busca semântica por relevância no Markdown original do PDF associado a um job. "
            "Divide o documento em chunks e ranqueia os trechos mais relevantes para a consulta via TF-IDF. "
            "Use para achar informação no texto do documento além do que já foi extraído."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["consulta", "job_id"],
            "properties": {
                "consulta": {"type": "string", "description": "Pergunta ou tema a buscar no documento."},
                "job_id": {"type": "string", "description": "Job cujo documento será pesquisado."},
                "top_k": {"type": "integer", "default": 5, "description": "Número de trechos a retornar (máx 15)."},
                "chunk_size": {"type": "integer", "default": 600, "description": "Tamanho dos chunks em caracteres."},
            },
        },
    },
    {
        "name": "perguntar_ao_documento",
        "description": (
            "RAG com IA: faz uma pergunta em linguagem natural sobre o conteúdo do documento original "
            "de um job, e o Claude responde com base no texto real (não apenas nos itens já extraídos)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["pergunta", "job_id"],
            "properties": {
                "pergunta": {"type": "string", "description": "Pergunta em linguagem natural sobre o documento."},
                "job_id": {"type": "string", "description": "Job cujo documento será usado como contexto."},
                "top_k": {"type": "integer", "default": 8, "description": "Chunks relevantes a enviar ao Claude."},
            },
        },
    },
    {
        "name": "pesquisar_geral",
        "description": (
            "Busca cruzada: (1) metadados dos jobs (instrução, arquivo), (2) itens extraídos, "
            "(3) texto do documento via RAG. Use quando não souber onde a informação está."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["termo"],
            "properties": {
                "termo": {"type": "string", "description": "Palavra, frase ou tema a buscar."},
                "job_id": {"type": "string", "description": "Opcional — restringe a um job."},
                "incluir_documento": {"type": "boolean", "default": True, "description": "Também buscar no Markdown do documento."},
                "top_k_doc": {"type": "integer", "default": 3, "description": "Trechos do documento a incluir."},
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────
# Implementação das ferramentas
# ─────────────────────────────────────────────────────────────

def tool_listar_jobs(args: dict) -> str:
    limite = min(int(args.get("limite", 20)), 100)
    offset = int(args.get("offset", 0))
    rows = _query(
        "SELECT job_id, source_file, instrucao, created_at, total "
        "FROM extraction_jobs ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limite, offset),
    )
    total_count = _scalar("SELECT COUNT(*) FROM extraction_jobs") or 0
    if not rows:
        return "Nenhum job de extração encontrado no banco de dados."
    lines = [
        f"📋 **Jobs de extração** — {total_count} registro(s) | mostrando {offset + 1}–{offset + len(rows)}\n",
        f"{'ID':<12} {'Arquivo':<30} {'Instrução':<40} {'Itens':<7} {'Data'}",
        "─" * 110,
    ]
    for r in rows:
        instr = (r["instrucao"] or "")[:38]
        src = (r["source_file"] or "")[:28]
        lines.append(f"{r['job_id']:<12} {src:<30} {instr:<40} {r['total']:<7} {_fmt_date(r['created_at'])}")
    return "\n".join(lines)


def tool_buscar_job(args: dict) -> str:
    job_id = (args.get("job_id") or "").strip()
    rows = _query("SELECT * FROM extraction_jobs WHERE job_id = %s", (job_id,))
    if not rows:
        return f"❌ Job `{job_id}` não encontrado."
    r = rows[0]
    items: list = json.loads(r.get("items_json") or "[]")

    lines = [
        f"## Job: {r['job_id']}",
        f"**Arquivo:** {r['source_file']}",
        f"**Instrução usada:** _{r['instrucao']}_",
        f"**Data:** {_fmt_date(r.get('created_at'))}",
        f"**Total de itens:** {r['total']}",
        "", "---", "### Itens extraídos", "",
    ]
    for i, item in enumerate(items, 1):
        lines.append(f"#### {i}. {item.get('resumo', '—')}")
        lines.append(f"- **Referência:** {item.get('trecho_referencia', '—')}")
        dados = item.get("dados", {})
        if dados:
            for k, v in dados.items():
                lines.append(f"- **{k}:** {v}")
        lines.append("")
    return "\n".join(lines)


def tool_pesquisar_itens(args: dict) -> str:
    termo = (args.get("termo") or "").lower()
    job_filter = args.get("job_id")
    rows = _query("SELECT job_id, source_file, instrucao, items_json FROM extraction_jobs")
    results = []
    for r in rows:
        if job_filter and r["job_id"] != job_filter:
            continue
        for item in json.loads(r.get("items_json") or "[]"):
            campos = " ".join([
                item.get("resumo", ""),
                item.get("trecho_referencia", ""),
                json.dumps(item.get("dados", {}), ensure_ascii=False),
            ]).lower()
            if termo in campos:
                results.append({"job_id": r["job_id"], "src": r["source_file"], "item": item})
    if not results:
        return f"🔍 Nenhum item encontrado para **\"{termo}\"**."
    lines = [f"🔍 **{len(results)} item(ns)** para `{termo}`\n"]
    for res in results:
        item = res["item"]
        lines += [
            f"**[{res['job_id']}]** {res['src'][:50]}",
            f"↳ {item.get('resumo', '—')}",
            f"  Referência: {item.get('trecho_referencia', '—')}",
            "",
        ]
    return "\n".join(lines)


def tool_status_banco(_args: dict) -> str:
    n_jobs = _scalar("SELECT COUNT(*) FROM extraction_jobs") or 0
    n_doc = _scalar("SELECT COUNT(*) FROM documents") or 0
    n_items = _scalar("SELECT COALESCE(SUM(total),0) FROM extraction_jobs") or 0
    last = _scalar("SELECT MAX(created_at) FROM extraction_jobs")
    return "\n".join([
        "## 🗄️ Status do Banco — Extrator Custom", "",
        f"- 🆔 **Jobs de extração:** {n_jobs}",
        f"- 📄 **Documentos únicos (Markdown):** {n_doc}",
        f"- 📋 **Itens extraídos:** {n_items}",
        f"- 🕐 **Última atualização:** {_fmt_date(last)}",
        "", "_Conectado via PostgreSQL + SSE_",
    ])


def _get_document_for_job(job_id: str) -> Optional[dict]:
    rows = _query(
        "SELECT d.doc_hash, d.source_file, d.markdown, j.instrucao "
        "FROM extraction_jobs j "
        "JOIN documents d ON d.doc_hash = j.doc_hash "
        "WHERE j.job_id = %s",
        (job_id,),
    )
    return rows[0] if rows else None


def tool_pesquisar_no_documento(args: dict) -> str:
    consulta = (args.get("consulta") or "").strip()
    job_id = (args.get("job_id") or "").strip()
    top_k = min(int(args.get("top_k", 5)), 15)
    chunk_size = max(200, min(int(args.get("chunk_size", 600)), 2000))

    if not consulta:
        return "❌ Parâmetro `consulta` é obrigatório."
    if not job_id:
        return "❌ Parâmetro `job_id` é obrigatório."

    doc = _get_document_for_job(job_id)
    if not doc:
        return f"❌ Documento não encontrado para o job `{job_id}`."

    md = doc.get("markdown") or ""
    if not md.strip():
        return "❌ Documento vazio."

    chunks = _chunk_markdown(md, chunk_size=chunk_size, overlap=100)
    ranked = _tfidf_rank(consulta, chunks, top_k=top_k)

    if not ranked:
        return (
            f"🔍 Nenhum trecho relevante encontrado para **\"{consulta}\"** no job `{job_id}`.\n"
            "_Tente reformular a consulta._"
        )

    output = [
        "## 📚 RAG — Trechos mais relevantes",
        f"> **Consulta:** _{consulta}_",
        f"> **Documento:** {doc.get('source_file', '—')} | {len(chunks)} chunks analisados\n",
    ]
    for rank, (score, chunk) in enumerate(ranked, 1):
        output.append(f"**Trecho #{rank}** — relevância `{score:.3f}`")
        output.append(f"> {_highlight(chunk, consulta).strip()}")
        output.append("")
    return "\n".join(output)


def tool_perguntar_ao_documento(args: dict) -> str:
    import urllib.request
    import json as _json

    pergunta = (args.get("pergunta") or "").strip()
    job_id = (args.get("job_id") or "").strip()
    top_k = min(int(args.get("top_k", 12)), 20)
    FULL_TEXT_LIMIT = 120_000

    if not pergunta:
        return "❌ Parâmetro `pergunta` é obrigatório."
    if not job_id:
        return "❌ Parâmetro `job_id` é obrigatório."

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "❌ ANTHROPIC_API_KEY não configurada no servidor MCP."

    doc = _get_document_for_job(job_id)
    if not doc:
        return f"❌ Documento não encontrado para o job `{job_id}`."

    md = (doc.get("markdown") or "").strip()
    src = doc.get("source_file") or "—"
    if not md:
        return "❌ Documento vazio."

    if len(md) <= FULL_TEXT_LIMIT:
        context = f"=== DOCUMENTO COMPLETO: {src} ===\n\n{md}"
        modo_badge = "📄 Texto completo"
    else:
        chunks = _chunk_markdown(md, chunk_size=800, overlap=150)
        ranked = _tfidf_rank(pergunta, chunks, top_k=top_k)
        rag_parts = []
        for i, (score, chunk) in enumerate(ranked, 1):
            rag_parts.append(f"[Trecho RAG #{i} | score {score:.3f}]\n{chunk}")
        context = "=== TRECHOS MAIS RELEVANTES ===\n\n" + "\n\n---\n\n".join(rag_parts)
        modo_badge = f"🔍 RAG TF-IDF ({len(ranked)} chunks)"

    system_prompt = (
        "Você é um assistente especialista em leitura de documentos. Responda perguntas "
        "com base EXCLUSIVAMENTE no conteúdo do documento fornecido. Cite cláusulas, seções "
        "ou trechos específicos quando disponíveis. Se a informação não estiver no documento, "
        "diga isso explicitamente. Seja preciso com valores, datas e nomes."
    )
    user_msg = f"PERGUNTA: {pergunta}\n\nCONTEXTO ({len(context):,} chars):\n\n{context}"

    payload = _json.dumps({
        "model": "claude-opus-4-5-20251101",
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = _json.loads(resp.read())
        answer = result["content"][0]["text"]
    except Exception as exc:
        logger.exception("Erro ao chamar Claude no MCP")
        return f"❌ Erro ao consultar Claude: {exc}"

    return "\n".join([
        "## 🤖 Resposta — Claude analisa o documento",
        f"> **Pergunta:** _{pergunta}_",
        f"_{modo_badge} | job `{job_id}` | arquivo: {src}_",
        "",
        answer,
    ])


def tool_pesquisar_geral(args: dict) -> str:
    termo = (args.get("termo") or "").strip()
    job_filter = (args.get("job_id") or "").strip()
    incl_doc = args.get("incluir_documento", True)
    top_k_doc = min(int(args.get("top_k_doc", 3)), 10)

    if not termo:
        return "❌ Parâmetro `termo` é obrigatório."

    termo_lower = termo.lower()
    output = [f"## 🔎 Busca Geral: `{termo}`", ""]
    encontrou_algo = False

    sql_jobs = (
        "SELECT job_id, source_file, instrucao, created_at, total, items_json, doc_hash "
        "FROM extraction_jobs WHERE job_id = %s" if job_filter
        else "SELECT job_id, source_file, instrucao, created_at, total, items_json, doc_hash FROM extraction_jobs"
    )
    jobs = _query(sql_jobs, (job_filter,) if job_filter else ())

    # SEÇÃO 1 — Metadados
    output.append("### 🗂️ 1. Metadados dos Jobs")
    meta_hits = [
        r for r in jobs
        if termo_lower in (r.get("instrucao") or "").lower()
        or termo_lower in (r.get("source_file") or "").lower()
        or termo_lower in (r.get("job_id") or "").lower()
    ]
    if meta_hits:
        encontrou_algo = True
        output.append(f"_{len(meta_hits)} job(s) encontrado(s)_\n")
        for r in meta_hits:
            output.append(f"- **[{r['job_id']}]** {r['source_file']} — _{r['instrucao'][:60]}_ | {r['total']} itens")
    else:
        output.append("_Nenhuma correspondência nos metadados._")
    output.append("")

    # SEÇÃO 2 — Itens extraídos
    output.append("### 📋 2. Itens Extraídos")
    item_hits = []
    for r in jobs:
        for item in json.loads(r.get("items_json") or "[]"):
            campos = " ".join([
                item.get("resumo", ""), item.get("trecho_referencia", ""),
                json.dumps(item.get("dados", {}), ensure_ascii=False),
            ]).lower()
            if termo_lower in campos:
                item_hits.append((r["job_id"], item))
    if item_hits:
        encontrou_algo = True
        output.append(f"_{len(item_hits)} item(ns) com `{termo}`_\n")
        for job_id_r, item in item_hits[:10]:
            output.append(f"- **[{job_id_r}]** {item.get('resumo', '—')}")
        if len(item_hits) > 10:
            output.append(f"\n_…e mais {len(item_hits) - 10} resultado(s)._")
    else:
        output.append(f"_Nenhum item com `{termo}`._")
    output.append("")

    # SEÇÃO 3 — RAG no documento
    output.append("### 📄 3. Documento — RAG (trechos mais relevantes)")
    if not incl_doc:
        output.append("_Busca no documento desativada._")
    else:
        doc_hits = 0
        seen_hashes = set()
        for r in jobs:
            dh = r.get("doc_hash")
            if not dh or dh in seen_hashes:
                continue
            seen_hashes.add(dh)
            doc_rows = _query("SELECT source_file, markdown FROM documents WHERE doc_hash = %s", (dh,))
            if not doc_rows:
                continue
            md = doc_rows[0].get("markdown") or ""
            if not md.strip():
                continue
            chunks = _chunk_markdown(md, chunk_size=600, overlap=100)
            ranked = _tfidf_rank(termo, chunks, top_k=top_k_doc)
            if not ranked:
                continue
            doc_hits += len(ranked)
            output.append(f"**[{r['job_id']}]** {doc_rows[0]['source_file'][:55]}")
            for score, chunk in ranked:
                hl = re.sub(rf"({re.escape(termo)})", r"**\1**", chunk.strip(), flags=re.IGNORECASE)
                output.append(f"> _score {score:.3f}_ — {hl[:500]}")
                output.append("")
        if doc_hits == 0:
            output.append("_Nenhum trecho relevante no(s) documento(s)._")
        else:
            encontrou_algo = True

    output.append("")
    if not encontrou_algo:
        output.append("---\n⚠️ **Nenhum resultado em nenhuma fonte.**\nVerifique a grafia ou tente um termo mais amplo.")

    return "\n".join(output)


HANDLERS: dict[str, Any] = {
    "listar_jobs": tool_listar_jobs,
    "buscar_job": tool_buscar_job,
    "pesquisar_itens": tool_pesquisar_itens,
    "status_banco": tool_status_banco,
    "pesquisar_no_documento": tool_pesquisar_no_documento,
    "perguntar_ao_documento": tool_perguntar_ao_documento,
    "pesquisar_geral": tool_pesquisar_geral,
}


# ─────────────────────────────────────────────────────────────
# Protocolo MCP — JSON-RPC 2.0
# ─────────────────────────────────────────────────────────────

def _ok(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_: Any, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": msg}}


def _dispatch(msg: dict) -> Optional[dict]:
    method = msg.get("method", "")
    id_ = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return _ok(id_, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "Extrator Custom MCP", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        })

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return _ok(id_, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if not handler:
            return _err(id_, -32601, f"Ferramenta desconhecida: {name}")
        try:
            return _ok(id_, {"content": [{"type": "text", "text": handler(args)}], "isError": False})
        except Exception as exc:
            logger.exception("Erro na ferramenta %s", name)
            return _err(id_, -32603, f"Erro interno: {exc}")

    return _err(id_, -32601, f"Método desconhecido: {method}")


# ─────────────────────────────────────────────────────────────
# OpenAPI — para integração com Open WebUI
# ─────────────────────────────────────────────────────────────

def _openapi_path(op_id: str, summary: str, schema: dict, required_body: bool = False) -> dict:
    return {
        "post": {
            "operationId": op_id,
            "summary": summary,
            "requestBody": {
                "required": required_body,
                "content": {"application/json": {"schema": schema}},
            },
            "responses": {"200": {"description": "Resultado"}},
        }
    }


OPENAPI_SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "Extrator Custom MCP Tools", "version": "1.0.0"},
    "paths": {
        "/tools/listar_jobs": _openapi_path(
            "listar_jobs", "Lista todos os jobs de extração.",
            {"type": "object", "properties": {"limite": {"type": "integer", "default": 20}, "offset": {"type": "integer", "default": 0}}},
        ),
        "/tools/buscar_job": _openapi_path(
            "buscar_job", "Detalhes completos de um job e seus itens.",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            required_body=True,
        ),
        "/tools/pesquisar_itens": _openapi_path(
            "pesquisar_itens", "Busca itens extraídos por texto livre.",
            {"type": "object", "required": ["termo"], "properties": {"termo": {"type": "string"}, "job_id": {"type": "string"}}},
            required_body=True,
        ),
        "/tools/status_banco": _openapi_path(
            "status_banco", "Estatísticas gerais do banco.", {"type": "object"},
        ),
        "/tools/pesquisar_no_documento": _openapi_path(
            "pesquisar_no_documento", "RAG TF-IDF no documento de um job.",
            {"type": "object", "required": ["consulta", "job_id"], "properties": {
                "consulta": {"type": "string"}, "job_id": {"type": "string"},
                "top_k": {"type": "integer", "default": 5}, "chunk_size": {"type": "integer", "default": 600},
            }},
            required_body=True,
        ),
        "/tools/perguntar_ao_documento": _openapi_path(
            "perguntar_ao_documento", "RAG com IA sobre o documento de um job.",
            {"type": "object", "required": ["pergunta", "job_id"], "properties": {
                "pergunta": {"type": "string"}, "job_id": {"type": "string"}, "top_k": {"type": "integer", "default": 8},
            }},
            required_body=True,
        ),
        "/tools/pesquisar_geral": _openapi_path(
            "pesquisar_geral", "Busca cruzada: metadados, itens e documento (RAG).",
            {"type": "object", "required": ["termo"], "properties": {
                "termo": {"type": "string"}, "job_id": {"type": "string"},
                "incluir_documento": {"type": "boolean", "default": True}, "top_k_doc": {"type": "integer", "default": 3},
            }},
            required_body=True,
        ),
    },
}


async def route_openapi(request: Request):
    return JSONResponse(OPENAPI_SPEC)


async def route_tool_call(request: Request):
    name = request.path_params.get("name", "")
    handler = HANDLERS.get(name)
    if not handler:
        return JSONResponse({"error": f"Ferramenta desconhecida: {name}"}, status_code=404)
    try:
        body = await request.body()
        args = json.loads(body) if body else {}
    except json.JSONDecodeError:
        args = {}
    try:
        result = handler(args)
        return JSONResponse({"result": result})
    except Exception as exc:
        logger.exception("Erro na ferramenta %s", name)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ─────────────────────────────────────────────────────────────
# Sessões SSE
# ─────────────────────────────────────────────────────────────

_sessions: dict[str, asyncio.Queue] = {}


async def route_sse(request: Request):
    sid = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[sid] = queue
    base = str(request.base_url).rstrip("/")
    post_url = f"{base}/messages?session_id={sid}"

    async def stream():
        try:
            yield f"event: endpoint\ndata: {post_url}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20)
                    if msg is None:
                        break
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sessions.pop(sid, None)
            logger.info("[MCP] Sessão encerrada: %s", sid[:8])

    logger.info("[MCP] Nova sessão SSE: %s", sid[:8])
    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


async def route_messages(request: Request):
    sid = request.query_params.get("session_id", "")
    queue = _sessions.get(sid)
    try:
        msg = json.loads(await request.body())
    except json.JSONDecodeError:
        return JSONResponse({"error": "JSON inválido"}, status_code=400)
    logger.info("[MCP] %s → %s", sid[:8], msg.get("method", "?"))
    response = _dispatch(msg)
    if response is not None:
        if queue is not None:
            await queue.put(response)
        else:
            return JSONResponse(response)
    return Response(status_code=202)


async def route_health(request: Request):
    return JSONResponse({"status": "ok", "service": "Extrator Custom MCP", "version": "1.0.0"})


app = Starlette(routes=[
    Route("/sse", route_sse, methods=["GET"]),
    Route("/messages", route_messages, methods=["POST"]),
    Route("/health", route_health, methods=["GET"]),
    Route("/openapi.json", route_openapi, methods=["GET"]),
    Route("/tools/{name}", route_tool_call, methods=["POST"]),
])

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", 8000))
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    logger.info("🚀 Extrator Custom MCP v1.0 em http://%s:%d/sse", host, port)
    logger.info("📡 Protocolo MCP 2024-11-05 · SSE + OpenAPI REST")
    logger.info("🗄️  Banco: %s", DB_URL.split("@")[-1])
    uvicorn.run(app, host=host, port=port, log_level="info")
