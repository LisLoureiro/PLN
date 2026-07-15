"""
app.py — Servidor Flask principal + MCP integrado em /mcp
Fluxo: usuário define a instrução → salva PDF → texto → chunks →
       vetorização guiada pela instrução → IA extrai conforme pedido → Store
"""

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from custom_extractor import CustomExtractor
from pdf_normalizer import PDFNormalizer
from store import Store
from vectorizer import Vectorizer

from mcp_server import app as mcp_asgi_app
from clause_library import setup_clause_routes, build_instruction_for_type

# ─────────────────────────────────────────────────────────────
# Logging com cores e separadores visuais
# ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"

class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: BLUE, logging.INFO: GREEN,
        logging.WARNING: YELLOW, logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }
    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        record.levelname = f"{color}{record.levelname:<8}{RESET}"
        record.msg = f"{record.msg}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

def sep(title: str = "", char: str = "─", width: int = 60):
    if title:
        pad = max(0, width - len(title) - 2)
        logger.info(f"{CYAN}{char * (pad // 2)} {title} {char * (pad - pad // 2)}{RESET}")
    else:
        logger.info(f"{CYAN}{char * width}{RESET}")


# ─────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────

flask_app = Flask(__name__)
flask_app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MIN_INSTRUCAO_CHARS = 8

# Configura rotas da biblioteca de cláusulas
setup_clause_routes(flask_app)


def _pgadmin_url(req) -> str:
    override = os.environ.get("PGADMIN_URL", "")
    if override:
        return override
    host = req.host
    if "app.github.dev" in host or "gitpod.io" in host:
        return "https://" + host.replace("-5000.", "-5050.")
    return "http://localhost:5050"


def _openwebui_url(req) -> str:
    override = os.environ.get("OPENWEBUI_URL", "")
    if override:
        return override
    host = req.host
    if "app.github.dev" in host or "gitpod.io" in host:
        return "https://" + host.replace("-5000.", "-3000.")
    return "http://localhost:3000"


store = Store()
sep("APP PRONTO", "═")
logger.info(f"{BOLD}🚀 Flask iniciando na porta 5000{RESET}")
logger.info(f"{BOLD}📡 MCP disponível em /mcp/sse e /mcp/openapi.json{RESET}")
sep(char="═")


# ─────────────────────────────────────────────────────────────
# Rotas Flask
# ─────────────────────────────────────────────────────────────

@flask_app.route("/")
def index():
    return render_template(
        "index.html",
        pgadmin_url=_pgadmin_url(request),
        openwebui_url=_openwebui_url(request),
    )


@flask_app.route("/library")
def library():
    """Redireciona para a página principal (biblioteca integrada)."""
    return redirect("/")


@flask_app.route("/db")
def open_db():
    return redirect(_pgadmin_url(request))


@flask_app.route("/ai")
def open_ai():
    return redirect(_openwebui_url(request))


@flask_app.route("/api/upload", methods=["POST"])
def upload():
    sep("EXTRAÇÃO INICIADA", "▶")

    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Apenas arquivos PDF são aceitos."}), 400

    # O usuário pode digitar uma instrução livre (campo 'instrucao') OU
    # escolher um tipo pré-definido no seletor do prompt (campo
    # 'clause_type', valores em GET /api/clause-types). Se ambos vierem,
    # 'clause_type' vira a instrução-base e 'instrucao' complementa como
    # observação adicional.
    instrucao = (request.form.get("instrucao") or "").strip()
    clause_type = (request.form.get("clause_type") or "").strip()

    if clause_type:
        try:
            instrucao = build_instruction_for_type(clause_type, extra=instrucao or None)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    elif len(instrucao) < MIN_INSTRUCAO_CHARS:
        return jsonify({"error": (
            "Descreva o que deve ser extraído do documento (campo 'instrucao'), "
            "ou selecione um tipo de cláusula (campo 'clause_type'). "
            f"Mínimo de {MIN_INSTRUCAO_CHARS} caracteres para 'instrucao'."
        )}), 400

    job_id = str(uuid.uuid4())[:8].upper()
    safe_name = re.sub(r"[^\w.\-]", "_", f.filename)
    pdf_path = UPLOAD_DIR / f"{job_id}_{safe_name}"

    try:
        logger.info(f"📄 Arquivo    : {f.filename}")
        logger.info(f"🆔 Job        : {job_id}")
        logger.info(f"📝 Instrução  : {instrucao[:200]}")

        sep("Etapa 0 — Salvando arquivo")
        f.save(str(pdf_path))

        size_mb = pdf_path.stat().st_size / 1_048_576
        logger.info(f"{GREEN}✔ Salvo em {pdf_path} ({size_mb:.1f} MB){RESET}")

        MAX_PDF_MB = float(os.environ.get("MAX_PDF_MB", "20"))
        if size_mb > MAX_PDF_MB:
            return jsonify({"error": (
                f"PDF muito grande ({size_mb:.1f} MB). Limite atual: {MAX_PDF_MB:.0f} MB."
            )}), 413

        sep("Etapa 1 — Hash + cache do documento")
        doc_hash = _sha256_file(pdf_path)
        cached = store.get_document_by_hash(doc_hash)

        if cached:
            logger.info(f"{YELLOW}⚡ PDF já normalizado antes (hash {doc_hash[:8]}) — reaproveitando Markdown{RESET}")
            markdown = cached["markdown"]
            from_cache = True
        else:
            logger.info("📄 PDF novo (ou alterado) — iniciando normalização…")
            from_cache = False

            sep("Etapa 2 — Normalizando PDF → Markdown")
            normalizer = PDFNormalizer(pdf_path)
            markdown = normalizer.to_markdown()
            del normalizer
            logger.info(f"{GREEN}✔ {len(markdown):,} chars de Markdown gerado{RESET}")
            logger.info(f"[DEBUG] Amostra do Markdown (primeiros 500 chars): {markdown[:500]}...")
            logger.info(f"[DEBUG] Contém 'CLÁUSULA': {'SIM' if 'CLÁUSULA' in markdown.upper() or 'CLAUSULA' in markdown.upper() else 'NÃO'}")
            logger.info(f"[DEBUG] Contém 'ENDEREÇO': {'SIM' if 'ENDEREÇO' in markdown.upper() or 'ENDERECO' in markdown.upper() or 'ENDEREÇAMENTO' in markdown.upper() else 'NÃO'}")
            logger.info(f"[DEBUG] Contém 'AVENIDA' ou 'RUA': {'SIM' if 'AVENIDA' in markdown.upper() or 'RUA' in markdown.upper() else 'NÃO'}")

            if len(markdown) < 200:
                return jsonify({"error": (
                    "Markdown gerado muito curto. O PDF pode ser escaneado (imagem) "
                    "ou protegido contra cópia."
                )}), 422

            sep("Etapa 3 — Salvando Markdown no PostgreSQL (cache por hash)")
            store.save_document(doc_hash=doc_hash, source_file=f.filename, markdown=markdown)
            logger.info(f"{GREEN}✔ Documento salvo no banco{RESET}")

        sep("Etapa 4 — Chunking do Markdown")
        chunks = _chunk_markdown(markdown)
        logger.info(f"{GREEN}✔ {len(chunks)} chunks gerados{RESET}")
        logger.info(f"[DEBUG] Exemplo de chunk (primeiros 200 chars): {chunks[0][:200] if chunks else 'SEM CHUNKS'}...")

        sep("Etapa 5 — Vetorização TF-IDF guiada pela instrução")
        vec = Vectorizer()
        vec.index_chunks(chunks)
        relevant = vec.search(instrucao)
        logger.info(f"{GREEN}✔ {len(relevant)} trechos relevantes para a instrução{RESET}")
        logger.info(f"[DEBUG] Exemplo de trecho relevante (primeiros 300 chars): {relevant[0][:300] if relevant else 'SEM TRECHOS RELEVANTES'}...")

        sep("Etapa 6 — Ollama (extração conforme instrução do usuário)")
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"🤖 Enviando {len(relevant)} trechos para o Ollama ({ollama_url})…")
        logger.info(f"[DEBUG] Total de caracteres enviados: {sum(len(t) for t in relevant):,}")
        logger.info(f"[DEBUG] CONTEÚDO COMPLETO DOS TRECHOS RELEVANTES:")
        for i, trecho in enumerate(relevant):
            logger.info(f"[DEBUG] TRECHO #{i+1}/{len(relevant)} ({len(trecho)} chars): {trecho}")
        extractor = CustomExtractor(base_url=ollama_url)
        items = extractor.extract(relevant, instrucao)
        logger.info(f"{GREEN}✔ {len(items)} itens extraídos{RESET}")

        sep("Etapa 7 — Persistência (PostgreSQL + ChromaDB)")
        store.save_job(
            job_id=job_id, doc_hash=doc_hash, source_file=f.filename,
            instrucao=instrucao, items=items,
        )
        logger.info(f"{GREEN}✔ Job {job_id} salvo com sucesso{RESET}")

        sep("EXTRAÇÃO CONCLUÍDA ✔", "▶")

        return jsonify({
            "job_id": job_id,
            "instrucao": instrucao,
            "clause_type": clause_type or None,
            "items": [i.to_dict() for i in items],
            "total": len(items),
            "from_cache": from_cache,
            "markdown_chars": len(markdown),
        })

    except Exception as e:
        logger.exception(f"{RED}✘ Erro inesperado{RESET}")
        return jsonify({"error": f"Erro interno: {e}"}), 500

    finally:
        if pdf_path.exists():
            pdf_path.unlink()
            logger.info("🗑  Arquivo temporário removido.")


@flask_app.route("/api/jobs", methods=["GET"])
def list_jobs():
    logger.info(f"{BLUE}→ GET /api/jobs{RESET}")
    return jsonify(store.list_jobs())


@flask_app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    logger.info(f"{BLUE}→ GET /api/jobs/{job_id}{RESET}")
    data = store.get_job(job_id)
    if not data:
        return jsonify({"error": "Job não encontrado"}), 404
    return jsonify(data)


@flask_app.route("/api/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    logger.info(f"{YELLOW}→ DELETE /api/jobs/{job_id}{RESET}")
    store.delete_job(job_id)
    return jsonify({"ok": True})


@flask_app.route("/api/v1/models", methods=["GET"])
def get_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "claude-3-haiku-20240307", "object": "model", "created": 1708600000, "owned_by": "openai"},
            {"id": "claude-3-sonnet-20240229", "object": "model", "created": 1708600000, "owned_by": "openai"},
            {"id": "claude-3-opus-20240229", "object": "model", "created": 1708600000, "owned_by": "openai"},
        ]
    })


@flask_app.route("/api/v1/public/models", methods=["GET"])
def get_public_models():
    return get_models()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def _chunk_markdown(md: str, chunk_size: int = 800, overlap: int = 100) -> list:
    sections = re.split(r"(?=\n#{2,3} )", md)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= chunk_size:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                end = min(start + chunk_size, len(sec))
                chunk = sec[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────
# WSGI composto: Flask na raiz, MCP em /mcp
# ─────────────────────────────────────────────────────────────

try:
    from a2wsgi import ASGIMiddleware
    mcp_wsgi = ASGIMiddleware(mcp_asgi_app)
except ImportError:
    mcp_wsgi = None
    logger.warning("⚠️  a2wsgi não instalado — MCP em /mcp indisponível. Instale: pip install a2wsgi")

if mcp_wsgi:
    app = DispatcherMiddleware(flask_app, {"/mcp": mcp_wsgi})
else:
    app = flask_app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"{BOLD}🌐 Servidor : http://0.0.0.0:{port}{RESET}")
    logger.info(f"{BOLD}📡 MCP SSE  : http://0.0.0.0:{port}/mcp/sse{RESET}")
    logger.info(f"{BOLD}📄 OpenAPI  : http://0.0.0.0:{port}/mcp/openapi.json{RESET}")
    run_simple("0.0.0.0", port, app, use_reloader=True, use_debugger=True)
