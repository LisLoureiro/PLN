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
from typing import List

from flask import Flask, jsonify, redirect, render_template, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from custom_extractor import CustomExtractor, ExtractionParseError
from pdf_normalizer import PDFNormalizer
from store import Store
from vectorizer import Vectorizer

from mcp_server import app as mcp_asgi_app
from clause_library import setup_clause_routes, build_instruction_for_type, ClauseLibrary, FULL_DOCUMENT_CLAUSE_TYPES

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
        if clause_type in FULL_DOCUMENT_CLAUSE_TYPES:
            # Checklists ESTRUTURAIS (ex.: art. 319 do CPC) têm requisitos
            # espalhados por partes muito diferentes do documento, muitas
            # vezes em trechos sem vocabulário em comum com a instrução
            # (a linha do valor da causa não menciona "requisito"). Filtrar
            # pelos chunks "mais parecidos" com a instrução arrisca deixar
            # de fora justamente os trechos que provam um requisito ausente.
            # Por isso, para esses tipos, o documento INTEIRO é enviado —
            # sem filtro de relevância.
            relevant = chunks
            logger.info(
                f"{YELLOW}⚡ Tipo '{clause_type}' exige cobertura ampla — "
                f"pulando filtro TF-IDF, enviando todos os {len(chunks)} chunks{RESET}"
            )
        else:
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
        # Cobertura ampla (documento inteiro) tende a gerar um contexto
        # bem maior que o padrão — usa um num_ctx maior para não truncar
        # justamente os trechos que provam um requisito ausente.
        num_ctx_override = 16384 if clause_type in FULL_DOCUMENT_CLAUSE_TYPES else None
        try:
            items = extractor.extract(relevant, instrucao, num_ctx_override=num_ctx_override)
        except ExtractionParseError as e:
            # IMPORTANTE: isto é diferente de "0 itens encontrados". O
            # modelo não devolveu um JSON parseável mesmo após retry com
            # prompt reforçado (ver custom_extractor.py) — normalmente
            # porque o contexto ficou grande demais ou o modelo "conversou"
            # em vez de extrair. Não deve ser reportado como "cláusula
            # ausente do documento", que é uma conclusão totalmente
            # diferente e enganosa para quem está revisando a petição.
            logger.error(f"{RED}✘ Modelo não retornou JSON válido após retries: {e}{RESET}")
            return jsonify({
                "error": (
                    "O modelo de IA não conseguiu retornar os dados no formato "
                    "esperado, mesmo após uma nova tentativa automática. Isso NÃO "
                    "significa que a cláusula/tópico esteja ausente do documento — "
                    "geralmente indica que o contexto enviado ficou grande demais "
                    "para o modelo local, ou a resposta veio como texto explicativo "
                    "em vez de dados extraídos. Tente novamente; se persistir, "
                    "reduza o tamanho do PDF ou refine a instrução."
                ),
                "parse_failure": True,
            }), 502
        logger.info(f"{GREEN}✔ {len(items)} itens extraídos{RESET}")

        sep("Etapa 7 — Persistência (PostgreSQL + ChromaDB)")
        store.save_job(
            job_id=job_id, doc_hash=doc_hash, source_file=f.filename,
            instrucao=instrucao, items=items,
        )
        logger.info(f"{GREEN}✔ Job {job_id} salvo com sucesso{RESET}")

        # Adiciona itens à Biblioteca de Cláusulas como pendentes
        library = ClauseLibrary()
        items_dict = [item.to_dict() for item in items]
        added = library.add_from_extraction(
            job_id=job_id,
            doc_hash=doc_hash,
            source_file=f.filename,
            items=items_dict,
            auto_approve=False,  # Itens novos começam como pendentes
            clause_type=clause_type,
        )
        logger.info(f"{GREEN}✔ {added} cláusula(s) adicionada(s) à biblioteca (pendente){RESET}")

        sep("EXTRAÇÃO CONCLUÍDA ✔", "▶")

        total_items = len(items)
        return jsonify({
            "job_id": job_id,
            "instrucao": instrucao,
            "clause_type": clause_type or None,
            "items": [i.to_dict() for i in items],
            "total": total_items,
            "from_cache": from_cache,
            "markdown_chars": len(markdown),
            "no_clauses_found": total_items == 0,  # Flag para frontend
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


# ─────────────────────────────────────────────────────────────
# Chunking híbrido: cláusula/seção → parágrafo → frase (fallback)
#
# Objetivo: cada chunk enviado ao Vectorizer/Ollama deve ser, sempre que
# possível, uma cláusula ou seção COMPLETA — não um pedaço arbitrário de
# N caracteres que pode começar no meio de uma cláusula e terminar no
# meio da próxima.
#
# Estratégia (3 níveis, cai para o próximo só quando necessário):
#   1. CLÁUSULA/SEÇÃO — corte primário nas fronteiras de heading ##/###
#      já marcadas pelo PDFNormalizer (cláusulas, seções, títulos). Cada
#      cláusula completa vira um chunk, do heading até o próximo heading
#      de mesmo nível. Headings #### (itens numerados como "3.1") NÃO são
#      fronteira de corte — continuam dentro do corpo da cláusula-mãe.
#   2. PARÁGRAFO — se uma cláusula isolada passar de MAX_CLAUSE_CHARS,
#      ela é subdividida em chunks por parágrafo, nunca cortando um
#      parágrafo no meio.
#   3. FRASE — último recurso, só usado se um único parágrafo (raro)
#      ainda assim passar de MAX_CLAUSE_CHARS: corta por frase.
#
# SEM overlap entre chunks: como cada chunk já é uma unidade textual
# completa (cláusula, parágrafo ou frase), repetir texto na fronteira
# não recupera contexto perdido — só gasta tokens à toa.
# ─────────────────────────────────────────────────────────────

MAX_CLAUSE_CHARS = int(os.environ.get("CHUNK_MAX_CLAUSE_CHARS", "2000"))
MIN_SECTION_CHARS = 80  # títulos "órfãos" (heading sem corpo) são anexados à seção seguinte

_HEADING_RE = re.compile(r'^(#+)\s+.*$')


def _chunk_markdown(md: str, max_clause_chars: int = MAX_CLAUSE_CHARS) -> list:
    """
    Particiona o Markdown normalizado preservando cláusulas/seções
    completas. Ver comentário acima para a estratégia em 3 níveis.

    Quando uma cláusula precisa ser subdividida (nível 2 ou 3), o
    heading da cláusula (##/### e o corpo #### que vier junto) é
    repetido em CADA sub-chunk resultante — sem isso, um fragmento como
    "Nos termos do art. 6º..." ficaria sem nenhuma pista de que pertence
    à cláusula "2.1 DA ISENÇÃO DO IMPOSTO DE RENDA", o que prejudica a
    extração. Isso é diferente do overlap entre chunks vizinhos (que foi
    removido): é a cláusula repetindo o PRÓPRIO título em pedaços dela
    mesma, não texto de um chunk vazando para o chunk seguinte.
    """
    sections = _split_by_clause_headings(md)
    sections = _merge_short_sections(sections, MIN_SECTION_CHARS)

    chunks = []
    for section in sections:
        if len(section) <= max_clause_chars:
            chunks.append(section)
            continue

        logger.info(
            "[Chunking] Cláusula com %d chars (> %d) — subdividindo por parágrafo.",
            len(section), max_clause_chars,
        )
        heading, body = _split_heading_and_body(section)
        budget = max(max_clause_chars - len(heading) - 2, 200)  # reserva espaço pro heading repetido

        if not body:
            chunks.append(heading)
            continue

        sub_chunks = _split_by_paragraph(body, budget)
        if heading:
            sub_chunks = [f"{heading}\n\n{sc}" for sc in sub_chunks]
        chunks.extend(sub_chunks)

    return [c.strip() for c in chunks if c.strip()]


def _split_heading_and_body(section: str) -> "tuple[str, str]":
    """
    Separa as linhas de heading (##, ### ou ####) do início da seção do
    restante do corpo. Usado para poder repetir o heading em cada
    sub-chunk quando a cláusula precisa ser dividida em pedaços.
    """
    lines = section.split('\n')
    heading_lines: List[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('#'):
            heading_lines.append(lines[i])
            i += 1
        else:
            break
    heading = '\n'.join(heading_lines).strip()
    body = '\n'.join(lines[i:]).strip()
    return heading, body


def _split_by_clause_headings(md: str) -> List[str]:
    """
    Divide o markdown em segmentos, cada um começando num heading de
    nível ## ou ### (cláusula/seção) e contendo todo o texto até o
    próximo heading ##/###. Headings #### (numeração como "3.1") e texto
    comum permanecem dentro do segmento — só ##/### quebram.

    Texto antes do primeiro heading (ex.: cabeçalho "AO JUÍZO...") forma
    o primeiro segmento (preâmbulo da petição/contrato).
    """
    lines = md.split('\n')
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        stripped = line.strip()
        m = _HEADING_RE.match(stripped)
        is_clause_boundary = bool(m) and len(m.group(1)) in (2, 3)

        if is_clause_boundary and current:
            sections.append('\n'.join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append('\n'.join(current).strip())

    return [s for s in sections if s.strip()]


def _merge_short_sections(sections: List[str], min_chars: int) -> List[str]:
    """
    Anexa seções muito curtas (ex.: um heading "## SEÇÃO 3" sem corpo
    próprio, ou um título solto) à seção SEGUINTE, para não gerar chunks
    inúteis de 1 linha. Se a última seção do documento sobrar pequena,
    é anexada à anterior em vez de descartada.
    """
    merged: List[str] = []
    carry = ""

    for sec in sections:
        combined = f"{carry}\n\n{sec}".strip() if carry else sec
        if len(combined) < min_chars:
            carry = combined
        else:
            merged.append(combined)
            carry = ""

    if carry:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{carry}".strip()
        else:
            merged.append(carry)

    return merged


def _split_by_paragraph(section: str, max_chars: int) -> List[str]:
    """
    Subdivide uma cláusula grande demais em pedaços por parágrafo
    (separados por linha em branco), nunca cortando um parágrafo no
    meio — exceto quando um parágrafo isolado também ultrapassa
    max_chars, caso em que ele cai para _split_by_sentence().
    """
    paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > max_chars:
            if current:
                chunks.append('\n\n'.join(current))
                current, current_len = [], 0
            logger.info(
                "[Chunking] Parágrafo com %d chars (> %d) — subdividindo por frase.",
                para_len, max_chars,
            )
            chunks.extend(_split_by_sentence(para, max_chars))
            continue

        if current and current_len + para_len + 2 > max_chars:
            chunks.append('\n\n'.join(current))
            current, current_len = [], 0

        current.append(para)
        current_len += para_len + 2

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def _split_by_sentence(paragraph: str, max_chars: int) -> List[str]:
    """
    Último recurso: corta um parágrafo monstruosamente longo por frase,
    agrupando frases consecutivas até chegar perto de max_chars.
    """
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current and current_len + sent_len + 1 > max_chars:
            chunks.append(' '.join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(' '.join(current))

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