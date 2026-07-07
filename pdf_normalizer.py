"""
pdf_normalizer.py
Converte um PDF para Markdown estruturado e normalizado.
Genérico — não depende do tipo de documento nem do que será extraído dele.

Estratégia:
  - Detecta títulos por tamanho/negrito de fonte
  - Preserva estrutura de seções (cláusulas, parágrafos, itens)
  - Remove cabeçalhos/rodapés repetidos
  - Normaliza espaços, hifenização e encoding
  - Processa páginas em lotes para evitar estouro de memória
"""

import gc
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:
    raise ImportError("Execute: pip install pdfplumber")


TITLE_SIZE_THRESHOLD   = 13.0
SECTION_SIZE_THRESHOLD = 11.5
BATCH_SIZE = 20

RE_CLAUSULA = re.compile(
    r"^(CL[ÁA]USULA\s+\d+[\w\.]*|"
    r"SE[ÇC][ÃA]O\s+[IVXLCDM\d]+|"
    r"\d{1,2}\.\d{0,2}\.?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇÜ])",
    re.IGNORECASE,
)

RE_NUMBERED   = re.compile(r"^\d+\.\d+")
RE_BLANK_MANY = re.compile(r"\n{3,}")
RE_HYPHEN_BR  = re.compile(r"-\n(\w)")


class PDFNormalizer:
    """
    Transforma um PDF em Markdown estruturado e normalizado.

    Uso:
        norm = PDFNormalizer("documento.pdf")
        md   = norm.to_markdown()
    """

    def __init__(self, pdf_path: Union[str, Path]):
        self._path = Path(pdf_path)
        if not self._path.exists():
            raise FileNotFoundError(f"PDF não encontrado: {self._path}")

    def to_markdown(self) -> str:
        pages = self._extract_pages()

        cleaned = self._remove_repeated_headers_footers(pages)
        del pages
        gc.collect()

        md = self._pages_to_markdown(cleaned)
        del cleaned
        gc.collect()

        md = self._post_process(md)

        logger.info(
            "[Normalizer] MD gerado: %d chars / ~%d linhas",
            len(md), md.count("\n"),
        )
        return md

    def _extract_pages(self) -> List[List[dict]]:
        pages = []
        with pdfplumber.open(str(self._path)) as pdf:
            total = len(pdf.pages)
            logger.info("[Normalizer] %d páginas no PDF.", total)

            for i in range(0, total, BATCH_SIZE):
                batch = pdf.pages[i:i + BATCH_SIZE]
                for j, page in enumerate(batch, i + 1):
                    blocks = self._page_to_blocks(page)
                    pages.append(blocks)
                    logger.debug("[Normalizer] Página %d/%d — %d blocos", j, total, len(blocks))
                del batch
                gc.collect()
        return pages

    def _page_to_blocks(self, page) -> List[dict]:
        words = page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
        if not words:
            return []

        lines: List[List[dict]] = []
        cur_line: List[dict] = []
        cur_top = None

        for w in words:
            top = round(w.get("top", 0), 0)
            if cur_top is None or abs(top - cur_top) < 3:
                cur_line.append(w)
                cur_top = top
            else:
                if cur_line:
                    lines.append(cur_line)
                cur_line = [w]
                cur_top = top

        if cur_line:
            lines.append(cur_line)

        blocks = []
        for line in lines:
            text = " ".join(w["text"] for w in line).strip()
            if not text:
                continue
            avg_size = sum(w.get("size", 10) for w in line) / len(line)
            is_bold = any(
                "bold" in w.get("fontname", "").lower() or "heavy" in w.get("fontname", "").lower()
                for w in line
            )
            x0 = line[0].get("x0", 0)
            blocks.append({"text": text, "size": avg_size, "bold": is_bold, "x0": x0})

        del words, lines
        gc.collect()
        return blocks

    def _remove_repeated_headers_footers(self, pages: List[List[dict]]) -> List[List[dict]]:
        if len(pages) < 3:
            return pages

        from collections import Counter
        freq: Counter = Counter()

        for pg in pages:
            seen = set()
            for b in pg:
                t = b["text"].strip()
                if t and t not in seen:
                    freq[t] += 1
                    seen.add(t)

        threshold = max(2, int(len(pages) * 0.4))
        repeating = {t for t, c in freq.items() if c >= threshold}

        if repeating:
            logger.info("[Normalizer] %d textos repetidos removidos (cabeçalho/rodapé).", len(repeating))

        cleaned = [[b for b in pg if b["text"].strip() not in repeating] for pg in pages]
        del freq
        gc.collect()
        return cleaned

    def _pages_to_markdown(self, pages: List[List[dict]]) -> str:
        lines_md: List[str] = []
        for pg_i, blocks in enumerate(pages):
            for b in blocks:
                text = b["text"].strip()
                if not text:
                    continue
                lines_md.append(self._classify_block(text, b["size"], b["bold"]))
            lines_md.append("")
            if pg_i % 25 == 0:
                gc.collect()
        return "\n".join(lines_md)

    def _classify_block(self, text: str, size: float, bold: bool) -> str:
        upper = text.upper() == text and len(text) > 4

        if size >= TITLE_SIZE_THRESHOLD or (bold and upper and len(text) < 80):
            return f"\n## {text}\n"
        if size >= SECTION_SIZE_THRESHOLD or RE_CLAUSULA.match(text):
            return f"\n### {text}\n"
        if RE_NUMBERED.match(text):
            return f"\n#### {text}\n"
        return text

    def _post_process(self, md: str) -> str:
        md = RE_HYPHEN_BR.sub(r"\1", md)
        md = unicodedata.normalize("NFC", md)
        md = "\n".join(l.rstrip() for l in md.split("\n"))
        md = RE_BLANK_MANY.sub("\n\n", md)
        md = re.sub(r"(\n---\n)+", "\n---\n", md)
        return md.strip()
