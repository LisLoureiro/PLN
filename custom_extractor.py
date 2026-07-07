"""
custom_extractor.py
Extrai itens estruturados de um documento com base em uma INSTRUÇÃO LIVRE
escrita pelo próprio usuário, em vez de um schema fixo (como era o caso do
extrator de obrigações do Agente Fiduciário em CPR-F).

O usuário delimita o que será extraído ANTES de qualquer coisa ser salva:
a instrução chega junto com o upload do PDF e guia tanto a busca dos
trechos relevantes (vectorizer.py) quanto o prompt enviado ao Claude aqui.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    raise ImportError("Execute: pip install anthropic")


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de dados — genérico, sem categorias fixas
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedItem:
    """
    Representa um item extraído do documento conforme a instrução do usuário.

    Diferente do CPR-F (categorias e campos fixos), aqui o item é um objeto
    JSON de formato livre — os campos dependem do que o usuário pediu.
    Dois campos são sempre garantidos para permitir listagem/busca genérica:
      - resumo:            frase curta e autoexplicativa do item
      - trecho_referencia: cláusula/seção/trecho do documento que embasa o item
    Qualquer outro campo pedido pelo usuário fica em `dados`.
    """

    def __init__(self, resumo: str, trecho_referencia: str, dados: Dict[str, Any]):
        self.resumo = resumo
        self.trecho_referencia = trecho_referencia
        self.dados = dados

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resumo": self.resumo,
            "trecho_referencia": self.trecho_referencia,
            "dados": self.dados,
        }

    def __repr__(self) -> str:
        return f"ExtractedItem(resumo={self.resumo[:60]!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(instrucao: str) -> str:
    return f"""\
Você é um assistente especialista em leitura e extração de informações de
documentos (contratos, escrituras, editais, laudos, atas, relatórios etc).

O usuário forneceu a seguinte INSTRUÇÃO, que define EXATAMENTE o que deve
ser extraído do documento. Siga-a com precisão — não extraia nada fora do
que ela pede, e não invente informação que não esteja no texto.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÃO DO USUÁRIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instrucao}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO DE SAÍDA (OBRIGATÓRIO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Retorne SOMENTE um array JSON válido (nada de markdown, nada de texto fora
do array). Cada elemento do array é um objeto com esta estrutura:

{{
  "resumo": string
      Frase curta (até ~20 palavras), autoexplicativa, resumindo o item
      encontrado — deve fazer sentido sozinha, sem depender do trecho original.

  "trecho_referencia": string
      Cláusula, seção, artigo, página ou citação literal (curta) do
      documento que comprova/embasa este item.
      Se não for identificável, use "não identificado".

  "dados": object
      Objeto livre com os campos específicos pedidos na instrução do
      usuário (nomes de chave em snake_case, em português, coerentes com
      o que foi pedido). Use os valores exatamente como aparecem no
      documento, sem reformular números, datas ou nomes próprios.
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS OBRIGATÓRIAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Retorne SOMENTE o array JSON — sem ```json, sem comentários, sem texto extra.
2. NÃO invente dados que não estejam no texto fornecido.
3. Se a mesma informação aparecer repetida em mais de um trecho, consolide
   em um único item (combine as referências em "trecho_referencia").
4. Se nada relevante para a instrução for encontrado nos trechos fornecidos,
   retorne exatamente: []
5. Não extraia informações fora do escopo da instrução do usuário, mesmo
   que pareçam interessantes.
6. Seja preciso com valores, datas, nomes e percentuais — copie como estão
   no documento.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Extrator
# ─────────────────────────────────────────────────────────────────────────────

class CustomExtractor:
    """
    Recebe trechos relevantes de um documento + a instrução do usuário e usa
    o Claude para extrair itens estruturados de formato livre.
    """

    MODEL = "claude-opus-4-5-20251101"
    MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("Defina ANTHROPIC_API_KEY ou passe api_key ao construtor.")
        self._client = anthropic.Anthropic(api_key=key)

    def extract(self, chunks: List[str], instrucao: str) -> List[ExtractedItem]:
        """Envia chunks + instrução ao Claude e retorna lista de ExtractedItem."""
        if not chunks:
            logger.warning("[Extractor] Nenhum chunk recebido.")
            return []
        if not instrucao or not instrucao.strip():
            raise ValueError("Instrução de extração vazia — o usuário precisa dizer o que extrair.")

        context = self._build_context(chunks)
        raw_json = self._call_api(context, instrucao)
        return self._parse_response(raw_json)

    def _build_context(self, chunks: List[str]) -> str:
        return "\n\n--- TRECHO ---\n\n".join(chunks)

    def _call_api(self, context: str, instrucao: str) -> str:
        system_prompt = _build_system_prompt(instrucao)
        user_msg = (
            "Analise os trechos abaixo, extraídos de um documento, e retorne "
            "os itens conforme a instrução e o formato JSON especificado.\n\n"
            f"{context}"
        )

        logger.info("[Extractor] Chamando Claude (%s) com %d chars de contexto…", self.MODEL, len(user_msg))

        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text
        logger.info("[Extractor] Resposta recebida — %d chars.", len(raw))
        return raw

    def _parse_response(self, raw: str) -> List[ExtractedItem]:
        clean = raw.strip()

        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
            clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error("[Extractor] JSON inválido: %s\nConteúdo: %s", e, raw[:500])
            return []

        if not isinstance(data, list):
            logger.error("[Extractor] Resposta não é uma lista: %s", type(data))
            return []

        items = []
        for i, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    logger.warning("[Extractor] Item %d não é objeto, ignorado.", i)
                    continue
                resumo = str(item.get("resumo", "")).strip()
                if not resumo:
                    logger.warning("[Extractor] Item %d sem resumo, ignorado.", i)
                    continue
                trecho = str(item.get("trecho_referencia", "não identificado")).strip()
                dados = item.get("dados", {})
                if not isinstance(dados, dict):
                    dados = {"valor": dados}
                items.append(ExtractedItem(resumo=resumo, trecho_referencia=trecho, dados=dados))
            except Exception as e:
                logger.warning("[Extractor] Erro no item %d: %s | %s", i, e, item)

        logger.info("[Extractor] %d itens parseados com sucesso.", len(items))
        return items
