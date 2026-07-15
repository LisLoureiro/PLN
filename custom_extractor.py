"""
custom_extractor.py
Extrai itens estruturados de um documento com base em uma INSTRUÇÃO LIVRE
escrita pelo próprio usuário, em vez de um schema fixo (como era o caso do
extrator de obrigações do Agente Fiduciário em CPR-F).

O usuário delimita o que será extraído ANTES de qualquer coisa ser salva:
a instrução chega junto com o upload do PDF e guia tanto a busca dos
trechos relevantes (vectorizer.py) quanto o prompt enviado ao Ollama aqui.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    raise ImportError("Execute: pip install requests")


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de dados — genérico, sem categorias fixas
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedItem:
    """
    Representa um item extraído do documento conforme a instrução do usuário.

    O schema é livre — os campos dependem do que o usuário pediu.
    Extrai automaticamente:
      - trecho_referencia: cláusula/seção/trecho do documento (se existir no JSON)
      - resumo:            gerado automaticamente concatenando valores disponíveis
      - dados:             todos os campos do JSON original (nenhum dado é perdido)
    """

    def __init__(self, trecho_referencia: str, dados: Dict[str, Any]):
        self.trecho_referencia = trecho_referencia
        self.dados = dados
        # Gera resumo automaticamente concatenando valores de texto não vazios
        valores_texto = []
        for v in dados.values():
            if isinstance(v, str) and v.strip():
                valores_texto.append(v.strip())
        self.resumo = " | ".join(valores_texto)[:200] if valores_texto else "Item extraído"

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
Extraia informações de documentos seguindo esta instrução:

{instrucao}

Responda em formato JSON array com esta estrutura:
[
  {{
    "resumo": "resumo curto do item encontrado",
    "trecho_referencia": "trecho do documento que comprova o item",
    "dados": {{campo1: "valor1", campo2: "valor2"}}
  }}
]

Se nada encontrado, retorne: [].
"""


# ─────────────────────────────────────────────────────────────────────────────
# Extrator
# ─────────────────────────────────────────────────────────────────────────────

class CustomExtractor:
    """
    Recebe trechos relevantes de um documento + a instrução do usuário e usa
    o Ollama (modelo local) para extrair itens estruturados de formato livre.
    """

    MODEL = "qwen2.5:7b-instruct"
    MAX_TOKENS = 8192

    def __init__(self, base_url: Optional[str] = None):
        self._ollama_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")) + "/api/chat"
        logger.info("[Extractor] CustomExtractor inicializado com Ollama em %s", self._ollama_url)

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
        prompt = f"{instrucao}\n\nAnalise os trechos abaixo e retorne os itens em formato JSON array:\n\n{context}"

        logger.info("[Extractor] Chamando Ollama (%s) com %d chars de contexto…", self.MODEL, len(prompt))
        logger.info("[DEBUG] INSTRUÇÃO: %s", instrucao)
        logger.info("[DEBUG] CONTEXTO (primeiros 500 chars): %s...", context[:500])
        logger.info("[DEBUG] CONTEXTO (últimos 200 chars): ...%s", context[-200:])

        response = requests.post(
            self._ollama_url,
            json={
                "model": self.MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": self.MAX_TOKENS
                }
            },
            timeout=600
        )
        response.raise_for_status()

        raw = response.json()["message"]["content"]
        logger.info("[Extractor] Resposta recebida — %d chars.", len(raw))
        logger.info("[DEBUG] RESPOSTA BRUTA COMPLETA: %s", raw)
        return raw

    def _parse_response(self, raw: str) -> List[ExtractedItem]:
        clean = raw.strip()

        logger.info("[DEBUG] INICIANDO PARSING DA RESPOSTA")
        logger.info("[DEBUG] RESPOSTA BRUTA (limpa, primeiros 500 chars): %s", clean[:500])

        # ── PASSO 1: Tenta extrair bloco JSON de markdown fence (```json ... ``` ou ``` ... ```)
        # O bloco pode estar em qualquer lugar da resposta (não só no início)
        json_block_match = re.search(r'```(?:json)?\s*\n(.*?)```', clean, re.DOTALL | re.IGNORECASE)
        if json_block_match:
            clean = json_block_match.group(1).strip()
            logger.info("[DEBUG] Bloco JSON extraído de markdown fence (%d chars)", len(clean))
        else:
            logger.info("[DEBUG] Nenhum markdown fence detectado, tentando extração por brackets...")

            # ── PASSO 2: Se não tem fence, tenta extrair do primeiro '[' até o último ']'
            # (busca o primeiro '[' e então encontra o ']' correspondente)
            bracket_start = clean.find('[')
            if bracket_start == -1:
                # Tenta com '{' para objetos JSON
                brace_start = clean.find('{')
                if brace_start != -1:
                    # Encontra o '}' correspondente (contando abre/fecha)
                    depth = 0
                    for i, char in enumerate(clean[brace_start:], start=brace_start):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                clean = clean[brace_start:i+1]
                                logger.info("[DEBUG] Objeto JSON extraído por brackets (%d chars)", len(clean))
                                break
                    else:
                        logger.warning("[DEBUG] Não conseguiu encontrar '}' correspondente")
                else:
                    logger.warning("[DEBUG] Nenhum '[' ou '{' encontrado na resposta")
            else:
                # Encontra o ']' correspondente
                depth = 0
                for i, char in enumerate(clean[bracket_start:], start=bracket_start):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            clean = clean[bracket_start:i+1]
                            logger.info("[DEBUG] Array JSON extraído por brackets (%d chars)", len(clean))
                            break
                else:
                    logger.warning("[DEBUG] Não conseguiu encontrar ']' correspondente")

        # ── PASSO 3: Tenta fazer parse do JSON extraído
        logger.info("[DEBUG] Conteúdo a ser parseado (primeiros 200 chars): %s", clean[:200])
        try:
            data = json.loads(clean)
            logger.info("[DEBUG] JSON parseado com sucesso! Tipo: %s", type(data).__name__)
        except json.JSONDecodeError as e:
            logger.error("[Extractor] FALHA DE PARSING JSON: %s", e)
            logger.error("[Extractor] RESPOSTA BRUTA COMPLETA PARA DEBUG: %s", raw)
            logger.error("[Extractor] CONTEÚDO EXTRAÍDO PARA PARSING: %s", clean)
            logger.error("[Extractor] ⚠️ Isto é uma FALHA DE PARSING, não 'nenhum item encontrado'")
            return []

        # Se for um dict, tenta extrair a lista de uma propriedade comum
        if isinstance(data, dict):
            logger.warning("[Extractor] Resposta é um dict. Chaves: %s", list(data.keys()))
            logger.info("[DEBUG] Conteúdo do dict: %s", data)
            # Tenta encontrar uma propriedade que seja uma lista
            for key, value in data.items():
                if isinstance(value, list):
                    logger.info("[Extractor] Usando lista da propriedade '%s' (%d elementos)", key, len(value))
                    data = value
                    break
            else:
                # Se não encontrar lista no topo, verifica se há lista em 'dados'
                if 'dados' in data and isinstance(data['dados'], dict):
                    logger.info("[DEBUG] Procurando lista dentro de 'dados'...")
                    for key, value in data['dados'].items():
                        if isinstance(value, list) and value:
                            logger.info("[Extractor] Convertendo dict único com lista em dados.%s (%d elementos)", key, len(value))
                            logger.info("[DEBUG] Estrutura da lista em dados.%s: %s", key, value[0] if value else [])
                            # Converte o formato único em múltiplos itens
                            resumo_base = data.get('resumo', '')
                            trecho_ref = data.get('trecho_referencia', 'não identificado')
                            converted = []
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    novo_item = {
                                        'resumo': f"{resumo_base} - {item.get(list(item.keys())[0], '')}" if resumo_base else str(item.get(list(item.keys())[0], '')),
                                        'trecho_referencia': trecho_ref,
                                        'dados': item
                                    }
                                    logger.info("[DEBUG] Item convertido: %s", novo_item)
                                    converted.append(novo_item)
                            data = converted
                            logger.info("[DEBUG] Conversão completa: %d itens criados", len(data))
                            break
                    else:
                        logger.error("[Extractor] Dict em 'dados' não contém nenhuma lista: %s", data['dados'])
                        return []
                else:
                    logger.error("[Extractor] Dict não contém nenhuma lista e não tem 'dados': %s", data)
                    return []

        if not isinstance(data, list):
            logger.error("[Extractor] Após tentativas, resposta ainda não é uma lista: %s", type(data))
            logger.error("[DEBUG] Conteúdo final: %s", data)
            return []

        logger.info("[DEBUG] Parseando %d itens da lista...", len(data))
        items = []
        for i, item in enumerate(data):
            try:
                logger.info("[DEBUG] Processando item %d: %s", i, item)
                if not isinstance(item, dict):
                    logger.warning("[Extractor] Item %d não é objeto, ignorado. Tipo: %s", i, type(item))
                    continue

                # Extrai trecho_referencia se existir (campo especial)
                trecho = str(item.pop("trecho_referencia", "não identificado")).strip()

                # Verifica se tem pelo menos um campo de texto não vazio
                tem_conteudo = False
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        tem_conteudo = True
                        break
                    elif isinstance(v, (int, float, bool)):
                        tem_conteudo = True
                        break

                if not tem_conteudo:
                    logger.warning("[Extractor] Item %d sem conteúdo útil, ignorado. Conteúdo: %s", i, item)
                    continue

                # Todos os outros campos vão para 'dados' (schema livre)
                extracted_item = ExtractedItem(trecho_referencia=trecho, dados=item)
                items.append(extracted_item)
                logger.info("[DEBUG] Item %d parseado com sucesso: %s", i, extracted_item)
            except Exception as e:
                logger.warning("[Extractor] Erro no item %d: %s | %s", i, e, item)

        logger.info("[Extractor] %d itens parseados com sucesso.", len(items))
        return items
