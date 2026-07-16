"""
full_scan.py
Varredura completa: orquestra extrações múltiplas sobre um documento
contra todos os tipos de cláusula do catálogo (ou um subconjunto).

Executa em background com progresso persistente no banco via FullScanModel.
"""

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from clause_library import CLAUSE_TYPE_CATALOG, build_instruction_for_type
from custom_extractor import CustomExtractor
from store import Store
from vectorizer import Vectorizer

logger = logging.getLogger(__name__)


class FullScanEngine:
    """
    Orquestra varreduras completas sobre documentos.

    Fluxo:
    1. start_async() cria registro FullScanModel e inicia thread
    2. _run_scan_background() executa loop de extrações por tipo
    3. A cada tipo concluído, atualiza progresso no banco
    4. get_status() retorna estado atual + resultados parciais
    5. get_results_by_group() retorna resultados consolidados por grupo
    """

    def __init__(self, store: Store, vectorizer: Vectorizer, extractor: CustomExtractor):
        self.store = store
        self.vectorizer = vectorizer
        self.extractor = extractor

    def start_async(self, doc_hash: str, clause_types: Optional[List[str]] = None) -> str:
        """
        Inicia varredura completa em background.

        Args:
            doc_hash: Hash do documento (deve existir em DocumentModel)
            clause_types: Lista de tipos para scan (None = todos do catálogo)

        Returns:
            scan_id para polling via get_status()
        """
        # Verifica se documento existe
        doc = self.store.get_document_by_hash(doc_hash)
        if not doc:
            raise ValueError(f"Documento com hash {doc_hash[:8]} não encontrado.")

        # Resolve tipos: usa todos do catálogo se não especificado
        if clause_types is None:
            clause_types = [k for k in CLAUSE_TYPE_CATALOG.keys() if k != "outros"]
        else:
            # Valida tipos fornecidos
            valid_types = set(CLAUSE_TYPE_CATALOG.keys())
            invalid = [ct for ct in clause_types if ct not in valid_types]
            if invalid:
                raise ValueError(f"Tipos inválidos: {invalid}")

        # Cria registro FullScanModel
        scan_id = str(uuid.uuid4())[:8].upper()
        self.store.create_full_scan(
            scan_id=scan_id,
            doc_hash=doc_hash,
            clause_types=clause_types,
            total_types=len(clause_types),
        )

        # Inicia thread com o scan
        thread = threading.Thread(
            target=self._run_scan_background,
            args=(scan_id, doc["markdown"], clause_types),
            daemon=True,
        )
        thread.start()

        logger.info("[FullScan] Scan %s iniciado (%d tipos, doc: %s)",
                    scan_id, len(clause_types), doc_hash[:8])
        return scan_id

    def _run_scan_background(
        self,
        scan_id: str,
        markdown: str,
        clause_types: List[str],
    ) -> None:
        """
        Executa o loop de extrações em background.

        Para cada clause_type:
        1. Resolve instrução via build_instruction_for_type()
        2. Indexa chunks do markdown
        3. Busca chunks relevantes
        4. Extrai itens via CustomExtractor
        5. Tagueia itens com clause_type
        6. Atualiza progresso no banco
        """
        results = {}
        total = len(clause_types)
        erro = None

        try:
            # Indexa chunks UMA vez (reusa para todos os tipos)
            from app import _chunk_markdown
            chunks = _chunk_markdown(markdown)
            logger.info("[FullScan] Scan %s: %d chunks gerados", scan_id, len(chunks))

            self.vectorizer.index_chunks(chunks)

            # Processa cada tipo
            for i, clause_type in enumerate(clause_types, 1):
                try:
                    logger.info("[FullScan] Scan %s: processando %s (%d/%d)",
                                scan_id, clause_type, i, total)

                    # Resolve instrução
                    instrucao = build_instruction_for_type(clause_type)
                    logger.debug("[FullScan] Instrução para %s: %s", clause_type, instrucao[:100])

                    # Busca chunks relevantes
                    relevant_chunks = self.vectorizer.search(instrucao)
                    logger.debug("[FullScan] %d chunks relevantes para %s", len(relevant_chunks), clause_type)

                    # Extrai itens
                    items = self.extractor.extract(relevant_chunks, instrucao)

                    # Tagueia itens com clause_type
                    tagged_items = []
                    for item in items:
                        item_dict = item.to_dict()
                        item_dict["clause_type"] = clause_type
                        item_dict["clause_type_label"] = self._get_label_for_type(clause_type)
                        tagged_items.append(item_dict)

                    results[clause_type] = tagged_items
                    logger.info("[FullScan] Scan %s: %s → %d itens", scan_id, clause_type, len(tagged_items))

                    # Atualiza progresso
                    self.store.update_full_scan_progress(scan_id, i, results)

                except Exception as e:
                    logger.exception("[FullScan] Erro no tipo %s: %s", clause_type, e)
                    results[clause_type] = [{"error": str(e)}]
                    self.store.update_full_scan_progress(scan_id, i, results)

            # Scan completado com sucesso
            self.store.complete_full_scan(scan_id, results)
            logger.info("[FullScan] Scan %s completado: %d tipos processados", scan_id, total)

        except Exception as e:
            erro = str(e)
            logger.exception("[FullScan] Scan %s terminou com erro fatal: %s", scan_id, erro)
            self.store.complete_full_scan(scan_id, results or {}, error=erro)

    def get_status(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retorna status atual do scan + progresso."""
        scan = self.store.get_full_scan(scan_id)
        if not scan:
            return None

        return {
            "scan_id": scan["scan_id"],
            "status": scan["status"],
            "progress": f"{scan['processed_types']}/{scan['total_types']}",
            "processed_types": scan["processed_types"],
            "total_types": scan["total_types"],
            "created_at": scan["created_at"],
            "completed_at": scan.get("completed_at"),
            "error_message": scan.get("error_message"),
            "has_partial_results": bool(scan.get("results")),
        }

    def get_results_by_group(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retorna resultados consolidados agrupados por Grupo 1/2/3/4.

        Estrutura:
        {
            "scan_id": "...",
            "status": "completed",
            "doc_hash": "...",
            "groups": {
                "1 - Contratuais clássicos": {
                    "dano_moral": {"label": "...", "total": 2, "items": [...]},
                    ...
                },
                "2 - Contencioso e petições": {...},
                "3 - Direito Civil material": {...},
                "4 - Requisitos processuais": {...}
            }
        }
        """
        scan = self.store.get_full_scan(scan_id)
        if not scan:
            return None

        # Agrupa resultados por grupo
        grouped = self._group_results_by_category(scan["results"])

        return {
            "scan_id": scan["scan_id"],
            "status": scan["status"],
            "doc_hash": scan["doc_hash"],
            "created_at": scan["created_at"],
            "completed_at": scan.get("completed_at"),
            "total_types": scan["total_types"],
            "processed_types": scan["processed_types"],
            "groups": grouped,
        }

    def _get_label_for_type(self, clause_type: str) -> str:
        """Retorna label legível para um clause_type."""
        if clause_type in CLAUSE_TYPE_CATALOG:
            return CLAUSE_TYPE_CATALOG[clause_type]["label"]
        return clause_type.replace("_", " ").title()

    def _group_results_by_category(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Agrupa resultados por Grupo 1/2/3/4 conforme comentários em clause_library.py.

        Mapeamento (baseado nos valores do enum ClauseType):
        - Grupo 1: Contratuais clássicos (FORCE_MAJEURE ... SOLICITATION)
        - Grupo 2: Contencioso e petições (PRESCRICAO_DECADENCIA ... FUNDAMENTACAO_CONSTITUCIONAL)
        - Grupo 3: Direito Civil material (RESPONSABILIDADE_CIVIL_EXTRACONTRATUAL ... DIREITO_SUCESSOES)
        - Grupo 4: Requisitos processuais (REQUISITOS_PETICAO_INICIAL_ART319)
        """
        from clause_library import ClauseType

        GROUPS = {
            "1 - Contratuais clássicos": [],
            "2 - Contencioso e petições": [],
            "3 - Direito Civil material": [],
            "4 - Requisitos processuais": [],
        }

        # Define grupos pelos valores do enum (não por índices)
        group_1 = {
            ClauseType.FORCE_MAJEURE.value,
            ClauseType.TERMINATION.value,
            ClauseType.PENALTY.value,
            ClauseType.PRICE_ADJUSTMENT.value,
            ClauseType.CONFIDENTIALITY.value,
            ClauseType.LIABILITY.value,
            ClauseType.INTELLECTUAL_PROPERTY.value,
            ClauseType.DISPUTE_RESOLUTION.value,
            ClauseType.PAYMENT.value,
            ClauseType.WARRANTY.value,
            ClauseType.INDEMNITY.value,
            ClauseType.CHANGE_OF_CONTROL.value,
            ClauseType.ASSIGNMENT.value,
            ClauseType.NON_COMPETE.value,
            ClauseType.SOLICITATION.value,
        }

        group_2 = {
            ClauseType.PRESCRICAO_DECADENCIA.value,
            ClauseType.HONORARIOS_ADVOCATICIOS.value,
            ClauseType.CORRECAO_MONETARIA_JUROS.value,
            ClauseType.TUTELA_LIMINAR.value,
            ClauseType.REGULARIDADE_FISCAL_CND.value,
            ClauseType.PROTESTO_NOTIFICACAO_EDITAL.value,
            ClauseType.MULTA_MORA_TRIBUTARIA.value,
            ClauseType.BASE_CALCULO_TRIBUTO.value,
            ClauseType.COMPENSACAO_RESTITUICAO_INDEBITO.value,
            ClauseType.ACAO_COLETIVA_SUBSTITUICAO.value,
            ClauseType.OBRIGACAO_ACESSORIA_FISCAL.value,
            ClauseType.DIVIDA_ATIVA_CDA.value,
            ClauseType.PEDIDOS_PROCESSUAIS.value,
            ClauseType.FUNDAMENTACAO_CONSTITUCIONAL.value,
        }

        group_3 = {
            ClauseType.RESPONSABILIDADE_CIVIL_EXTRACONTRATUAL.value,
            ClauseType.DANO_MORAL.value,
            ClauseType.DANO_MATERIAL_LUCROS_CESSANTES.value,
            ClauseType.NEXO_CAUSALIDADE_CULPA.value,
            ClauseType.ENRIQUECIMENTO_SEM_CAUSA.value,
            ClauseType.PRESCRICAO_CIVIL_CC.value,
            ClauseType.CLAUSULA_PENAL_CIVIL.value,
            ClauseType.VICIO_REDIBITORIO_EVICCAO.value,
            ClauseType.OBRIGACAO_FAZER_NAO_FAZER.value,
            ClauseType.POSSE_PROPRIEDADE_USUCAPIAO.value,
            ClauseType.CONTRATO_CIVIL_TIPICO.value,
            ClauseType.DIREITO_FAMILIA_ALIMENTOS.value,
            ClauseType.DIREITO_SUCESSOES.value,
        }

        group_4 = {
            ClauseType.REQUISITOS_PETICAO_INICIAL_ART319.value,
        }

        group_mapping = {
            "1 - Contratuais clássicos": group_1,
            "2 - Contencioso e petições": group_2,
            "3 - Direito Civil material": group_3,
            "4 - Requisitos processuais": group_4,
        }

        # Distribui resultados pelos grupos
        for clause_type, items in results.items():
            group_name = None
            for g, types_set in group_mapping.items():
                if clause_type in types_set:
                    group_name = g
                    break

            if group_name:
                GROUPS[group_name].append({
                    "clause_type": clause_type,
                    "label": self._get_label_for_type(clause_type),
                    "total": len(items),
                    "items": items,
                })
            else:
                logger.warning("[FullScan] Tipo %s não encontrado em nenhum grupo, ignorando.", clause_type)

        # Converte listas em dicts para resposta
        return {g: {item["clause_type"]: item for item in items} for g, items in GROUPS.items()}
