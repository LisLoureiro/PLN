"""
clause_library.py
Biblioteca de cláusulas aprovadas — Knowledge Management Interno.

Permite que advogados busquem precedentes de como cláusulas foram
redigidas em contratos anteriores da empresa, E TAMBÉM como determinados
tópicos recorrentes foram fundamentados/pedidos em petições anteriores
(cumprimento de sentença, mandado de segurança, ação de repetição de
indébito etc).

ALTERAÇÕES NESTA VERSÃO:
- Corrigido bug em _detect_clause_type (ClauseType.PAYMENT_Terms não existia).
- ClauseType ampliado com categorias de contencioso tributário/cível,
  identificadas a partir de petições reais (cumprimento de sentença,
  repetição de indébito, mandados de segurança sobre CND/protesto/PIS-
  COFINS/multa de mora do eSocial).
- Novo CLAUSE_TYPE_CATALOG: para cada tipo, guarda um "label" (nome amigável
  para exibição em um <select> no front-end) e uma "search_instruction"
  (descrição detalhada, escrita para ser usada como a INSTRUÇÃO enviada ao
  extrator/LLM — ou seja, o que o modelo deve efetivamente procurar no
  documento quando esse tipo for selecionado).
- Novas funções auxiliares list_clause_type_options() e
  build_instruction_for_type() para dar suporte a um seletor de tipo no
  prompt de extração, em vez de depender só de texto livre.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Boolean, JSON,
    ForeignKey, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ApprovalStatus(Enum):
    """Status de aprovação de uma cláusula."""
    PENDING = "pending"           # Aguardando aprovação
    APPROVED = "approved"         # Aprovada pelo advogado
    REJECTED = "rejected"         # Rejeitada
    MODIFIED = "modified"         # Aprovada com modificações


class ClauseType(Enum):
    """
    Tipos de cláusulas/tópicos comuns nos documentos da biblioteca.

    Dividido em dois grupos:
    1) Cláusulas contratuais "clássicas" (contratos em geral).
    2) Tópicos de contencioso/petições (identificados a partir de petições
       tributárias e cíveis federais reais — cumprimento de sentença,
       repetição de indébito, mandados de segurança).
    """
    # ── Grupo 1: cláusulas contratuais ──────────────────────────────────
    FORCE_MAJEURE = "forca_maior"
    TERMINATION = "rescisao"
    PENALTY = "multa"
    PRICE_ADJUSTMENT = "reajuste"
    CONFIDENTIALITY = "confidencialidade"
    LIABILITY = "responsabilidade"
    INTELLECTUAL_PROPERTY = "propriedade_intelectual"
    DISPUTE_RESOLUTION = "resolucao_disputas"
    PAYMENT = "pagamento"
    WARRANTY = "garantia"
    INDEMNITY = "indenizacao"
    CHANGE_OF_CONTROL = "mudanca_controle"
    ASSIGNMENT = "cessao"
    NON_COMPETE = "nao_concorrencia"
    SOLICITATION = "solicitacao"

    # ── Grupo 2: tópicos de contencioso / petições ──────────────────────
    PRESCRICAO_DECADENCIA = "prescricao_decadencia"
    HONORARIOS_ADVOCATICIOS = "honorarios_advocaticios"
    CORRECAO_MONETARIA_JUROS = "correcao_monetaria_juros"
    TUTELA_LIMINAR = "tutela_liminar"
    REGULARIDADE_FISCAL_CND = "regularidade_fiscal_cnd"
    PROTESTO_NOTIFICACAO_EDITAL = "protesto_notificacao_edital"
    MULTA_MORA_TRIBUTARIA = "multa_mora_tributaria"
    BASE_CALCULO_TRIBUTO = "base_calculo_tributo"
    COMPENSACAO_RESTITUICAO_INDEBITO = "compensacao_restituicao_indebito"
    ACAO_COLETIVA_SUBSTITUICAO = "acao_coletiva_substituicao"
    OBRIGACAO_ACESSORIA_FISCAL = "obrigacao_acessoria_fiscal"
    DIVIDA_ATIVA_CDA = "divida_ativa_cda"
    PEDIDOS_PROCESSUAIS = "pedidos_processuais"
    FUNDAMENTACAO_CONSTITUCIONAL = "fundamentacao_constitucional"

    # ── Grupo 3: Direito Civil material / termos jurídicos comuns ──────
    RESPONSABILIDADE_CIVIL_EXTRACONTRATUAL = "responsabilidade_civil_extracontratual"
    DANO_MORAL = "dano_moral"
    DANO_MATERIAL_LUCROS_CESSANTES = "dano_material_lucros_cessantes"
    NEXO_CAUSALIDADE_CULPA = "nexo_causalidade_culpa"
    ENRIQUECIMENTO_SEM_CAUSA = "enriquecimento_sem_causa"
    PRESCRICAO_CIVIL_CC = "prescricao_civil_cc"
    CLAUSULA_PENAL_CIVIL = "clausula_penal_civil"
    VICIO_REDIBITORIO_EVICCAO = "vicio_redibitorio_eviccao"
    OBRIGACAO_FAZER_NAO_FAZER = "obrigacao_fazer_nao_fazer"
    POSSE_PROPRIEDADE_USUCAPIAO = "posse_propriedade_usucapiao"
    CONTRATO_CIVIL_TIPICO = "contrato_civil_tipico"
    DIREITO_FAMILIA_ALIMENTOS = "direito_familia_alimentos"
    DIREITO_SUCESSOES = "direito_sucessoes"

    # ── Grupo 4: Requisitos processuais da petição inicial (CPC) ───────
    REQUISITOS_PETICAO_INICIAL_ART319 = "requisitos_peticao_inicial_art319"

    OTHER = "outros"


# ─────────────────────────────────────────────────────────────────────────
# Checklist explícito dos requisitos da petição inicial — Art. 319 do CPC
# (Lei 13.105/2015), incisos I a VII e §§1º a 2º.
#
# Cada entrada abaixo corresponde a UM requisito legal e documenta:
#   - "dispositivo":     o artigo/inciso exato da lei;
#   - "descricao":       o que a lei exige nesse requisito;
#   - "campo_esperado":  a chave que DEVE aparecer no JSON `dados` do item
#                        extraído para esse requisito — isto é o "dicionário
#                        explícito" do schema esperado, para não depender de
#                        o LLM inventar nomes de campo diferentes a cada
#                        execução.
#
# Este dict é a fonte única da verdade: tanto a instrução de extração
# (ver CLAUSE_TYPE_CATALOG[REQUISITOS_PETICAO_INICIAL_ART319]) quanto
# qualquer validação/checklist no front-end podem ser geradas a partir dele,
# em vez de duplicar a lista de requisitos em texto solto.
# ─────────────────────────────────────────────────────────────────────────
REQUISITOS_PETICAO_INICIAL_ART319: Dict[str, Dict[str, str]] = {
    "inciso_I_juizo": {
        "dispositivo": "Art. 319, I, CPC",
        "descricao": "O juízo (comarca/vara/foro/subseção judiciária) a que a petição é dirigida.",
        "campo_esperado": "juizo_destinatario",
    },
    "inciso_II_qualificacao_partes": {
        "dispositivo": "Art. 319, II, CPC",
        "descricao": (
            "Qualificação completa do autor E do réu — nome, prenome, "
            "estado civil, existência de união estável, profissão, "
            "número de inscrição no CPF (ou CNPJ, se pessoa jurídica), "
            "endereço eletrônico (e-mail), domicílio e residência. É UM "
            "único requisito legal que abrange as duas partes; informe a "
            "qualificação de cada uma (autor e réu) dentro do mesmo campo, "
            "e marque \"AUSENTE\" apenas se a qualificação de AMBAS "
            "estiver faltando — se só uma das partes estiver "
            "qualificada, descreva o que falta em vez de marcar ausente."
        ),
        "campo_esperado": "qualificacao_partes",
    },
    "inciso_III_fatos_fundamentos": {
        "dispositivo": "Art. 319, III, CPC",
        "descricao": (
            "O fato e os fundamentos jurídicos do pedido — a causa de "
            "pedir remota (fatos) e próxima (fundamentos legais/jurídicos "
            "que os fatos autorizam a invocar)."
        ),
        "campo_esperado": "causa_de_pedir",
    },
    "inciso_IV_pedido": {
        "dispositivo": "Art. 319, IV, CPC",
        "descricao": (
            "O pedido, com suas especificações — pedido certo e "
            "determinado (art. 322/324 CPC), incluindo pedidos "
            "cumulados, se houver."
        ),
        "campo_esperado": "pedidos",
    },
    "inciso_V_valor_causa": {
        "dispositivo": "Art. 319, V, CPC (c/c arts. 291 a 293 CPC)",
        "descricao": "O valor atribuído à causa, e o critério usado para chegar a ele, se explicitado.",
        "campo_esperado": "valor_da_causa",
    },
    "inciso_VI_provas": {
        "dispositivo": "Art. 319, VI, CPC",
        "descricao": (
            "As provas com que o autor pretende demonstrar a verdade dos "
            "fatos alegados (documental, testemunhal, pericial, "
            "depoimento pessoal, etc.)."
        ),
        "campo_esperado": "provas_requeridas",
    },
    "inciso_VII_opcao_audiencia": {
        "dispositivo": "Art. 319, VII, CPC",
        "descricao": (
            "A opção expressa do autor pela realização, ou não, de "
            "audiência de conciliação ou de mediação (art. 334 CPC)."
        ),
        "campo_esperado": "opcao_audiencia_conciliacao",
    },
    "paragrafo_1_diligencia_dados_ausentes": {
        "dispositivo": "Art. 319, §1º, CPC",
        "descricao": (
            "Caso o autor não disponha das informações do inciso II, "
            "requerimento ao juízo de diligências necessárias para "
            "obtê-las (ex.: expedição de ofício a órgãos públicos)."
        ),
        "campo_esperado": "requerimento_diligencia_dados_reu",
    },
    "paragrafo_2_recusa_expressa_autocomposicao": {
        "dispositivo": "Art. 319, §2º, CPC",
        "descricao": (
            "Se o autor optar por não realizar a autocomposição, essa "
            "recusa deve constar de forma expressa na petição (o "
            "silêncio não basta)."
        ),
        "campo_esperado": "recusa_expressa_autocomposicao",
    },
}


def _build_peticao_inicial_art319_instruction() -> str:
    """
    Monta a instrução de extração do checklist do art. 319 do CPC a partir
    de REQUISITOS_PETICAO_INICIAL_ART319, para que a lista de requisitos
    nunca fique dessincronizada entre o texto do prompt e o schema
    documentado no código.
    """
    total = len(REQUISITOS_PETICAO_INICIAL_ART319)

    linhas = [
        f"Você está analisando uma PETIÇÃO INICIAL civil. O art. 319 do "
        f"CPC (Lei 13.105/2015) exige EXATAMENTE {total} requisitos "
        f"obrigatórios — listados abaixo — e você precisa verificar "
        f"TODOS os {total}, um a um, sem pular nenhum e sem parar de "
        f"procurar após encontrar os primeiros.",
        "",
        f"IMPORTANTE: os {total} requisitos normalmente NÃO estão "
        "concentrados numa única seção — eles ficam espalhados por "
        "partes bem diferentes da petição: o cabeçalho de abertura "
        "(juízo), o parágrafo de qualificação das partes (logo após o "
        "cabeçalho), o corpo da narrativa (fatos e fundamentos jurídicos, "
        "geralmente sob títulos como \"DOS FATOS\" e \"DO DIREITO\"), a "
        "seção de pedidos (geralmente no final, sob \"DOS PEDIDOS\" ou "
        "\"ANTE O EXPOSTO\"), a linha do valor da causa (quase sempre nas "
        "últimas linhas, antes da assinatura), e a manifestação sobre "
        "audiência de conciliação (pode aparecer tanto no meio do texto "
        "quanto num dos itens finais do pedido). Portanto, use o "
        f"documento INTEIRO como referência — não restrinja a busca a um "
        "único trecho ou seção, mesmo que pareça o lugar mais óbvio.",
        "",
        f"Retorne um ÚNICO item, cujo campo `dados` seja um objeto JSON "
        f"contendo EXATAMENTE estas {total} chaves (uma por requisito "
        "legal, nem mais nem menos). Para cada chave, informe o conteúdo "
        "encontrado no documento; se o requisito não constar da petição, "
        "use o valor \"AUSENTE\" — nunca omita uma chave, mesmo que o "
        "requisito correspondente não tenha sido encontrado.",
        "",
    ]
    for i, info in enumerate(REQUISITOS_PETICAO_INICIAL_ART319.values(), start=1):
        linhas.append(
            f"{i}. \"{info['campo_esperado']}\" ({info['dispositivo']}): "
            f"{info['descricao']}"
        )
    linhas.append("")
    linhas.append(
        f"Antes de responder, confira que os {total} itens acima foram "
        "todos avaliados (presente ou AUSENTE) — não é aceitável "
        "verificar só os requisitos mais fáceis de achar e ignorar o "
        "resto. No campo `trecho_referencia`, cite o(s) trecho(s) do "
        "documento que fundamentam sua avaliação de quais requisitos "
        f"estão presentes ou ausentes. No campo `resumo`, informe "
        f"quantos dos {total} requisitos do art. 319 foram atendidos "
        f"(ex.: \"6 de {total} requisitos atendidos — ausente: valor da "
        "causa e opção de audiência\")."
    )
    return "\n".join(linhas)


# ─────────────────────────────────────────────────────────────────────────
# Tipos que exigem COBERTURA AMPLA do documento inteiro, em vez da busca
# TF-IDF normal (que só seleciona os trechos mais "parecidos" com a
# instrução). O checklist do art. 319 do CPC é um checklist ESTRUTURAL —
# seus 9 requisitos ficam espalhados por partes muito diferentes da
# petição (cabeçalho, qualificação, fatos, pedidos, fecho), muitas vezes
# em trechos que não compartilham vocabulário com a instrução em si (ex.:
# a linha do valor da causa não menciona "requisito" nem "CPC"). Por isso,
# para estes tipos, app.py deve enviar TODOS os chunks do documento ao
# extrator, em vez de filtrar pelos mais relevantes via Vectorizer.
# ─────────────────────────────────────────────────────────────────────────
FULL_DOCUMENT_CLAUSE_TYPES = {
    ClauseType.REQUISITOS_PETICAO_INICIAL_ART319.value,
}


# ─────────────────────────────────────────────────────────────────────────
# Catálogo de tipos: label amigável + instrução de busca para o extrator.
#
# `search_instruction` é escrita para ser usada DIRETAMENTE como o campo
# `instrucao` do CustomExtractor (ver custom_extractor.py) — ou seja, deve
# ser específica o bastante para orientar tanto o TF-IDF (seleção de
# trechos relevantes) quanto o prompt final do LLM.
# ─────────────────────────────────────────────────────────────────────────
CLAUSE_TYPE_CATALOG: Dict[str, Dict[str, str]] = {
    # ── Contratuais ─────────────────────────────────────────────────────
    ClauseType.FORCE_MAJEURE.value: {
        "label": "Força maior / caso fortuito",
        "search_instruction": (
            "Extraia todas as cláusulas que tratam de força maior, caso "
            "fortuito ou eventos alheios à vontade das partes. Para cada "
            "uma, informe: o evento que exime a responsabilidade, se há "
            "prazo de suspensão das obrigações, e a cláusula de referência."
        ),
    },
    ClauseType.TERMINATION.value: {
        "label": "Rescisão / término do contrato",
        "search_instruction": (
            "Extraia todas as cláusulas sobre rescisão, resilição ou "
            "término do contrato. Informe: motivo/hipótese de rescisão, "
            "se há aviso prévio ou notificação exigida, eventual multa "
            "rescisória e a cláusula de referência."
        ),
    },
    ClauseType.PENALTY.value: {
        "label": "Multa / penalidade contratual",
        "search_instruction": (
            "Extraia todas as cláusulas de multa, penalidade ou mora "
            "contratual. Informe: valor ou percentual da multa, o evento "
            "que a gera (inadimplemento, atraso, rescisão etc.) e a "
            "cláusula de referência."
        ),
    },
    ClauseType.PRICE_ADJUSTMENT.value: {
        "label": "Reajuste de preço/valor",
        "search_instruction": (
            "Extraia todas as cláusulas de reajuste de preço, aluguel ou "
            "valor do contrato. Informe: índice usado (ex.: IGP-M, IPCA), "
            "periodicidade do reajuste, data-base e a cláusula de "
            "referência."
        ),
    },
    ClauseType.CONFIDENTIALITY.value: {
        "label": "Confidencialidade / sigilo",
        "search_instruction": (
            "Extraia todas as cláusulas de confidencialidade ou sigilo de "
            "informações. Informe: o que é considerado informação "
            "confidencial, prazo de vigência da obrigação de sigilo após "
            "o término do contrato, e a cláusula de referência."
        ),
    },
    ClauseType.LIABILITY.value: {
        "label": "Responsabilidade civil / limitação de responsabilidade",
        "search_instruction": (
            "Extraia todas as cláusulas de responsabilidade civil ou "
            "limitação de responsabilidade entre as partes. Informe: "
            "limite de valor (se houver), tipos de dano cobertos ou "
            "excluídos, e a cláusula de referência."
        ),
    },
    ClauseType.INTELLECTUAL_PROPERTY.value: {
        "label": "Propriedade intelectual",
        "search_instruction": (
            "Extraia todas as cláusulas sobre propriedade intelectual, "
            "direitos autorais ou titularidade de criações. Informe: a "
            "quem pertence o direito, se há licença de uso, e a cláusula "
            "de referência."
        ),
    },
    ClauseType.DISPUTE_RESOLUTION.value: {
        "label": "Resolução de disputas (foro / arbitragem)",
        "search_instruction": (
            "Extraia todas as cláusulas de eleição de foro ou de "
            "arbitragem para resolução de disputas. Informe: foro ou "
            "câmara arbitral eleitos, regras aplicáveis, e a cláusula de "
            "referência."
        ),
    },
    ClauseType.PAYMENT.value: {
        "label": "Pagamento / forma de pagamento",
        "search_instruction": (
            "Extraia todas as cláusulas sobre forma, prazo e condições de "
            "pagamento. Informe: valor, periodicidade (mensal, parcela "
            "única etc.), meio de pagamento e a cláusula de referência."
        ),
    },
    ClauseType.WARRANTY.value: {
        "label": "Garantia",
        "search_instruction": (
            "Extraia todas as cláusulas de garantia (de produto, serviço "
            "ou performance). Informe: prazo de garantia, o que está "
            "coberto, exclusões, e a cláusula de referência."
        ),
    },
    ClauseType.INDEMNITY.value: {
        "label": "Indenização",
        "search_instruction": (
            "Extraia todas as cláusulas de indenização (indemnity). "
            "Informe: hipóteses que geram o dever de indenizar, limite de "
            "valor se houver, e a cláusula de referência."
        ),
    },
    ClauseType.CHANGE_OF_CONTROL.value: {
        "label": "Mudança de controle societário",
        "search_instruction": (
            "Extraia todas as cláusulas relativas a mudança de controle "
            "societário. Informe: o que se considera mudança de controle, "
            "consequência contratual (ex.: direito de rescisão), e a "
            "cláusula de referência."
        ),
    },
    ClauseType.ASSIGNMENT.value: {
        "label": "Cessão de contrato/direitos",
        "search_instruction": (
            "Extraia todas as cláusulas sobre cessão do contrato ou de "
            "direitos/obrigações a terceiros. Informe: se exige anuência "
            "prévia da outra parte, condições, e a cláusula de referência."
        ),
    },
    ClauseType.NON_COMPETE.value: {
        "label": "Não concorrência",
        "search_instruction": (
            "Extraia todas as cláusulas de não concorrência. Informe: "
            "prazo de vigência da restrição, escopo geográfico/setorial, "
            "e a cláusula de referência."
        ),
    },
    ClauseType.SOLICITATION.value: {
        "label": "Não aliciamento (solicitation)",
        "search_instruction": (
            "Extraia todas as cláusulas de não aliciamento de funcionários "
            "ou clientes. Informe: prazo de vigência, escopo, e a cláusula "
            "de referência."
        ),
    },

    # ── Contencioso / petições ──────────────────────────────────────────
    ClauseType.PRESCRICAO_DECADENCIA.value: {
        "label": "Prescrição / decadência",
        "search_instruction": (
            "Extraia todas as passagens que tratam de prescrição ou "
            "decadência do direito de cobrança/restituição, incluindo "
            "prescrição quinquenal em matéria tributária. Informe: o "
            "prazo aplicado, o marco inicial de contagem (ex.: ajuizamento "
            "da ação, recolhimento indevido) e o trecho de referência."
        ),
    },
    ClauseType.HONORARIOS_ADVOCATICIOS.value: {
        "label": "Honorários advocatícios",
        "search_instruction": (
            "Extraia todas as passagens sobre honorários advocatícios "
            "(sucumbenciais, contratuais ou em execução individual de "
            "sentença coletiva). Informe: fundamento legal ou "
            "jurisprudencial citado (ex.: Súmula 345 do STJ), percentual "
            "ou critério de fixação pedido, e o trecho de referência."
        ),
    },
    ClauseType.CORRECAO_MONETARIA_JUROS.value: {
        "label": "Correção monetária / juros (SELIC)",
        "search_instruction": (
            "Extraia todas as passagens sobre correção monetária e juros "
            "de mora aplicados a valores discutidos na ação, incluindo "
            "referências à taxa SELIC. Informe: índice/taxa usado, termo "
            "inicial da incidência, e o trecho de referência."
        ),
    },
    ClauseType.TUTELA_LIMINAR.value: {
        "label": "Tutela liminar / urgência / evidência",
        "search_instruction": (
            "Extraia todos os argumentos usados para pedir a concessão de "
            "medida liminar, tutela de urgência ou tutela de evidência. "
            "Para cada um, informe: se é fumus boni iuris ou periculum in "
            "mora, o argumento central, o fundamento legal citado (ex.: "
            "art. 7º, III, da Lei 12.016/2009; art. 300 ou 311 do CPC), e "
            "o trecho de referência."
        ),
    },
    ClauseType.REGULARIDADE_FISCAL_CND.value: {
        "label": "Regularidade fiscal / CND / CPEN",
        "search_instruction": (
            "Extraia todas as passagens relacionadas à necessidade ou "
            "negativa de emissão de Certidão Negativa de Débito (CND) ou "
            "Certidão Positiva com Efeitos de Negativa (CPEN). Informe: "
            "motivo alegado para a restrição, impacto na atividade da "
            "empresa (ex.: licitações, contratos), e o trecho de "
            "referência."
        ),
    },
    ClauseType.PROTESTO_NOTIFICACAO_EDITAL.value: {
        "label": "Protesto de título/CDA e notificação por edital",
        "search_instruction": (
            "Extraia todas as passagens sobre protesto de título ou CDA "
            "em cartório e sobre a validade da notificação/intimação "
            "(pessoal ou por edital). Informe: se houve tentativa de "
            "notificação em endereço incorreto, se as vias de localização "
            "pessoal foram esgotadas antes do edital, e o trecho de "
            "referência."
        ),
    },
    ClauseType.MULTA_MORA_TRIBUTARIA.value: {
        "label": "Multa de mora tributária",
        "search_instruction": (
            "Extraia todas as passagens sobre multa de mora aplicada pelo "
            "Fisco (ex.: art. 61 da Lei 9.430/96), incluindo alegações de "
            "cobrança indevida antes do vencimento. Informe: percentual "
            "da multa, o fato gerador alegado, o argumento de ilegalidade, "
            "e o trecho de referência."
        ),
    },
    ClauseType.BASE_CALCULO_TRIBUTO.value: {
        "label": "Base de cálculo de tributo (exclusões)",
        "search_instruction": (
            "Extraia todas as passagens que discutem o que deve ou não "
            "compor a base de cálculo de um tributo ou contribuição (ex.: "
            "exclusão do ICMS, ISS ou do próprio PIS/COFINS da base de "
            "cálculo do PIS/COFINS). Informe: o tributo discutido, o "
            "valor/parcela que se pede para excluir, o precedente citado "
            "(ex.: RE 574.706/PR - Tema 69), e o trecho de referência."
        ),
    },
    ClauseType.COMPENSACAO_RESTITUICAO_INDEBITO.value: {
        "label": "Compensação / restituição de indébito",
        "search_instruction": (
            "Extraia todas as passagens sobre pedido de restituição ou "
            "compensação de tributo pago indevidamente. Informe: o valor "
            "ou período do indébito, a forma pedida (restituição em "
            "precatório/RPV ou compensação administrativa), o índice de "
            "atualização, e o trecho de referência."
        ),
    },
    ClauseType.ACAO_COLETIVA_SUBSTITUICAO.value: {
        "label": "Ação coletiva / substituição processual",
        "search_instruction": (
            "Extraia todas as passagens que tratam da ação coletiva de "
            "origem e da qualidade de substituído processual do autor. "
            "Informe: número da ação coletiva, entidade/sindicato autor, "
            "o direito reconhecido na sentença coletiva, e o trecho de "
            "referência."
        ),
    },
    ClauseType.OBRIGACAO_ACESSORIA_FISCAL.value: {
        "label": "Obrigação acessória fiscal (eSocial/DCTFWeb)",
        "search_instruction": (
            "Extraia todas as passagens sobre obrigações acessórias "
            "fiscais eletrônicas (eSocial, DCTFWeb, DARF) e eventuais "
            "falhas sistêmicas alegadas. Informe: qual evento/declaração "
            "está em discussão, o prazo legal aplicável, o problema "
            "relatado (ex.: geração automática de multa indevida), e o "
            "trecho de referência."
        ),
    },
    ClauseType.DIVIDA_ATIVA_CDA.value: {
        "label": "Dívida ativa / CDA / PGFN",
        "search_instruction": (
            "Extraia todas as passagens sobre inscrição em dívida ativa e "
            "Certidões de Dívida Ativa (CDA) discutidas na ação. Informe: "
            "número da inscrição/processo administrativo, valor, órgão "
            "(PGFN/RFB), motivo alegado de irregularidade, e o trecho de "
            "referência."
        ),
    },
    ClauseType.PEDIDOS_PROCESSUAIS.value: {
        "label": "Pedidos (requerimentos finais da petição)",
        "search_instruction": (
            "Extraia todos os pedidos formulados na petição (itens do "
            "capítulo 'Dos Pedidos'). Para cada um, informe um resumo "
            "objetivo do que é pedido e a letra/alínea correspondente no "
            "documento."
        ),
    },
    ClauseType.FUNDAMENTACAO_CONSTITUCIONAL.value: {
        "label": "Fundamentação constitucional/legal citada",
        "search_instruction": (
            "Extraia todos os dispositivos constitucionais, legais ou "
            "precedentes de tribunais superiores (STF/STJ/TNU/TST) "
            "citados como fundamento jurídico central da tese. Informe: o "
            "dispositivo ou precedente (ex.: art. 195, I, CF; RE "
            "574.706/PR; Súmula 345/STJ), o que ele estabelece, e o "
            "trecho de referência."
        ),
    },

    # ── Direito Civil material / termos jurídicos comuns ────────────────
    ClauseType.RESPONSABILIDADE_CIVIL_EXTRACONTRATUAL.value: {
        "label": "Responsabilidade civil extracontratual (ato ilícito)",
        "search_instruction": (
            "Extraia todas as passagens que tratam de responsabilidade "
            "civil extracontratual (aquiliana). Para cada uma, informe "
            "nos campos de `dados`: {\"ato_ilícito\": a conduta imputada "
            "(ação ou omissão), \"fundamento_legal\": dispositivo citado "
            "(ex.: art. 186, 187 ou 927 do Código Civil), "
            "\"responsabilidade_subjetiva_ou_objetiva\": se depende de "
            "culpa/dolo ou é objetiva (ex.: art. 933, 927, §único, CC), "
            "\"excludente_alegada\": culpa exclusiva da vítima, fato de "
            "terceiro, caso fortuito ou força maior, se houver}. Informe "
            "também o trecho de referência."
        ),
    },
    ClauseType.DANO_MORAL.value: {
        "label": "Dano moral",
        "search_instruction": (
            "Extraia todas as passagens sobre dano moral. Em `dados`, "
            "informe: {\"fato_gerador\": o que causou o abalo moral, "
            "\"fundamento_legal\": dispositivo citado (ex.: art. 5º, V e "
            "X, CF; art. 186 e 927 do CC), \"valor_pretendido\": quantia "
            "pedida a título de indenização, se houver, "
            "\"cumulacao_dano_estetico\": se há pedido cumulado de dano "
            "estético (Súmula 387/STJ), \"criterio_fixacao\": critério "
            "de arbitramento citado (proporcionalidade, razoabilidade, "
            "caráter pedagógico/punitivo)}. Inclua o trecho de "
            "referência."
        ),
    },
    ClauseType.DANO_MATERIAL_LUCROS_CESSANTES.value: {
        "label": "Dano material / lucros cessantes",
        "search_instruction": (
            "Extraia todas as passagens sobre dano material (dano "
            "emergente) e lucros cessantes. Em `dados`, informe: "
            "{\"tipo\": \"dano_emergente\" ou \"lucros_cessantes\" ou "
            "\"ambos\", \"fundamento_legal\": dispositivo citado (ex.: "
            "art. 402 e 403 do CC), \"valor_ou_criterio_calculo\": valor "
            "pedido ou método de cálculo usado, \"comprovacao\": meio de "
            "prova indicado (notas fiscais, perícia contábil, etc.)}. "
            "Inclua o trecho de referência."
        ),
    },
    ClauseType.NEXO_CAUSALIDADE_CULPA.value: {
        "label": "Nexo de causalidade / culpa",
        "search_instruction": (
            "Extraia as passagens que discutem os elementos da "
            "responsabilidade civil (conduta, culpa/dolo, dano, nexo "
            "causal — art. 186 CC). Em `dados`, informe: {\"conduta\": "
            "ação ou omissão apontada, \"elemento_subjetivo\": culpa ou "
            "dolo alegado (ou dispensa por responsabilidade objetiva), "
            "\"nexo_causal_argumento\": como se liga a conduta ao dano, "
            "\"teoria_causalidade\": se mencionada (dano direto e "
            "imediato, causalidade adequada, etc.)}. Inclua o trecho de "
            "referência."
        ),
    },
    ClauseType.ENRIQUECIMENTO_SEM_CAUSA.value: {
        "label": "Enriquecimento sem causa",
        "search_instruction": (
            "Extraia as passagens sobre enriquecimento sem causa. Em "
            "`dados`, informe: {\"fundamento_legal\": dispositivo citado "
            "(art. 884 a 886 do CC), \"quem_enriqueceu\": parte "
            "apontada como beneficiada indevidamente, "
            "\"quem_empobreceu\": parte que sofreu o prejuízo "
            "correspondente, \"valor_restituivel\": valor pedido a "
            "título de restituição}. Inclua o trecho de referência."
        ),
    },
    ClauseType.PRESCRICAO_CIVIL_CC.value: {
        "label": "Prescrição civil (Código Civil)",
        "search_instruction": (
            "Extraia as passagens sobre prazos prescricionais civis "
            "(fora da esfera tributária). Em `dados`, informe: "
            "{\"prazo_aplicado\": prazo em anos (ex.: 10 anos - regra "
            "geral do art. 205 CC; ou prazo especial do art. 206 CC), "
            "\"fundamento_legal\": dispositivo citado, \"termo_inicial\": "
            "marco de início da contagem (art. 189 CC - nascimento da "
            "pretensão), \"causa_suspensao_interrupcao\": se há "
            "alegação de suspensão/interrupção do prazo (arts. 197 a "
            "204 CC)}. Inclua o trecho de referência."
        ),
    },
    ClauseType.CLAUSULA_PENAL_CIVIL.value: {
        "label": "Cláusula penal (Código Civil)",
        "search_instruction": (
            "Extraia todas as cláusulas penais (multa contratual civil, "
            "arts. 408 a 416 do CC). Em `dados`, informe: "
            "{\"tipo\": \"compensatória\" ou \"moratória\", "
            "\"valor_ou_percentual\": valor/percentual fixado, "
            "\"limite_legal\": se respeita o limite do art. 412 CC (não "
            "pode exceder o valor da obrigação principal), "
            "\"evento_gerador\": inadimplemento total, parcial ou mora}. "
            "Inclua a cláusula/trecho de referência."
        ),
    },
    ClauseType.VICIO_REDIBITORIO_EVICCAO.value: {
        "label": "Vícios redibitórios / evicção",
        "search_instruction": (
            "Extraia as passagens sobre vícios redibitórios (arts. 441 a "
            "446 do CC) ou evicção (arts. 447 a 457 do CC). Em `dados`, "
            "informe: {\"instituto\": \"vicio_redibitorio\" ou "
            "\"eviccao\", \"fundamento_legal\": dispositivo citado, "
            "\"defeito_ou_perda_alegada\": defeito oculto do bem ou "
            "perda da coisa por sentença/ato administrativo, "
            "\"remedio_pedido\": redibição (devolução), abatimento no "
            "preço, ou indenização}. Inclua o trecho de referência."
        ),
    },
    ClauseType.OBRIGACAO_FAZER_NAO_FAZER.value: {
        "label": "Obrigação de fazer / não fazer",
        "search_instruction": (
            "Extraia as passagens sobre obrigações de fazer ou não "
            "fazer. Em `dados`, informe: {\"tipo\": \"fazer\" ou "
            "\"nao_fazer\", \"conduta_exigida\": o que deve ou não ser "
            "feito, \"fundamento_legal\": dispositivo citado (ex.: arts. "
            "247 a 251 do CC; art. 497 e 536/537 do CPC para tutela "
            "específica e multa/astreintes), \"multa_cominatoria\": "
            "valor/periodicidade da multa pedida em caso de "
            "descumprimento, se houver}. Inclua o trecho de referência."
        ),
    },
    ClauseType.POSSE_PROPRIEDADE_USUCAPIAO.value: {
        "label": "Posse / propriedade / usucapião",
        "search_instruction": (
            "Extraia as passagens sobre posse, propriedade ou usucapião. "
            "Em `dados`, informe: {\"instituto\": \"posse\", "
            "\"propriedade\" ou \"usucapiao\", \"fundamento_legal\": "
            "dispositivo citado (ex.: arts. 1.196 a 1.224 do CC para "
            "posse; arts. 1.238 a 1.244 do CC para usucapião — "
            "extraordinária/ordinária/especial), \"tempo_de_posse\": "
            "período alegado e se contínuo/mansa e pacífica, "
            "\"tipo_usucapiao\": modalidade invocada, se aplicável}. "
            "Inclua o trecho de referência."
        ),
    },
    ClauseType.CONTRATO_CIVIL_TIPICO.value: {
        "label": "Contrato civil típico (compra e venda, locação, comodato, mútuo, doação)",
        "search_instruction": (
            "Extraia as passagens que caracterizam o tipo contratual "
            "civil em discussão. Em `dados`, informe: {\"tipo_contrato\": "
            "\"compra_e_venda\" (arts. 481-532 CC), \"locacao\" (arts. "
            "565-578 CC / Lei 8.245/91), \"comodato\" (arts. 579-585 "
            "CC), \"mutuo\" (arts. 586-592 CC), \"doacao\" (arts. "
            "538-564 CC) ou \"prestacao_de_servicos\" (arts. 593-609 "
            "CC), \"objeto\": o que é vendido/locado/emprestado/doado, "
            "\"partes\": quem figura como cada parte (ex.: locador/"
            "locatário, comodante/comodatário)}. Inclua o trecho de "
            "referência."
        ),
    },
    ClauseType.DIREITO_FAMILIA_ALIMENTOS.value: {
        "label": "Direito de família / alimentos",
        "search_instruction": (
            "Extraia as passagens sobre pensão alimentícia ou outros "
            "temas de direito de família. Em `dados`, informe: "
            "{\"fundamento_legal\": dispositivo citado (ex.: art. 1.694 "
            "a 1.710 do CC; Lei 5.478/68 - Lei de Alimentos), "
            "\"criterio_binomio\": referência ao binômio/trinômio "
            "necessidade-possibilidade-proporcionalidade (art. 1.694, "
            "§1º, CC), \"valor_ou_percentual_pedido\": valor ou "
            "percentual de alimentos pedido, \"beneficiario\": quem "
            "pede os alimentos, \"obrigado\": de quem se pede}. Inclua o "
            "trecho de referência."
        ),
    },
    ClauseType.DIREITO_SUCESSOES.value: {
        "label": "Direito das sucessões (herança, testamento, inventário)",
        "search_instruction": (
            "Extraia as passagens sobre sucessão hereditária, testamento "
            "ou inventário. Em `dados`, informe: {\"fundamento_legal\": "
            "dispositivo citado (ex.: art. 1.784 CC - princípio da "
            "saisine; art. 1.829 CC - ordem de vocação hereditária; art. "
            "1.846 CC - legítima; arts. 1.857 e ss. CC - testamento; "
            "arts. 610 a 673 do CPC - inventário e partilha), "
            "\"instituto\": \"inventario\", \"testamento\", \"legitima\" "
            "ou \"vocacao_hereditaria\", \"herdeiros_envolvidos\": quem "
            "figura na disputa/partilha, \"bens_ou_valor\": bens ou "
            "valor da herança em discussão}. Inclua o trecho de "
            "referência."
        ),
    },

    # ── Requisitos processuais da petição inicial (CPC) ─────────────────
    ClauseType.REQUISITOS_PETICAO_INICIAL_ART319.value: {
        "label": "Checklist da petição inicial (art. 319 do CPC)",
        "search_instruction": _build_peticao_inicial_art319_instruction(),
    },

    ClauseType.OTHER.value: {
        "label": "Outro (instrução livre)",
        "search_instruction": (
            "Descreva livremente o que deve ser extraído do documento."
        ),
    },
}


def list_clause_type_options() -> List[Dict[str, str]]:
    """
    Retorna a lista de tipos de cláusula disponíveis, no formato adequado
    para popular um <select>/dropdown no front-end:
        [{"value": "multa", "label": "Multa / penalidade contratual"}, ...]

    A descrição completa (search_instruction) NÃO é enviada aqui para não
    poluir o dropdown — ela é resolvida no backend via
    build_instruction_for_type() quando o usuário efetivamente escolhe um
    tipo e dispara a extração.
    """
    return [
        {"value": value, "label": info["label"]}
        for value, info in CLAUSE_TYPE_CATALOG.items()
    ]


def build_instruction_for_type(clause_type: str, extra: Optional[str] = None) -> str:
    """
    Resolve a instrução de extração a partir do tipo de cláusula escolhido
    pelo usuário no seletor do prompt.

    Args:
        clause_type: valor do ClauseType escolhido (ex.: "multa").
        extra: texto adicional opcional digitado pelo usuário para refinar
               o pedido (ex.: "apenas cláusulas acima de R$ 10.000,00").

    Returns:
        A instrução final, pronta para ser passada ao CustomExtractor.

    Raises:
        ValueError: se o tipo não existir no catálogo.
    """
    info = CLAUSE_TYPE_CATALOG.get(clause_type)
    if info is None:
        raise ValueError(f"Tipo de cláusula desconhecido: {clause_type!r}")

    instruction = info["search_instruction"]
    if extra and extra.strip():
        instruction += f"\n\nObservação adicional do usuário: {extra.strip()}"
    return instruction


class ApprovedClause(Base):
    """
    Cláusula aprovada que faz parte da biblioteca de precedentes.

    Cada registro representa uma cláusula que foi extraída de um contrato
    e aprovada por um advogado como boa referência futura.
    """
    __tablename__ = "approved_clauses"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Metadados da cláusula
    clause_type = Column(String(50), nullable=False, index=True)  # ClauseType value
    title = Column(String(200))  # Título descritivo da cláusula

    # Conteúdo
    original_text = Column(Text, nullable=False)  # Texto original da cláusula
    standardized_text = Column(Text)  # Versão padronizada (se aprovada com mods)
    summary = Column(Text, nullable=False)  # Resumo do que a cláusula faz

    # Contexto
    job_id = Column(String(64), nullable=False, index=True)  # Job original
    doc_hash = Column(String(64), nullable=False, index=True)  # Documento original
    source_file = Column(Text)  # Nome do arquivo original

    # Tags e metadados adicionais
    tags = Column(JSON)  # Lista de tags personalizadas
    extra_metadata = Column(JSON)  # Metadados adicionais (setor, valor, etc.)

    # Aprovação
    approval_status = Column(String(20), default=ApprovalStatus.PENDING.value, index=True)
    approved_by = Column(String(100))  # Nome/email do aprovador
    approved_at = Column(DateTime)  # Data de aprovação
    notes = Column(Text)  # Notas do aprovador

    # Métricas de uso
    times_used = Column(Integer, default=0)  # Quantas vezes foi usada como referência
    last_used_at = Column(DateTime)  # Última vez que foi consultada

    # Rastreabilidade
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "clause_type": self.clause_type,
            "clause_type_label": CLAUSE_TYPE_CATALOG.get(self.clause_type, {}).get(
                "label", self.clause_type
            ),
            "title": self.title or f"Cláusula {self.clause_type}",
            "original_text": self.original_text,
            "standardized_text": self.standardized_text,
            "summary": self.summary,
            "job_id": self.job_id,
            "source_file": self.source_file,
            "tags": self.tags or [],
            "extra_metadata": self.extra_metadata or {},
            "approval_status": self.approval_status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "notes": self.notes,
            "times_used": self.times_used,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ClauseLibrary:
    """
    Gerenciador da biblioteca de cláusulas aprovadas.

    Funcionalidades:
    - Adicionar cláusulas extraídas à biblioteca
    - Aprovar/rejeitar cláusulas
    - Buscar cláusulas similares por tipo
    - Buscar por TF-IDF (texto livre)
    - Gerar relatórios de uso
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Args:
            db_url: URL do banco PostgreSQL. Se None, usa a mesma configuração do Store.
        """
        if db_url is None:
            import os
            db_url = os.environ.get(
                "DATABASE_URL",
                "postgresql://extractor:extractor123@localhost:5432/extractor"
            )

        self._engine = create_engine(db_url, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)
        self._create_tables()

        logger.info("[ClauseLibrary] Inicializado com banco de dados.")

    def _create_tables(self):
        """Cria tabelas se não existirem."""
        Base.metadata.create_all(self._engine)

    # ─────────────────────────────────────────────────────────────────────
    # Adicionar cláusulas
    # ─────────────────────────────────────────────────────────────────────

    def add_from_extraction(
        self,
        job_id: str,
        doc_hash: str,
        source_file: str,
        items: List[Dict],
        auto_approve: bool = False,
        clause_type: Optional[str] = None,
    ) -> int:
        """
        Adiciona itens extraídos à biblioteca para aprovação.

        Args:
            job_id: ID do job de extração
            doc_hash: Hash do documento original
            source_file: Nome do arquivo
            items: Lista de itens extraídos (do job)
            auto_approve: Se True, aprova automaticamente (cuidado!)
            clause_type: Se informado (o usuário escolheu um tipo no
                         seletor do prompt), usa esse tipo diretamente em
                         vez de tentar detectar automaticamente por
                         palavra-chave.

        Returns:
            Número de cláusulas adicionadas
        """
        added = 0

        with self._Session() as session:
            for item in items:
                # Se o usuário já informou o tipo (via seletor), usa-o;
                # caso contrário, tenta detectar automaticamente.
                resolved_type = clause_type or self._detect_clause_type(item)

                clause = ApprovedClause(
                    clause_type=resolved_type,
                    title=item.get("resumo", "")[:200],
                    original_text=item.get("trecho_referencia", ""),
                    summary=item.get("resumo", ""),
                    job_id=job_id,
                    doc_hash=doc_hash,
                    source_file=source_file,
                    tags=self._extract_tags(item),
                    extra_metadata=item.get("dados", {}),
                    approval_status=ApprovalStatus.APPROVED.value if auto_approve else ApprovalStatus.PENDING.value,
                )

                session.add(clause)
                added += 1

            session.commit()

        logger.info("[ClauseLibrary] %d cláusulas adicionadas do job %s", added, job_id)
        return added

    # ─────────────────────────────────────────────────────────────────────
    # Aprovação
    # ─────────────────────────────────────────────────────────────────────

    def approve_clause(
        self,
        clause_id: int,
        approved_by: str,
        notes: Optional[str] = None,
        standardized_text: Optional[str] = None,
    ) -> bool:
        """
        Aprova uma cláusula para a biblioteca.

        Args:
            clause_id: ID da cláusula
            approved_by: Quem aprovou
            notes: Notas do aprovador
            standardized_text: Texto padronizado (se houve modificações)

        Returns:
            True se aprovou com sucesso
        """
        with self._Session() as session:
            clause = session.query(ApprovedClause).filter_by(id=clause_id).first()
            if not clause:
                return False

            clause.approval_status = ApprovalStatus.APPROVED.value
            clause.approved_by = approved_by
            clause.approved_at = datetime.utcnow()
            clause.notes = notes
            if standardized_text:
                clause.standardized_text = standardized_text

            session.commit()

        logger.info("[ClauseLibrary] Cláusula %d aprovada por %s", clause_id, approved_by)
        return True

    def reject_clause(self, clause_id: int, rejected_by: str, notes: Optional[str] = None) -> bool:
        """Rejeita uma cláusula."""
        with self._Session() as session:
            clause = session.query(ApprovedClause).filter_by(id=clause_id).first()
            if not clause:
                return False

            clause.approval_status = ApprovalStatus.REJECTED.value
            clause.notes = notes

            session.commit()

        logger.info("[ClauseLibrary] Cláusula %d rejeitada por %s", clause_id, rejected_by)
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Busca
    # ─────────────────────────────────────────────────────────────────────

    def search_by_type(self, clause_type: str, approved_only: bool = False) -> List[Dict]:
        """
        Busca cláusulas por tipo.

        Args:
            clause_type: Tipo da cláusula (ex: 'forca_maior', 'multa',
                         'tutela_liminar', 'base_calculo_tributo' etc. —
                         ver ClauseType/CLAUSE_TYPE_CATALOG). Se vazio,
                         retorna todos os tipos.
            approved_only: Se True, retorna apenas aprovadas. Padrão=False
                         para incluir pendentes.

        Returns:
            Lista de cláusulas
        """
        with self._Session() as session:
            # Se não tem tipo específico, busca todos (sem filter_by clause_type)
            if clause_type and clause_type.strip():
                query = session.query(ApprovedClause).filter_by(clause_type=clause_type)
            else:
                query = session.query(ApprovedClause)

            if approved_only:
                query = query.filter_by(approval_status=ApprovalStatus.APPROVED.value)

            results = query.order_by(ApprovedClause.times_used.desc()).all()

            # Atualiza métricas de uso
            for clause in results:
                clause.times_used += 1
                clause.last_used_at = datetime.utcnow()

            session.commit()

            return [clause.to_dict() for clause in results]

    def get_recent_clauses(self, limit: int = 5) -> List[Dict]:
        """
        Retorna as cláusulas mais recentemente extraídas.

        Args:
            limit: Quantidade de cláusulas a retornar (padrão: 5)

        Returns:
            Lista de cláusulas ordenadas por data de criação decrescente
        """
        with self._Session() as session:
            results = session.query(ApprovedClause)\
                .order_by(ApprovedClause.created_at.desc())\
                .limit(limit)\
                .all()

            return [clause.to_dict() for clause in results]

    def search_tfidf(self, query: str, clause_type: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """
        Busca cláusulas usando TF-IDF (busca semântica).

        Args:
            query: Texto da busca (ex: "como já redigimos cláusula de força maior")
            clause_type: Filtrar por tipo (opcional)
            top_k: Quantidade de resultados

        Returns:
            Lista de cláusulas ordenadas por relevância
        """
        from tfidf_utils import tfidf_rank

        with self._Session() as session:
            # Busca cláusulas aprovadas
            query_db = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.APPROVED.value
            )

            if clause_type:
                query_db = query_db.filter_by(clause_type=clause_type)

            clauses = query_db.all()

            if not clauses:
                return []

            # Prepara textos para busca (mantendo o índice alinhado com
            # `clauses`, para não depender de comparação de string na hora
            # de religar o score à cláusula original — ver nota abaixo).
            texts = []
            for clause in clauses:
                text = f"{clause.title or ''} {clause.summary or ''}"
                if clause.tags:
                    text += " " + " ".join(clause.tags)
                texts.append(text)

            # TF-IDF: tfidf_rank retorna pares (score, texto). Em vez de
            # religar o score à cláusula original comparando strings (o
            # que falha se duas cláusulas tiverem texto idêntico), usamos
            # um mapa texto -> índices e consumimos cada índice uma única
            # vez, na ordem de score retornada.
            scored = tfidf_rank(query, texts, top_k=len(texts))  # [(score, text), ...]
            text_to_indices: Dict[str, List[int]] = {}
            for idx, text in enumerate(texts):
                text_to_indices.setdefault(text, []).append(idx)

            results = []
            used_indices = set()
            for score, text in scored[:top_k]:
                candidates = [i for i in text_to_indices.get(text, []) if i not in used_indices]
                if not candidates:
                    continue
                idx = candidates[0]
                used_indices.add(idx)

                clause = clauses[idx]
                clause.times_used += 1
                clause.last_used_at = datetime.utcnow()
                results.append({**clause.to_dict(), "relevance_score": float(score)})

            session.commit()
            return results

    def search_similar(self, clause_id: int, top_k: int = 5) -> List[Dict]:
        """
        Busca cláusulas similares a uma específica.

        Útil para: "Quais outras formas já redigimos esta cláusula?"
        """
        with self._Session() as session:
            reference = session.query(ApprovedClause).filter_by(id=clause_id).first()
            if not reference:
                return []

            # Busca usando o resumo como query
            return self.search_tfidf(
                reference.summary,
                clause_type=reference.clause_type,
                top_k=top_k + 1  # +1 para excluir a própria
            )[:top_k]

    # ─────────────────────────────────────────────────────────────────────
    # Relatórios
    # ─────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Estatísticas da biblioteca."""
        with self._Session() as session:
            total = session.query(ApprovedClause).count()
            approved = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.APPROVED.value
            ).count()
            pending = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.PENDING.value
            ).count()

            # Por tipo — usa o LABEL amigável do catálogo (ex.: "Pedidos
            # (requerimentos finais da petição)"), não o value bruto do
            # enum (ex.: "pedidos_processuais"), que é o que aparecia nos
            # cards de estatística da biblioteca.
            by_type = {}
            for type_enum in ClauseType:
                count = session.query(ApprovedClause).filter_by(
                    clause_type=type_enum.value,
                    approval_status=ApprovalStatus.APPROVED.value
                ).count()
                if count > 0:
                    label = CLAUSE_TYPE_CATALOG.get(type_enum.value, {}).get(
                        "label", type_enum.value
                    )
                    by_type[label] = count

            # Mais usadas
            most_used = session.query(ApprovedClause).filter_by(
                approval_status=ApprovalStatus.APPROVED.value
            ).order_by(ApprovedClause.times_used.desc()).limit(5).all()

            return {
                "total_clauses": total,
                "approved": approved,
                "pending": pending,
                "by_type": by_type,
                "most_used": [c.to_dict() for c in most_used],
            }

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _detect_clause_type(self, item: Dict) -> str:
        """
        Detecta automaticamente o tipo de cláusula/tópico por
        palavra-chave. Usado apenas quando o usuário NÃO escolheu
        explicitamente um tipo no seletor do prompt (add_from_extraction
        com clause_type=None).
        """
        resumo = item.get("resumo", "").lower()

        # Palavras-chave para cada tipo. Ordem importa: tipos mais
        # específicos vêm antes de tipos mais genéricos que poderiam
        # capturar as mesmas palavras (ex.: "multa" tributária antes de
        # "multa" contratual genérica).
        keywords = {
            # Contencioso / petições (mais específico primeiro)
            ClauseType.MULTA_MORA_TRIBUTARIA.value: [
                "multa de mora", "art. 61", "darf", "dctfweb", "esocial",
            ],
            ClauseType.REGULARIDADE_FISCAL_CND.value: [
                "cnd", "cpen", "certidão negativa", "regularidade fiscal",
            ],
            ClauseType.PROTESTO_NOTIFICACAO_EDITAL.value: [
                "protesto", "cartório", "edital", "notificação",
                "intimação",
            ],
            ClauseType.DIVIDA_ATIVA_CDA.value: [
                "dívida ativa", "cda", "pgfn", "inscrição em dívida",
            ],
            ClauseType.BASE_CALCULO_TRIBUTO.value: [
                "base de cálculo", "pis/cofins", "pis e cofins", "icms",
                "faturamento", "receita bruta",
            ],
            ClauseType.COMPENSACAO_RESTITUICAO_INDEBITO.value: [
                "restituição", "compensação", "indébito", "repetição de indébito",
            ],
            ClauseType.ACAO_COLETIVA_SUBSTITUICAO.value: [
                "ação coletiva", "substituído processual", "substituídos",
            ],
            ClauseType.OBRIGACAO_ACESSORIA_FISCAL.value: [
                "obrigação acessória", "dctfweb", "esocial",
            ],
            ClauseType.TUTELA_LIMINAR.value: [
                "liminar", "fumus boni iuris", "periculum in mora",
                "tutela de urgência", "tutela de evidência",
            ],
            ClauseType.HONORARIOS_ADVOCATICIOS.value: [
                "honorários advocatícios", "verba honorária", "súmula 345",
            ],
            ClauseType.CORRECAO_MONETARIA_JUROS.value: [
                "correção monetária", "taxa selic", "juros de mora",
            ],
            ClauseType.PRESCRICAO_DECADENCIA.value: [
                "prescrição quinquenal", "prescrição", "decadência",
            ],
            ClauseType.FUNDAMENTACAO_CONSTITUCIONAL.value: [
                "inconstitucional", "inconstitucionalidade", "repercussão geral",
            ],
            ClauseType.PEDIDOS_PROCESSUAIS.value: [
                "requer-se", "requer o exequente", "dos pedidos",
            ],
            # Direito Civil material
            ClauseType.RESPONSABILIDADE_CIVIL_EXTRACONTRATUAL.value: [
                "ato ilícito", "responsabilidade extracontratual", "aquiliana",
                "art. 186", "art. 927",
            ],
            ClauseType.DANO_MORAL.value: [
                "dano moral", "abalo moral", "dano extrapatrimonial",
            ],
            ClauseType.DANO_MATERIAL_LUCROS_CESSANTES.value: [
                "dano material", "lucros cessantes", "dano emergente",
            ],
            ClauseType.NEXO_CAUSALIDADE_CULPA.value: [
                "nexo causal", "nexo de causalidade", "culpa exclusiva",
            ],
            ClauseType.ENRIQUECIMENTO_SEM_CAUSA.value: [
                "enriquecimento sem causa", "enriquecimento ilícito",
            ],
            ClauseType.PRESCRICAO_CIVIL_CC.value: [
                "art. 205", "art. 206", "prescrição decenal", "pretensão",
            ],
            ClauseType.CLAUSULA_PENAL_CIVIL.value: [
                "cláusula penal", "multa compensatória", "multa moratória",
            ],
            ClauseType.VICIO_REDIBITORIO_EVICCAO.value: [
                "vício redibitório", "vícios redibitórios", "evicção",
            ],
            ClauseType.OBRIGACAO_FAZER_NAO_FAZER.value: [
                "obrigação de fazer", "obrigação de não fazer", "astreintes",
                "multa cominatória",
            ],
            ClauseType.POSSE_PROPRIEDADE_USUCAPIAO.value: [
                "usucapião", "posse mansa e pacífica", "ad usucapionem",
            ],
            ClauseType.CONTRATO_CIVIL_TIPICO.value: [
                "compra e venda", "comodato", "mútuo", "doação", "locação",
            ],
            ClauseType.DIREITO_FAMILIA_ALIMENTOS.value: [
                "pensão alimentícia", "alimentos", "binômio necessidade",
            ],
            ClauseType.DIREITO_SUCESSOES.value: [
                "inventário", "herança", "testamento", "vocação hereditária",
                "partilha",
            ],
            # Requisitos processuais
            ClauseType.REQUISITOS_PETICAO_INICIAL_ART319.value: [
                "art. 319", "petição inicial", "requisitos da petição",
            ],
            # Contratuais
            ClauseType.FORCE_MAJEURE.value: [
                "força maior", "caso fortuito", "evento imprevisto",
            ],
            ClauseType.TERMINATION.value: [
                "rescisão", "término", "encerramento", "cancelamento",
            ],
            ClauseType.PENALTY.value: [
                "multa", "penalidade", "sanção", "mora",
            ],
            ClauseType.PRICE_ADJUSTMENT.value: [
                "reajuste", "correção", "indexador", "inflação",
            ],
            ClauseType.CONFIDENTIALITY.value: [
                "confidencial", "sigilo", "informação confidencial",
            ],
            ClauseType.LIABILITY.value: [
                "responsabilidade", "limitação", "danos",
            ],
            ClauseType.INTELLECTUAL_PROPERTY.value: [
                "propriedade intelectual", "direitos autorais",
            ],
            ClauseType.DISPUTE_RESOLUTION.value: [
                "disputa", "controvérsia", "arbitragem", "judiciário",
            ],
            ClauseType.PAYMENT.value: [
                "pagamento", "parcela", "mensalidade", "fatura",
            ],
            ClauseType.WARRANTY.value: [
                "garantia",
            ],
            ClauseType.INDEMNITY.value: [
                "indenização", "indenizar",
            ],
            ClauseType.CHANGE_OF_CONTROL.value: [
                "mudança de controle", "controle societário",
            ],
            ClauseType.ASSIGNMENT.value: [
                "cessão", "cedido", "cessionário",
            ],
            ClauseType.NON_COMPETE.value: [
                "não concorrência", "não concorrer",
            ],
            ClauseType.SOLICITATION.value: [
                "não aliciamento", "aliciar",
            ],
        }

        # Busca por palavras-chave, na ordem definida acima
        for type_value, keys in keywords.items():
            if any(key in resumo for key in keys):
                return type_value

        # Padrão
        return ClauseType.OTHER.value

    def _extract_tags(self, item: Dict) -> List[str]:
        """Extrai tags do item extraído."""
        tags = []

        # Tira do resumo
        resumo = item.get("resumo", "")
        if "%" in resumo:
            tags.append("percentual")
        if "R$" in resumo or "reais" in resumo.lower():
            tags.append("monetario")
        if "diário" in resumo.lower() or "dia" in resumo.lower():
            tags.append("prazo_diario")
        if "mensal" in resumo.lower():
            tags.append("prazo_mensal")

        return tags


# ─────────────────────────────────────────────────────────────────────────
# API REST endpoints (para integrar com app.py)
# ─────────────────────────────────────────────────────────────────────────

def setup_clause_routes(app):
    """Configura rotas Flask para a biblioteca de cláusulas."""

    @app.route("/api/clause-types", methods=["GET"])
    def list_clause_types():
        """
        Lista os tipos de cláusula disponíveis para popular o seletor do
        prompt de extração (dropdown no front-end). Cada item tem
        {"value", "label"} — a descrição completa fica só no backend e é
        resolvida em /api/clause-types/<value>/instruction.
        """
        return {"results": list_clause_type_options()}

    @app.route("/api/clause-types/<clause_type>/instruction", methods=["GET"])
    def get_clause_type_instruction(clause_type):
        """
        Retorna a instrução de extração completa associada a um tipo de
        cláusula. O front-end chama isso quando o usuário seleciona um
        tipo no dropdown, para mostrar (ou já enviar) a instrução que
        será usada na extração — sem o usuário precisar escrevê-la à mão.
        """
        from flask import request
        extra = request.args.get("extra")
        try:
            instruction = build_instruction_for_type(clause_type, extra=extra)
        except ValueError as e:
            return {"error": str(e)}, 404
        return {
            "clause_type": clause_type,
            "label": CLAUSE_TYPE_CATALOG[clause_type]["label"],
            "instruction": instruction,
        }

    @app.route("/api/clauses", methods=["GET"])
    def list_clauses():
        """Lista cláusulas com filtros."""
        from flask import request

        clause_type = request.args.get("type")
        status = request.args.get("status", "approved")  # Padrão: aprovadas

        library = ClauseLibrary()

        # Converte status para approved_only
        if status == "approved":
            approved_only = True
        elif status == "pending":
            approved_only = False
        else:
            # "all" ou qualquer outro valor
            approved_only = False

        if clause_type:
            results = library.search_by_type(clause_type, approved_only)
        else:
            results = library.search_by_type("", approved_only)

        # Se status=approved_only, filtra novamente para garantir
        if status == "approved":
            results = [r for r in results if r.get("approval_status") == "approved"]
        elif status == "pending":
            results = [r for r in results if r.get("approval_status") == "pending"]

        return {"results": results, "total": len(results)}

    @app.route("/api/clauses/recent", methods=["GET"])
    def list_recent_clauses():
        """Lista as cláusulas mais recentes (para carregar ao abrir a tela)."""
        from flask import request

        limit = int(request.args.get("limit", 5))
        status = request.args.get("status", "approved")  # Padrão: aprovadas

        library = ClauseLibrary()
        all_results = library.get_recent_clauses(limit=limit * 2)  # Pega mais para filtrar

        # Filtra por status
        if status == "approved":
            results = [r for r in all_results if r.get("approval_status") == "approved"][:limit]
        elif status == "pending":
            results = [r for r in all_results if r.get("approval_status") == "pending"][:limit]
        else:
            results = all_results[:limit]

        return {"results": results, "total": len(results)}

    @app.route("/api/clauses/search", methods=["POST"])
    def search_clauses():
        """Busca cláusulas por TF-IDF."""
        from flask import request

        data = request.json
        query = data.get("query", "")
        clause_type = data.get("type")
        top_k = data.get("top_k", 10)

        if not query:
            return {"error": "Query é obrigatória"}, 400

        library = ClauseLibrary()
        results = library.search_tfidf(query, clause_type, top_k)

        return {"results": results, "total": len(results)}

    @app.route("/api/clauses/<int:clause_id>/approve", methods=["POST"])
    def approve_clause(clause_id):
        """Aprova uma cláusula."""
        from flask import request

        data = request.json
        approved_by = data.get("approved_by")
        notes = data.get("notes")
        standardized_text = data.get("standardized_text")

        if not approved_by:
            return {"error": "approved_by é obrigatório"}, 400

        library = ClauseLibrary()
        success = library.approve_clause(clause_id, approved_by, notes, standardized_text)

        if not success:
            return {"error": "Cláusula não encontrada"}, 404

        return {"ok": True}

    @app.route("/api/clauses/<int:clause_id>/reject", methods=["POST"])
    def reject_clause(clause_id):
        """Rejeita uma cláusula."""
        from flask import request

        data = request.json
        rejected_by = data.get("rejected_by")
        notes = data.get("notes")

        if not rejected_by:
            return {"error": "rejected_by é obrigatório"}, 400

        library = ClauseLibrary()
        success = library.reject_clause(clause_id, rejected_by, notes)

        if not success:
            return {"error": "Cláusula não encontrada"}, 404

        return {"ok": True}

    @app.route("/api/clauses/stats", methods=["GET"])
    def clause_stats():
        """Estatísticas da biblioteca."""
        library = ClauseLibrary()
        return library.get_stats()

    @app.route("/api/jobs/<job_id>/add-to-library", methods=["POST"])
    def add_job_to_library(job_id):
        """Adiciona itens de um job à biblioteca."""
        from flask import request
        from store import Store

        data = request.json
        auto_approve = data.get("auto_approve", False)
        clause_type = data.get("clause_type")  # opcional: tipo já escolhido no prompt

        store = Store()
        job = store.get_job(job_id)

        if not job:
            return {"error": "Job não encontrado"}, 404

        library = ClauseLibrary()
        added = library.add_from_extraction(
            job_id=job_id,
            doc_hash="",  # TODO: pegar do job
            source_file=job.get("source_file", ""),
            items=job.get("items", []),
            auto_approve=auto_approve,
            clause_type=clause_type,
        )

        return {"ok": True, "added": added}