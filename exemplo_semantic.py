"""
exemplo_semantic.py
Exemplo de uso de embeddings semânticos locais no projeto.
"""

from semantic_embedding import create_semantic_embedding


def exemplo_basic():
    """Exemplo básico de uso."""
    print("\n=== Exemplo Básico ===\n")

    # Criar embedding semântico
    embedder = create_semantic_embedding(
        method="sentence-transformers",
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Textos para testar
    textos = [
        "O inquilino deve pagar aluguel mensalmente",
        "Locatário obrigado a pagar mensalidade",
        "O carro é vermelho e rápido",
        "Penalidade por atraso no pagamento",
        "Multa de 10% por inadimplemento",
    ]

    # Gerar embeddings
    embeddings = embedder.embed(textos)

    print(f"Textos processados: {len(textos)}")
    print(f"Dimensionalidade do embedding: {len(embeddings[0])}")

    # Calcular similaridades
    print("\nSimilaridades:")
    for i in range(len(textos)):
        for j in range(i + 1, len(textos)):
            sim = embedder.similarity(textos[i], textos[j])
            print(f"  {i}↔{j}: {sim:.3f}")


def exemplo_hibrido():
    """Exemplo com método híbrido (mais leve)."""
    print("\n=== Exemplo Híbrido ===\n")

    embedder = create_semantic_embedding(method="hybrid")

    textos = [
        "Cláusula de multa por rescisão antecipada",
        "Penalidade em caso de término antes do prazo",
        "O gato está no telhado",
    ]

    embeddings = embedder.embed(textos)

    print(f"Textos processados: {len(textos)}")
    print(f"Dimensionalidade do embedding: {len(embeddings[0])}")


def exemplo_busca_semantica():
    """Exemplo de busca semântica."""
    print("\n=== Busca Semântica ===\n")

    embedder = create_semantic_embedding(
        method="sentence-transformers",
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Base de conhecimento (itens extraídos de documentos)
    itens = [
        "Multa de 10% por rescisão antecipada do contrato",
        "Penalidade de mora de 2% ao mês em caso de atraso",
        "Juros de 1% ao mês sobre saldo devedor",
        "Cláusula de reajuste anual pelo IGP-M",
        "Seguro obrigatório contra incêndio",
    ]

    # Query do usuário
    query = "sanções por não pagar"

    # Embeddings
    query_emb = np.array(embedder.embed([query])[0])
    itens_emb = np.array(embedder.embed(itens))

    # Similaridade de cosseno
    similarities = np.dot(itens_emb, query_emb)

    # Ordenar por relevância
    ranking = sorted(zip(similarities, itens), reverse=True)

    print(f"Query: '{query}'\n")
    print("Resultados mais relevantes:")
    for sim, item in ranking:
        print(f"  {sim:.3f} → {item}")


def exemplo_com_store():
    """Exemplo de integração com store.py."""
    print("\n=== Integração com Store ===\n")

    from store import Store
    import numpy as np

    store = Store()

    # Criar embedding semântico
    embedder = create_semantic_embedding(
        method="sentence-transformers",
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Buscar itens extraídos
    query = "multas e penalidades"

    # Coletar todos os itens de todos os jobs
    jobs = store.list_jobs()
    todos_itens = []
    for job in jobs:
        job_detail = store.get_job(job["job_id"])
        if job_detail:
            for item in job_detail.get("items", []):
                todos_itens.append({
                    "job_id": job["job_id"],
                    "resumo": item.get("resumo", ""),
                    "dados": item.get("dados", {}),
                })

    if not todos_itens:
        print("Nenhum item encontrado.")
        return

    # Embeddings
    resumos = [item["resumo"] for item in todos_itens]
    embeddings = np.array(embedder.embed(resumos))
    query_emb = np.array(embedder.embed([query])[0])

    # Similaridades
    similarities = np.dot(embeddings, query_emb)

    # Ranking
    ranking = sorted(zip(similarities, todos_itens), reverse=True)

    print(f"Query: '{query}'\n")
    print("Resultados:")
    for sim, item in ranking[:5]:
        if sim > 0.3:  # Threshold de similaridade
            print(f"\n  [{sim:.3f}] {item['resumo']}")
            print(f"    Job: {item['job_id']}")


if __name__ == "__main__":
    import numpy as np

    # Executar exemplos
    try:
        exemplo_basic()
    except Exception as e:
        print(f"Erro no exemplo básico: {e}")

    try:
        exemplo_hibrido()
    except Exception as e:
        print(f"Erro no exemplo híbrido: {e}")

    try:
        exemplo_busca_semantica()
    except Exception as e:
        print(f"Erro na busca semântica: {e}")

    # try:
    #     exemplo_com_store()
    # except Exception as e:
    #     print(f"Erro na integração com store: {e}")
