from src.retrieval.fts_search import search_keyword
from src.retrieval.vector_search import search_semantic
from src.retrieval.reranker import rerank_results
from src.retrieval.schemas import RetrievedChunk


VECTOR_K = 10
FTS_K = 10
FINAL_K = 5


def search_hybrid(
    query: str,
    top_k: int = FINAL_K,
) -> list[RetrievedChunk]:

     
    # Step 1: Retrieve candidates
     

    vector_chunks = search_semantic(
        query=query,
        top_k=VECTOR_K,
        rerank=False,
    )

    fts_chunks = search_keyword(
        query=query,
        top_k=FTS_K,
    )

     
    # Step 2: Deduplicate
     

    combined: dict[str, RetrievedChunk] = {}

    for chunk in vector_chunks:
        combined[chunk.id] = chunk

    for chunk in fts_chunks:
        combined[chunk.id] = chunk

    candidates = list(combined.values())

     
    # Step 3: Cohere rerank
     

    return rerank_results(
        query=query,
        chunks=candidates,
        top_k=top_k,
    )