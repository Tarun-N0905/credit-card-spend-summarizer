import cohere

from src.api.v1.core.settings import settings
from src.api.v1.retrieval.schemas import RetrievedChunk

_client = cohere.ClientV2(api_key=settings.cohere_api_key)


def rerank_results(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:

    if not chunks:
        return []

    response = _client.rerank(
        model=settings.cohere_rerank_model,
        query=query,
        documents=[c.chunk_text for c in chunks],
        top_n=min(top_k, len(chunks)),
    )

    reranked = []

    for result in response.results:
        reranked.append(chunks[result.index])

    return reranked
