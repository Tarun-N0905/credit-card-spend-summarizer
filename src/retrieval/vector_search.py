import json

from src.core.db import get_db
from src.core.embeddings import embed_query

from src.retrieval.schemas import RetrievedChunk
from src.retrieval.reranker import rerank_results


VECTOR_TOP_K = 20
FINAL_TOP_K = 5


def _vector_to_pg(vector: list[float]) -> str:
    return json.dumps(vector)


def search_semantic(
    query: str,
    top_k: int = FINAL_TOP_K,
) -> list[RetrievedChunk]:

    query_embedding = embed_query(query)

    vector_str = _vector_to_pg(query_embedding)

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                chunk_text,
                content_type,
                page_number,
                section_name,
                metadata,
                position,
                1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (
                vector_str,
                vector_str,
                VECTOR_TOP_K,
            ),
        ).fetchall()

    chunks = [
        RetrievedChunk(
            id=str(r["id"]),
            chunk_text=r["chunk_text"],
            score=float(r["score"]),
            content_type=r["content_type"],
            page_number=r["page_number"],
            section_name=r["section_name"],
            metadata=r["metadata"] or {},
            position=r["position"],
        )
        for r in rows
    ]

    return rerank_results(
        query=query,
        chunks=chunks,
        top_k=top_k,
    )