import json

from src.api.v1.core.db import get_db
from src.api.v1.core.embeddings import embed_query

from src.api.v1.retrieval.schemas import RetrievedChunk
from src.api.v1.retrieval.reranker import rerank_results

VECTOR_TOP_K = 10
FINAL_TOP_K = 5


def _vector_to_pg(vector: list[float]) -> str:
    return json.dumps(vector)


def search_semantic(
    query: str,
    top_k: int = FINAL_TOP_K,
    rerank: bool = True,
) -> list[RetrievedChunk]:
    """
    Semantic retrieval using pgvector.

    Flow:
        Query
          ↓
        Embed Query
          ↓
        Cosine Similarity Search
          ↓
        Top 10 Candidates
          ↓
        (Optional) Cohere Rerank
          ↓
        Final Results
    """

    query_embedding = embed_query(query)

    vector_str = _vector_to_pg(query_embedding)

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                dc.id,
                dc.chunk_text,
                dc.content_type,
                dc.page_number,
                dc.section_name,
                dc.metadata,
                dc.position,
                d.document_name,
                1 - (dc.embedding <=> %s::vector) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            ORDER BY dc.embedding <=> %s::vector
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
            document_name=r["document_name"],
            metadata=r["metadata"] or {},
            position=r["position"],
        )
        for r in rows
    ]

    if not rerank:
        return chunks

    return rerank_results(
        query=query,
        chunks=chunks,
        top_k=min(top_k, len(chunks)),
    )
