from src.core.db import get_db

from src.retrieval.schemas import RetrievedChunk


def search_keyword(
    query: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:

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
                ts_rank_cd(
                    search_vector,
                    plainto_tsquery(
                        'english',
                        %s
                    )
                ) AS score
            FROM document_chunks
            WHERE search_vector @@ plainto_tsquery(
                'english',
                %s
            )
            ORDER BY score DESC
            LIMIT %s
            """,
            (
                query,
                query,
                top_k,
            ),
        ).fetchall()

    return [
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