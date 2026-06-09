from src.api.v1.core.db import get_db

from src.api.v1.retrieval.schemas import RetrievedChunk


def search_keyword(
    query: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:

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
                ts_rank_cd(
                    dc.search_vector,
                    plainto_tsquery(
                        'english',
                        %s
                    )
                ) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.search_vector @@ plainto_tsquery(
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
            document_name=r["document_name"],
            metadata=r["metadata"] or {},
            position=r["position"],
        )
        for r in rows
    ]
