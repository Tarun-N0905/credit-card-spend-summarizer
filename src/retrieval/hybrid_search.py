import json

from src.core.db import get_db
from src.core.embeddings import embed_query

from src.retrieval.schemas import RetrievedChunk


VECTOR_K = 15
FTS_K = 15
FINAL_K = 5

RRF_K = 60


def _vector_to_pg(vector: list[float]) -> str:
    return json.dumps(vector)


def _semantic(query: str):

    embedding = embed_query(query)

    vector_str = _vector_to_pg(embedding)

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
                VECTOR_K,
            ),
        ).fetchall()

    return rows


def _fts(query: str):

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
                FTS_K,
            ),
        ).fetchall()

    return rows


def search_hybrid(
    query: str,
    top_k: int = FINAL_K,
) -> list[RetrievedChunk]:

    semantic_rows = _semantic(query)
    fts_rows = _fts(query)

    scores = {}
    chunks = {}

    for rank, row in enumerate(semantic_rows, start=1):

        cid = str(row["id"])

        scores[cid] = scores.get(cid, 0.0) + (
            1 / (RRF_K + rank)
        )

        chunks[cid] = row

    for rank, row in enumerate(fts_rows, start=1):

        cid = str(row["id"])

        scores[cid] = scores.get(cid, 0.0) + (
            1 / (RRF_K + rank)
        )

        chunks[cid] = row

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []

    for cid, score in ranked:

        row = chunks[cid]

        results.append(
            RetrievedChunk(
                id=cid,
                chunk_text=row["chunk_text"],
                score=score,
                content_type=row["content_type"],
                page_number=row["page_number"],
                section_name=row["section_name"],
                metadata=row["metadata"] or {},
                position=row["position"],
            )
        )

    return results