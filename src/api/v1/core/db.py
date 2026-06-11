import json
import logging
import uuid
from contextlib import contextmanager
from typing import Generator
import msgpack
import psycopg
from psycopg.rows import dict_row
from src.api.v1.core.embeddings import embed_documents
from src.api.v1.core.settings import settings
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)


def _get_connection() -> psycopg.Connection:
    return psycopg.connect(settings.pg_connection_string, row_factory=dict_row)


@contextmanager
def get_db() -> Generator[psycopg.Connection, None, None]:
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Document registration ────────────────────────────────────────────────────


def upsert_document(document_name: str, document_type: str) -> str:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE document_name = %s",
            (document_name,),
        ).fetchone()
        if row:
            logger.info(
                "Document already registered: %s → %s", document_name, row["id"]
            )
            return str(row["id"])
        doc_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO documents (id, document_name, document_type) VALUES (%s, %s, %s)",
            (doc_id, document_name, document_type),
        )
        logger.info("Registered new document: %s → %s", document_name, doc_id)
        return doc_id


# ── Chunk storage ────────────────────────────────────────────────────────────


def store_chunks(chunks: list[dict], doc_id: str) -> int:
    if not chunks:
        return 0
    texts = [c["content"] for c in chunks]
    embeddings = embed_documents(texts)
    stored = 0
    with get_db() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            meta = chunk.get("metadata", {})
            conn.execute(
                """
                INSERT INTO document_chunks (
                    id, document_id, chunk_hash, chunk_text, embedding,
                    content_type, page_number, section_name, metadata, position
                ) VALUES (
                    %s, %s, %s, %s, %s::vector,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_hash) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    doc_id,
                    chunk["chunk_hash"],
                    chunk["content"],
                    json.dumps(embedding),
                    chunk.get("content_type"),
                    meta.get("page_number"),
                    meta.get("section"),
                    json.dumps(meta),
                    json.dumps(meta.get("position")) if meta.get("position") else None,
                ),
            )
            stored += 1
    logger.info("store_chunks: stored %d chunks for doc_id=%s", stored, doc_id)
    return stored


# ── Deduplication support ────────────────────────────────────────────────────


def get_existing_hashes() -> set[str]:
    with get_db() as conn:
        rows = conn.execute("SELECT chunk_hash FROM document_chunks").fetchall()
    return {row["chunk_hash"] for row in rows}


# ── Checkpoint-backed conversation helpers ───────────────────────────────────


def _decode_ext_message(ext: msgpack.ExtType) -> dict | None:
    try:
        inner = msgpack.unpackb(ext.data, raw=False)
        if isinstance(inner, (list, tuple)) and len(inner) >= 3:
            fields = inner[2] if isinstance(inner[2], dict) else {}
        elif isinstance(inner, dict):
            fields = inner
        else:
            return None

        msg_type = (fields.get("type") or "").lower()
        raw_content = fields.get("content") or ""

        if isinstance(raw_content, list):
            content = " ".join(
                part.get("text", "")
                for part in raw_content
                if isinstance(part, dict) and part.get("text")
            ).strip()
        else:
            content = str(raw_content).strip()

        if msg_type == "human":
            return {"role": "user", "content": content}
        elif msg_type == "ai" and content:
            return {"role": "assistant", "content": content}
        return None
    except Exception as e:
        logger.debug("[ext] failed to decode ExtType: %s", e)
        return None


def _decode_dict_message(item: dict) -> dict | None:
    try:
        msg_type = (item.get("type") or "").lower()
        raw_content = item.get("content") or ""
        if isinstance(raw_content, list):
            content = " ".join(
                part.get("text", "")
                for part in raw_content
                if isinstance(part, dict) and part.get("text")
            ).strip()
        else:
            content = str(raw_content).strip()
        if msg_type == "human":
            return {"role": "user", "content": content}
        elif msg_type == "ai" and content:
            return {"role": "assistant", "content": content}
        return None
    except Exception:
        return None


def _extract_messages_from_blob(blob_value) -> list[dict]:
    """
    Decode the messages-channel checkpoint blob into a list of
    {"role": "user"|"assistant", "content": "..."} dicts.
    Includes both human AND AI messages (with non-empty content).
    """
    try:
        data = msgpack.unpackb(blob_value, raw=False)
    except Exception:
        try:
            data = (
                json.loads(blob_value)
                if isinstance(blob_value, (str, bytes))
                else blob_value
            )
        except Exception:
            return []

    if not isinstance(data, list):
        return []

    seen = set()
    result = []
    for item in data:
        if isinstance(item, msgpack.ExtType):
            msg = _decode_ext_message(item)
        elif isinstance(item, dict):
            msg = _decode_dict_message(item)
        else:
            continue
        if not msg:
            continue
        key = (msg["role"], msg["content"])
        if key not in seen:
            seen.add(key)
            result.append(msg)
    return result


def _get_latest_messages_blob(conn, thread_id: str) -> tuple[bytes | None, str | None]:
    cp_row = conn.execute(
        """
        SELECT
            checkpoint->>'ts' AS ts,
            checkpoint->'channel_versions'->>'messages' AS messages_version
        FROM checkpoints
        WHERE thread_id = %s
        ORDER BY checkpoint->>'ts' DESC
        LIMIT 1
        """,
        (thread_id,),
    ).fetchone()

    if not cp_row:
        return None, None

    ts = cp_row["ts"]
    messages_version = cp_row["messages_version"]

    if messages_version:
        blob_row = conn.execute(
            """
            SELECT blob
            FROM checkpoint_blobs
            WHERE thread_id = %s
              AND channel = 'messages'
              AND version = %s
            """,
            (thread_id, messages_version),
        ).fetchone()
    else:
        blob_row = conn.execute(
            """
            SELECT blob
            FROM checkpoint_blobs
            WHERE thread_id = %s
              AND channel = 'messages'
            ORDER BY version DESC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()

    blob = blob_row["blob"] if blob_row else None
    return blob, ts


def _decode_agent_response_ext(ext: msgpack.ExtType) -> str | None:
    """
    LangGraph serialises AgentResponse as ExtType(code=5) whose data unpacks to:
      [module_str, 'AgentResponse', {field: value, ...}, 'model_validate_json']
    The answer lives at inner[2]['answer'].
    """
    try:
        inner = msgpack.unpackb(ext.data, raw=False)
        if not (isinstance(inner, (list, tuple)) and len(inner) >= 3):
            return None
        if inner[1] != "AgentResponse":
            return None
        fields = inner[2]
        if not isinstance(fields, dict):
            return None
        answer = fields.get("answer") or ""
        return str(answer).strip() or None
    except Exception as e:
        logger.debug("[_decode_agent_response_ext] failed: %s", e)
        return None


def _decode_response_blob(blob_value) -> str | None:
    """
    Decode a response-channel blob.

    LangGraph stores AgentResponse as ExtType(code=5):
      inner = [module, 'AgentResponse', {'answer': '...', ...}, 'model_validate_json']

    Falls back to plain dict / str for forward-compatibility.
    """
    try:
        data = msgpack.unpackb(blob_value, raw=False)
    except Exception:
        try:
            data = (
                json.loads(blob_value)
                if isinstance(blob_value, (str, bytes))
                else blob_value
            )
        except Exception:
            return None

    if isinstance(data, str):
        return data.strip() or None

    if isinstance(data, dict):
        answer = data.get("answer") or data.get("content") or ""
        return str(answer).strip() or None

    # ExtType — the actual production format (AgentResponse)
    if isinstance(data, msgpack.ExtType):
        return _decode_agent_response_ext(data)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, msgpack.ExtType):
                answer = _decode_agent_response_ext(item)
                if answer:
                    return answer
            elif isinstance(item, dict):
                answer = item.get("answer") or item.get("content") or ""
                if answer:
                    return str(answer).strip()

    return None


def _get_response_pairs(conn, thread_id: str) -> list[tuple[str, str]]:
    """
    Return (query, answer) pairs from response-channel blobs, ordered by version.
    The query field lets us match each answer back to the right human message
    without relying on fragile positional pairing.
    """
    rows = conn.execute(
        """
        SELECT blob
        FROM checkpoint_blobs
        WHERE thread_id = %s
          AND channel = 'response'
        ORDER BY version ASC
        """,
        (thread_id,),
    ).fetchall()
    pairs = []
    for row in rows:
        try:
            data = msgpack.unpackb(row["blob"], raw=False)
        except Exception:
            continue
        if not isinstance(data, msgpack.ExtType):
            continue
        try:
            inner = msgpack.unpackb(data.data, raw=False)
            if not (isinstance(inner, (list, tuple)) and len(inner) >= 3):
                continue
            if inner[1] != "AgentResponse":
                continue
            fields = inner[2]
            if not isinstance(fields, dict):
                continue
            query = str(fields.get("query") or "").strip()
            answer = str(fields.get("answer") or "").strip()
            if query and answer:
                pairs.append((query, answer))
        except Exception:
            continue
    return pairs


def get_conversation_messages(session_id: str) -> list[dict]:
    """
    Retrieve the full conversation (user + assistant messages) for a session.

    Primary strategy: extract both human and AI messages from the checkpoint's
    messages channel blob — this works when response_node appends an AIMessage
    (post-fix sessions).

    Fallback: match response-channel blobs to human messages by query text.
    This handles old checkpoints where response_node never wrote AIMessages.
    Matching by query instead of position avoids off-by-one errors caused by
    duplicate human messages being deduped while their responses are not.
    """
    with get_db() as conn:
        blob, ts = _get_latest_messages_blob(conn, session_id)
        if not blob:
            return []

        all_messages = _extract_messages_from_blob(blob)
        has_ai = any(m["role"] == "assistant" for m in all_messages)

        if has_ai:
            for m in all_messages:
                m["created_at"] = ts or ""
            return all_messages

        # Fallback: use response channel, match by query text
        response_pairs = _get_response_pairs(conn, session_id)

    # Build query→answer map (last answer wins for repeated queries)
    query_to_answer: dict[str, str] = {}
    for query, answer in response_pairs:
        query_to_answer[query] = answer

    human_messages = [m for m in all_messages if m["role"] == "user"]

    result = []
    for human in human_messages:
        human["created_at"] = ts or ""
        result.append(human)
        answer = query_to_answer.get(human["content"])
        if answer:
            result.append(
                {"role": "assistant", "content": answer, "created_at": ts or ""}
            )

    return result


def list_conversations() -> list[dict]:
    with get_db() as conn:
        thread_rows = conn.execute(
            """
            SELECT
                thread_id,
                MIN(checkpoint->>'ts') AS created_at
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY MIN(checkpoint->>'ts') DESC
            """,
        ).fetchall()

        if not thread_rows:
            return []

        results = []
        for row in thread_rows:
            blob, _ = _get_latest_messages_blob(conn, row["thread_id"])
            preview = ""
            if blob:
                msgs = _extract_messages_from_blob(blob)
                user_msgs = [m for m in msgs if m["role"] == "user"]
                if user_msgs:
                    preview = user_msgs[0]["content"][:100]

            results.append(
                {
                    "session_id": row["thread_id"],
                    "preview": preview,
                    "created_at": row["created_at"] or "",
                }
            )

        return results


def delete_conversation(session_id: str) -> None:
    with get_db() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(
                f"DELETE FROM {table} WHERE thread_id = %s",  # noqa: S608
                (session_id,),
            )
    logger.info("Deleted checkpoints for session_id=%s", session_id)


# ── Health check ─────────────────────────────────────────────────────────────


def check_db_connection() -> bool:
    try:
        with get_db() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False


# ── SQL agent helper ─────────────────────────────────────────────────────────


def get_sql_database() -> SQLDatabase:
    return SQLDatabase.from_uri(settings.cc_db_connection_string)
