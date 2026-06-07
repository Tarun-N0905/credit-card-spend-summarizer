
"""
src/core/db.py

All runtime database operations for the Credit Card Spend Summarizer.

Responsibilities:
  - Connection management (single connection pool via psycopg)
  - Document registration (upsert_document)
  - Chunk storage with embeddings (store_chunks)
  - Hash lookup for deduplication (get_existing_hashes)
  - Conversation and message persistence
  - Customer/transaction queries called by SQL tools

Schema is managed separately via schema.sql — no DDL here.
"""

import json
import logging
import uuid
from contextlib import contextmanager
from typing import Generator
import psycopg
from psycopg.rows import dict_row
from src.core.embeddings import embed_documents
from src.core.settings import settings
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _get_connection() -> psycopg.Connection:
    """Open and return a new psycopg connection using the configured DSN.

    A new connection is created per call. For a production service a
    connection pool (psycopg_pool.ConnectionPool) would be used instead,
    but for a capstone project a per-request connection is straightforward
    and avoids pool lifecycle complexity.

    Returns:
        An open psycopg.Connection with autocommit=False (default).

    Raises:
        psycopg.OperationalError : If the database is unreachable.
    """
    return psycopg.connect(settings.pg_connection_string, row_factory=dict_row)

    


@contextmanager
def get_db() -> Generator[psycopg.Connection, None, None]:
    """Context manager that yields a DB connection and handles cleanup.

    Commits on clean exit, rolls back on any exception, and always closes
    the connection. Use this for all DB operations:

        with get_db() as conn:
            conn.execute(...)

    Yields:
        psycopg.Connection
    """
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Document registration
# ---------------------------------------------------------------------------

def upsert_document(document_name: str, document_type: str) -> str:
    """Insert a new document record or return the existing one's UUID.

    Re-ingesting the same filename reuses the same doc_id so chunk rows
    remain consistently associated and can be cleaned up by doc_id if
    needed (ON DELETE CASCADE on the document_chunks FK).

    Args:
        document_name : Filename of the PDF (e.g. "rewards_guide.pdf").
        document_type : Full path or type label for the document.

    Returns:
        UUID string of the document record.
    """
    with get_db() as conn:
        # Check if document already exists by name
        row = conn.execute(
            "SELECT id FROM documents WHERE document_name = %s",
            (document_name,),
        ).fetchone()

        if row:
            logger.info("Document already registered: %s → %s", document_name, row["id"])
            return str(row["id"])

        doc_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO documents (id, document_name, document_type)
            VALUES (%s, %s, %s)
            """,
            (doc_id, document_name, document_type),
        )
        logger.info("Registered new document: %s → %s", document_name, doc_id)
        return doc_id


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Embed and store a list of deduplicated chunks into document_chunks.

    Steps:
      1. Extract content strings from all chunks
      2. Embed in batches via embed_documents()
      3. INSERT each chunk row with its vector, metadata, and hash

    The chunk dicts are expected to have "chunk_hash" already injected by
    deduplication.deduplicate_chunks() so no re-hashing is needed here.

    Args:
        chunks : List of chunk dicts with keys:
                   content, content_type, chunk_hash, metadata
                 metadata is expected to have: page_number, section,
                 element_type, position (JSONB dict or None)
        doc_id : UUID string of the parent document record.

    Returns:
        Number of chunks successfully stored.

    Raises:
        Exception : Propagates DB or embedding errors to the caller.
    """
    if not chunks:
        logger.info("store_chunks: no chunks to store")
        return 0

    # ── Step 1: Batch embed all chunk texts ───────────────────────────────
    texts = [c["content"] for c in chunks]
    embeddings = embed_documents(texts)

    # ── Step 2: Insert each chunk row ─────────────────────────────────────
    stored = 0
    with get_db() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            meta = chunk.get("metadata", {})
            chunk_id = str(uuid.uuid4())

            conn.execute(
                """
                INSERT INTO document_chunks (
                    id,
                    document_id,
                    chunk_hash,
                    chunk_text,
                    embedding,
                    content_type,
                    page_number,
                    section_name,
                    metadata,
                    position
                ) VALUES (
                    %s, %s, %s, %s,
                    %s::vector,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_hash) DO NOTHING
                """,
                (
                    chunk_id,
                    doc_id,
                    chunk["chunk_hash"],
                    chunk["content"],
                    json.dumps(embedding),          # cast to vector in SQL
                    chunk.get("content_type"),
                    meta.get("page_number"),
                    meta.get("section"),
                    json.dumps(meta),               # full metadata as JSONB
                    json.dumps(meta.get("position")) if meta.get("position") else None,
                ),
            )
            stored += 1

    logger.info("store_chunks: stored %d chunks for doc_id=%s", stored, doc_id)
    return stored


# ---------------------------------------------------------------------------
# Deduplication support
# ---------------------------------------------------------------------------

def get_existing_hashes() -> set[str]:
    """Fetch all chunk_hash values currently stored in document_chunks.

    Called by deduplication.deduplicate_chunks() before ingestion so that
    chunks already present from a previous ingestion run are not re-inserted.

    Returns:
        Set of 64-character SHA256 hex strings.

    Raises:
        Exception : Propagates DB errors — let ingestion handle them.
    """
    with get_db() as conn:
        rows = conn.execute("SELECT chunk_hash FROM document_chunks").fetchall()
    hashes = {row["chunk_hash"] for row in rows}
    logger.debug("get_existing_hashes: %d hashes loaded from DB", len(hashes))
    return hashes


# ---------------------------------------------------------------------------
# Conversation storage
# ---------------------------------------------------------------------------

def get_or_create_conversation(session_id: str) -> str:
    """Return the conversation UUID for a session, creating it if needed.

    Each browser session (identified by session_id UUID) maps to exactly
    one conversation record. This is looked up on every chat request so
    messages are appended to the correct conversation.

    Args:
        session_id : UUID string from the Streamlit session state.

    Returns:
        UUID string of the conversation record.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE session_id = %s",
            (session_id,),
        ).fetchone()

        if row:
            return str(row["id"])

        conversation_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO conversations (id, session_id) VALUES (%s, %s)",
            (conversation_id, session_id),
        )
        logger.info("Created conversation %s for session %s", conversation_id, session_id)
        return conversation_id


def save_message(conversation_id: str, role: str, content: str) -> None:
    """Persist a single chat message to the messages table.

    Called after every user input and agent response so the full
    conversation history is stored in PostgreSQL.

    Args:
        conversation_id : UUID of the parent conversation record.
        role            : "user" or "assistant".
        content         : The message text.
    """
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), conversation_id, role, content),
        )


def get_conversation_messages(conversation_id: str) -> list[dict]:
    """Retrieve all messages for a conversation ordered by creation time.

    Used by the LangGraph agent to reconstruct conversation history on
    each request so the LLM has full context of the prior turns.

    Args:
        conversation_id : UUID of the conversation.

    Returns:
        List of dicts with keys: role, content, created_at.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_db_connection() -> bool:
    """Verify the database is reachable by running a lightweight query.

    Called by the FastAPI /health endpoint and the Streamlit system status
    page. Returns True if the DB responds, False otherwise — never raises.
    """
    try:
        with get_db() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False
    

# ---------------------------------------------------------------------------
# Conversation listing and deletion (used by API endpoints)
# ---------------------------------------------------------------------------

def list_conversations() -> list[dict]:
    """Return all conversations with a first-message preview and timestamp.

    Called by GET /api/v1/conversations. The preview is the content of the
    first user message in that conversation, truncated to 100 characters.
    Conversations with no messages are still returned with an empty preview.

    Returns:
        List of dicts with keys: session_id, preview, created_at.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                c.session_id,
                COALESCE(
                    LEFT(
                        (SELECT content FROM messages
                         WHERE conversation_id = c.id
                         ORDER BY created_at ASC
                         LIMIT 1),
                        100
                    ),
                    ''
                ) AS preview,
                c.created_at
            FROM conversations c
            ORDER BY c.created_at DESC
            """,
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(session_id: str) -> None:
    """Delete a conversation and all its messages by session_id.

    The messages table has ON DELETE CASCADE on the conversation_id FK,
    so deleting the conversation row removes all messages automatically.

    Args:
        session_id : The session UUID string from the Streamlit client.
    """
    with get_db() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE session_id = %s",
            (session_id,),
        )
    logger.info("Deleted conversation for session_id=%s", session_id)

def get_sql_database() -> SQLDatabase:
    return SQLDatabase.from_uri(
        settings.cc_db_connection_string
    )
