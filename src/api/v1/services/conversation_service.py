import logging
from uuid import UUID

from src.api.v1.core.db import (
    get_rag_db_session,
)  # SQLAlchemy session for credit_multimodel_rag

logger = logging.getLogger(__name__)


def get_or_create_conversation(session_id: str) -> UUID:
    """
    Return the UUID of the existing conversation for session_id,
    or insert a new row and return its UUID.
    """
    with get_rag_db_session() as session:
        row = session.execute(
            "SELECT id FROM conversations WHERE session_id = :sid LIMIT 1",
            {"sid": session_id},
        ).fetchone()

        if row:
            return row[0]

        result = session.execute(
            "INSERT INTO conversations (session_id) VALUES (:sid) RETURNING id",
            {"sid": session_id},
        )
        session.commit()
        return result.fetchone()[0]


def load_recent_messages(conversation_id: UUID, limit: int = 6) -> list[dict]:
    """
    Return the last `limit` messages (chronological order) for a conversation.
    Returns list of {role: str, content: str}.
    """
    with get_rag_db_session() as session:
        rows = session.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = :cid
            ORDER BY created_at DESC
            LIMIT :lim
            """,
            {"cid": str(conversation_id), "lim": limit},
        ).fetchall()

    # Reverse so oldest is first (chronological)
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def save_message(conversation_id: UUID, role: str, content: str) -> None:
    """Persist a single message to the messages table."""
    with get_rag_db_session() as session:
        session.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (:cid, :role, :content)
            """,
            {"cid": str(conversation_id), "role": role, "content": content},
        )
        session.commit()
