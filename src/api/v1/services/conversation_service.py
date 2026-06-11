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
