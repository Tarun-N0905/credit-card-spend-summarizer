from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.v1.core.db import (
    list_conversations,
    get_or_create_conversation,
    get_conversation_messages,
    delete_conversation,
)

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


@router.get("/conversations")
async def list_all_conversations():
    """
    Returns all past conversations as a list.

    Calls list_conversations() from db.py.
    Response: [{"session_id": str, "preview": str, "created_at": str}, ...]

    Used by ui/api_client.py fetch_conversations().
    """
    try:
        return list_conversations()
    except Exception:
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


@router.get("/conversations/{session_id}/messages")
async def load_conversation_messages(session_id: str):
    """
    Returns all messages for a given session_id.

    Flow:
      1. Resolve session_id → conversation UUID via get_or_create_conversation()
         (safe to call — creates only if missing, which won't happen for known sessions)
      2. Call get_conversation_messages(conversation_id)
      3. Return list of {role, content, created_at}

    Used by ui/api_client.py load_conversation_messages().
    """
    try:
        conversation_id = get_or_create_conversation(session_id)
        messages = get_conversation_messages(conversation_id)
        return messages
    except Exception:
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


@router.delete("/conversations/{session_id}")
async def remove_conversation(session_id: str):
    """
    Deletes a conversation and all its messages.

    Calls delete_conversation(session_id) from db.py.
    The DB schema cascades the delete to the messages table automatically.
    Response: {"status": "deleted"}

    Used by ui/api_client.py delete_conversation().
    """
    try:
        delete_conversation(session_id)
        return {"status": "deleted"}
    except Exception:
        return JSONResponse(status_code=500, content=_UNAVAILABLE)
