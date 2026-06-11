from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.v1.core.db import (
    list_conversations,
    get_conversation_messages,
    delete_conversation,
)

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


@router.get("/conversations")
async def list_all_conversations():
    try:
        return list_conversations()
    except Exception as e:
        print(f"[conversations] error: {e}")
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


@router.get("/conversations/{session_id}/messages")
async def load_conversation_messages(session_id: str):
    try:
        msgs = get_conversation_messages(session_id)
        print(f"[messages] session={session_id} count={len(msgs)}")
        return msgs
    except Exception as e:
        print(f"[messages] error: {e}")
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


@router.delete("/conversations/{session_id}")
async def remove_conversation(session_id: str):
    try:
        delete_conversation(session_id)
        return {"status": "deleted"}
    except Exception:
        return JSONResponse(status_code=500, content=_UNAVAILABLE)
