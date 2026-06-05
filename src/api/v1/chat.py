from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.db import get_or_create_conversation, save_message, get_conversation_messages

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    Accepts {session_id, message}.

    Flow:
      1. get_or_create_conversation(session_id)  → conversation UUID
      2. save_message(conversation_id, "user", message)
      3. get_conversation_messages(conversation_id) → history for agent
      4. Run LangGraph agent (stubbed until agent is built)
      5. save_message(conversation_id, "assistant", reply)
      6. Return {"reply": reply}

    All exceptions return the generic safe error message.
    """
    try:
        conversation_id = get_or_create_conversation(body.session_id)
        save_message(conversation_id, "user", body.message)

        history = get_conversation_messages(conversation_id)

        # --- Agent call (stubbed — replace with real agent when built) ---
        reply = _stub_agent(history, body.message)
        # -----------------------------------------------------------------

        save_message(conversation_id, "assistant", reply)
        return {"reply": reply}

    except Exception:
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


def _stub_agent(history: list, message: str) -> str:
    """
    Temporary stub that returns a placeholder reply.
    Replace the body of this function with the real LangGraph agent call
    once src/agents/graph.py is implemented.
    """
    return (
        "Agent not yet connected. Your message was saved and the system is ready "
        "for the LangGraph integration."
    )