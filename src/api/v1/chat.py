import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.db import get_or_create_conversation, save_message, get_conversation_messages
from src.agents.graph import run_credit_card_agent

logger = logging.getLogger(__name__)

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
      3. get_conversation_messages(conversation_id) → history for context
      4. Run LangGraph agent (routes to KB or SQL)
      5. save_message(conversation_id, "assistant", reply)
      6. Return {"reply": reply}

    All exceptions return the generic safe error message.
    """
    try:
        conversation_id = get_or_create_conversation(body.session_id)
        save_message(conversation_id, "user", body.message)

        # Load conversation history (for context awareness)
        history = get_conversation_messages(conversation_id)

        # --- Call real LangGraph agent ---
        agent_response = run_credit_card_agent(body.message)
        
        # Extract the answer text
        reply = agent_response.get("answer", "I couldn't generate a response. Please try again.")
        
        # Log route taken for monitoring
        route = agent_response.get("route_taken", "unknown")
        logger.info(f"[chat] Route: {route}, Query: {body.message[:50]}...")

        save_message(conversation_id, "assistant", reply)
        return {"reply": reply}

    except Exception as exc:
        logger.error(f"[chat] Error: {exc}")
        return JSONResponse(status_code=500, content=_UNAVAILABLE)