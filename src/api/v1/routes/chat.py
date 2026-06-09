from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.v1.core.db import get_or_create_conversation, save_message
from src.api.v1.agents.graph import run_credit_card_agent

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


def _extract_reply(agent_response: dict) -> str:
    """
    SpendSummaryResponse  → uses 'summary_text'
    AgentResponse (KB)    → uses 'answer'
    Error fallback        → uses 'answer'
    """
    return (
        agent_response.get("summary_text")
        or agent_response.get("answer")
        or "I couldn't generate a response. Please try again."
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        conversation_id = get_or_create_conversation(body.session_id)
        save_message(conversation_id, "user", body.message)

        agent_response = run_credit_card_agent(body.message, session_id=body.session_id)

        reply = _extract_reply(agent_response)
        route = agent_response.get("route_taken", "unknown")
        print(f"[chat] route={route} query={body.message[:60]}")

        save_message(conversation_id, "assistant", reply)
        return {"reply": reply}

    except Exception as exc:
        print(f"[chat] error: {exc}")
        return JSONResponse(status_code=500, content=_UNAVAILABLE)
