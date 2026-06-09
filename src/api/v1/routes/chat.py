from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.api.v1.core.db import get_or_create_conversation, save_message
from src.api.v1.agents.graph import run_credit_card_agent, run_credit_card_agent_stream

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


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """
    Streaming endpoint for Streamlit.

    Runs the full LangGraph pipeline (routing, retrieval, SQL) and then
    streams the final LLM answer token-by-token as Server-Sent Events.

    SSE event format
    ────────────────
    data: <token>          — incremental answer text; accumulate on the client
    data: [DONE]           — answer is complete; stop writing to the stream widget
    data: [META] <json>    — final metadata: route_taken, page_no, document_name,
                             sql_query_executed, image_paths
    data: [ERROR] <msg>    — something went wrong; display msg to the user

    """
    conversation_id = get_or_create_conversation(body.session_id)
    save_message(conversation_id, "user", body.message)

    accumulated: list[str] = []

    def event_stream():
        for chunk in run_credit_card_agent_stream(
            body.message, session_id=body.session_id
        ):
            # Collect answer tokens so we can persist the full reply afterwards.
            # chunk format: "data: <payload>\n\n"
            payload = chunk.removeprefix("data: ").rstrip("\n")
            if payload and not payload.startswith("["):
                accumulated.append(payload)
            yield chunk

        # Persist the complete reply after streaming finishes.
        full_reply = "".join(accumulated).strip()
        if full_reply:
            try:
                save_message(conversation_id, "assistant", full_reply)
            except Exception as exc:
                print(f"[chat/stream] failed to persist reply: {exc}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent nginx / proxies from buffering SSE chunks.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
