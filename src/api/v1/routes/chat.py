from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from src.api.v1.agents.graph import run_credit_card_agent, run_credit_card_agent_stream
from src.api.v1.core.guardrails import GuardrailViolation, guard_input

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


def _extract_reply(agent_response: dict) -> str:
    return (
        agent_response.get("summary_text")
        or agent_response.get("answer")
        or "I couldn't generate a response. Please try again."
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        guard_input(body.message)
        agent_response = run_credit_card_agent(body.message, session_id=body.session_id)
        reply = _extract_reply(agent_response)
        print(
            f"[chat] route={agent_response.get('route_taken')} query={body.message[:60]}"
        )
        return {"reply": reply}
    except GuardrailViolation as exc:
        print(f"[chat] guardrail blocked: {exc.guard}")
        return JSONResponse(status_code=400, content={"detail": exc.message})
    except Exception as exc:
        print(f"[chat] error: {exc}")
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """
    Streaming endpoint. Checkpointer persists all messages automatically

    SSE format (unchanged):
      data: <token>        — incremental answer
      data: [DONE]         — stream complete
      data: [META] <json>  — route_taken, page_no, document_name, sql_query_executed, image_paths
      data: [ERROR] <msg>  — error
    """

    def event_stream():
        try:
            guard_input(body.message)
        except GuardrailViolation as exc:
            print(f"[chat/stream] guardrail blocked: {exc.guard}")
            yield f"data: [ERROR] {exc.message}\n\n"
            return

        yield from run_credit_card_agent_stream(
            body.message, session_id=body.session_id
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
