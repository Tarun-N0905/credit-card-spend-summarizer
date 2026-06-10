from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json
from src.api.v1.core.db import get_or_create_conversation, save_message
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

        conversation_id = get_or_create_conversation(body.session_id)
        save_message(conversation_id, "user", body.message)

        agent_response = run_credit_card_agent(body.message, session_id=body.session_id)

        reply = _extract_reply(agent_response)
        route = agent_response.get("route_taken", "unknown")
        print(f"[chat] route={route} query={body.message[:60]}")

        save_message(conversation_id, "assistant", reply)
        return {"reply": reply}

    except GuardrailViolation as exc:
        print(f"[chat] guardrail blocked: {exc.guard}")
        return JSONResponse(status_code=400, content={"detail": exc.message})
    except Exception as exc:
        print(f"[chat] error: {exc}")
        return JSONResponse(status_code=500, content=_UNAVAILABLE)


def _persist_reply(conversation_id: str, accumulated: list[str]):
    """Background task: joins accumulated tokens and saves to DB."""
    full_reply = "".join(accumulated).strip()
    if full_reply:
        try:
            save_message(conversation_id, "assistant", full_reply)
            print(f"[chat/stream] persisted reply ({len(full_reply)} chars)")
        except Exception as exc:
            print(f"[chat/stream] failed to persist reply: {exc}")


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, background_tasks: BackgroundTasks):
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

    Fix: save_message is registered as a BackgroundTask so it always runs
    after the stream is fully consumed, even if the generator exits early.
    """
    conversation_id = get_or_create_conversation(body.session_id)
    save_message(conversation_id, "user", body.message)

    accumulated: list[str] = []

    def event_stream():
        try:
            guard_input(body.message)
        except GuardrailViolation as exc:
            print(f"[chat/stream] guardrail blocked: {exc.guard}")
            yield f"data: [ERROR] {exc.message}\n\n"
            return

        for chunk in run_credit_card_agent_stream(
            body.message, session_id=body.session_id
        ):
            # Strip SSE prefix; skip control frames like [DONE] / [META] / [ERROR]
            payload = chunk.removeprefix("data: ").strip()
            if payload and not payload.startswith("["):
                try:
                    token = json.loads(payload)
                except Exception:
                    token = payload
                if isinstance(token, str):
                    accumulated.append(token)
            yield chunk

    # BackgroundTasks runs _after_ the StreamingResponse is fully sent,
    # guaranteeing the assistant reply is persisted regardless of client behaviour.
    background_tasks.add_task(_persist_reply, conversation_id, accumulated)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
        background=background_tasks,
    )