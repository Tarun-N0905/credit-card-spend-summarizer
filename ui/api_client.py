"""
ui/api_client.py

All HTTP calls to the FastAPI backend.
No rendering happens here — pure API communication only.

Every function handles its own exceptions and sets st.session_state.error
with a friendly message on failure. Raw exceptions never propagate to
the Streamlit layer.

Imported by:
  - ui/components/chat.py
  - ui/components/list_view.py
"""

import streamlit as st
import requests

from state import clear_error, go_to_list

API_BASE_URL = "http://localhost:8000"


def fetch_conversations() -> bool:
    """
    Call GET /api/v1/conversations to load all past sessions ordered
    by most recent activity.

    On success:
      - Populates st.session_state.conversations with list of dicts:
        [{session_id, preview, created_at}, ...]
      - Sets backend_error = False
      - Returns True

    On failure:
      - Sets st.session_state.backend_error = True
      - Clears st.session_state.conversations
      - Returns False
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/conversations",
            timeout=10,
        )
        response.raise_for_status()
        st.session_state.conversations = response.json()
        st.session_state.backend_error = False
        return True
    except Exception:
        st.session_state.conversations = []
        st.session_state.backend_error = True
        return False


def load_conversation_messages(session_id: str) -> bool:
    """
    Call GET /api/v1/conversations/{session_id}/messages and populate
    st.session_state.messages with the full message history.

    Normalises API response to {role, content, timestamp} format
    that render_message() expects.

    On success: populates messages, returns True.
    On failure: sets error banner, stays on list view, returns False.

    Args:
        session_id : UUID string of the conversation to load.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/conversations/{session_id}/messages",
            timeout=10,
        )
        response.raise_for_status()
        raw_messages = response.json()
        st.session_state.messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": _format_ts(msg.get("created_at", "")),
            }
            for msg in raw_messages
        ]
        return True
    except Exception:
        st.session_state.error = (
            "⚠ Could not load that conversation. Please try again."
        )
        return False


def send_chat_message(user_input: str) -> str | None:
    """
    POST the user message to FastAPI POST /api/v1/chat.

    Passes session_id so the backend can persist the message to the
    correct conversation via get_or_create_conversation().

    On success: returns the assistant reply string.
    On failure: sets st.session_state.error, returns None.
    Never raises — all exceptions caught internally.

    Args:
        user_input : The user's message text.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": user_input,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("reply", "No response received.")
    except requests.exceptions.ConnectionError:
        st.session_state.error = (
            "⚠ Service temporarily unavailable. Please try again later."
        )
        return None
    except requests.exceptions.Timeout:
        st.session_state.error = (
            "⚠ The request timed out. Please try again."
        )
        return None
    except Exception:
        st.session_state.error = (
            "⚠ Service temporarily unavailable. Please try again later."
        )
        return None


def delete_conversation(session_id: str) -> bool:
    """
    Call DELETE /api/v1/conversations/{session_id} to permanently remove
    the conversation and all its messages from the database.

    On success: returns True — caller handles state reset and navigation.
    On failure: sets error banner, resets confirm_delete so the button
    reappears, returns False. Chat state is preserved on failure.

    Args:
        session_id : UUID string of the conversation to delete.
    """
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/conversations/{session_id}",
            timeout=10,
        )
        response.raise_for_status()
        return True
    except Exception:
        st.session_state.error = (
            "⚠ Could not delete the conversation. Please try again."
        )
        st.session_state.confirm_delete = False
        return False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _format_ts(iso_str: str) -> str:
    """
    Convert an ISO datetime string from the API into HH:MM display format.
    Returns the original string unchanged if parsing fails.

    Args:
        iso_str : ISO 8601 datetime string e.g. "2026-06-05T14:32:00"
    """
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso_str).strftime("%H:%M")
    except Exception:
        return iso_str
