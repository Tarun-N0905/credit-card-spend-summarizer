"""
ui/state.py

All Streamlit session state initialisation and navigation helpers.
No rendering happens here — pure state management only.

Imported by:
  - ui/app.py
  - ui/components/chat.py
  - ui/components/list_view.py
  - ui/api_client.py
"""

import uuid
from datetime import datetime

import streamlit as st


def init_session_state() -> None:
    """
    Initialise all required session state keys on first page load.
    Safe to call on every Streamlit rerun — only sets keys that
    do not already exist.

    Keys:
      view           : "list" | "chat" — controls which screen renders
      session_id     : UUID string of the active conversation (None in list)
      messages       : list of {role, content, timestamp} for active chat
      is_loading     : bool — shows typing indicator while awaiting API
      error          : str | None — user-visible error message
      confirm_delete : bool — whether delete confirmation is showing
      conversations  : list of past conversation dicts fetched from API
      backend_error  : bool — True if backend was unreachable on list load
    """
    defaults = {
        "view": "list",
        "session_id": None,
        "messages": [],
        "is_loading": False,
        "error": None,
        "confirm_delete": False,
        "conversations": [],
        "backend_error": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_error() -> None:
    """Reset st.session_state.error to None."""
    st.session_state.error = None


def add_message(role: str, content: str) -> None:
    """
    Append a new message dict to st.session_state.messages.

    Args:
        role    : "user" or "assistant"
        content : the message text
    """
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M"),
    })


def start_new_chat() -> None:
    """
    Generate a fresh session UUID and switch to the chat view.

    UUID is generated client-side — no backend call needed here.
    The conversation row is created in the DB on the first message sent
    via get_or_create_conversation() in the FastAPI chat endpoint.
    """
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.error = None
    st.session_state.confirm_delete = False
    st.session_state.view = "chat"


def go_to_list() -> None:
    """
    Return to the list view and clear all active chat state.

    The conversation is preserved in the DB — this does not delete anything.
    conversations and backend_error are reset so the list re-fetches fresh
    on the next render.
    """
    st.session_state.view = "list"
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.error = None
    st.session_state.confirm_delete = False
    st.session_state.conversations = []
    st.session_state.backend_error = False
