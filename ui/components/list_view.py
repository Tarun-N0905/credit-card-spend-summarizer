"""
ui/components/list_view.py

All rendering for the list view — past conversations, empty states,
backend error state, and the New Chat button.

Imported by:
  - ui/app.py
"""

import html
from datetime import datetime

import streamlit as st

from state import start_new_chat, go_to_list
from api_client import fetch_conversations, load_conversation_messages


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_date(iso_str: str) -> str:
    """
    Convert an ISO datetime string into a readable date label
    e.g. "05 Jun 2026". Used on conversation cards.

    Args:
        iso_str : ISO 8601 datetime string.
    """
    try:
        return datetime.fromisoformat(iso_str).strftime("%d %b %Y")
    except Exception:
        return iso_str


# ── Sub-renders ───────────────────────────────────────────────────────────────

def render_new_chat_button() -> None:
    """
    Render the New Chat button. Always shown in the list view regardless
    of backend state so the user can always start a fresh conversation.
    Clicking generates a fresh UUID client-side and enters chat view.
    """
    if st.button("＋  New Chat", key="new_chat_btn"):
        start_new_chat()
        st.rerun()


def render_backend_error_state() -> None:
    """
    Render the error state shown when the backend is unreachable on load.

    Shows a friendly message with two buttons:
      Retry     — re-fetches conversations (reruns the page)
      New Chat  — always available even when the backend is down
    """
    st.markdown("""
    <div class="cs-empty">
        <div class="cs-empty-icon">⚠️</div>
        <div class="cs-empty-text">
            Unable to connect to the service.<br>
            Please check the backend and try again.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_retry, col_new = st.columns(2)
    with col_retry:
        if st.button("↺  Retry", key="retry_btn"):
            st.session_state.backend_error = False
            st.rerun()
    with col_new:
        render_new_chat_button()


def render_empty_conversations_state() -> None:
    """
    Render the empty state when the backend is healthy but no past
    conversations exist. Shows a prompt and the New Chat button.
    This is a normal happy-path state, not an error.
    """
    st.markdown("""
    <div class="cs-empty">
        <div class="cs-empty-icon">💬</div>
        <div class="cs-empty-text">
            No previous conversations.<br>
            Start a new one below.
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_new_chat_button()


def render_conversation_card(conv: dict, index: int) -> None:
    """
    Render a single past conversation as a styled card with an Open button.

    Shows the first message preview text and the conversation date.
    Preview text is HTML-escaped so special characters in message content
    never break the card layout.
    Clicking Open loads the conversation messages and switches to chat view.
    If loading fails, error banner shows and list view stays visible.

    Args:
        conv  : Dict with keys: session_id, preview, created_at
        index : Unique index to key the Streamlit button per card
    """
    safe_preview = html.escape(conv.get("preview", "No messages yet"))
    date_label   = _format_date(conv.get("created_at", ""))

    st.markdown(f"""
    <div class="cs-card">
        <div class="cs-card-preview">{safe_preview}</div>
        <div class="cs-card-date">{date_label}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Open", key=f"open_conv_{index}"):
        session_id = conv["session_id"]
        success = load_conversation_messages(session_id)
        if success:
            st.session_state.session_id = session_id
            st.session_state.view = "chat"
            st.session_state.confirm_delete = False
        st.rerun()


# ── Main list view render ─────────────────────────────────────────────────────

def render_list_view() -> None:
    """
    Render the full list view screen.

    Flow:
      1. fetch_conversations() from backend
      2. Backend error  → render_backend_error_state()
      3. No conversations → render_empty_conversations_state()
      4. Conversations exist → New Chat button + conversation cards
    """
    fetch_conversations()

    if st.session_state.backend_error:
        render_backend_error_state()
        return

    conversations = st.session_state.conversations

    if not conversations:
        render_empty_conversations_state()
        return

    render_new_chat_button()
    st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="cs-section-label">Previous Conversations</div>',
        unsafe_allow_html=True,
    )
    for i, conv in enumerate(conversations):
        render_conversation_card(conv, i)
