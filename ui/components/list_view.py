"""
ui/components/list_view.py

All rendering for the list view — past conversations, empty states,
backend error state, and the New Chat button.

Imported by:
  - ui/app.py
"""

from datetime import datetime

import streamlit as st

from state import start_new_chat, go_to_list
from api_client import fetch_conversations, load_conversation_messages, delete_conversation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_date(iso_str: str) -> str:
    """
    Convert an ISO datetime string into a readable date label
    e.g. "05 Jun 2026". Used on conversation cards.
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
    """
    if st.button("＋  New Chat", key="new_chat_btn"):
        start_new_chat()
        st.rerun()


def render_backend_error_state() -> None:
    """
    Render the error state shown when the backend is unreachable on load.
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
    conversations exist.
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
    Render a single past conversation card with Open and Delete buttons
    together on the right side of the card.

    Layout: card wraps a two-column row — preview+date on the left,
    Open + Delete buttons on the right — so both buttons sit inside
    the card boundary as a unified action group.

    Args:
        conv  : Dict with keys: session_id, preview, created_at
        index : Unique index to key the Streamlit buttons per card
    """
    preview = conv.get("preview", "No messages yet")
    date_label = _format_date(conv.get("created_at", ""))

    # Card shell — text content only; buttons rendered via Streamlit columns below
    st.markdown(f"""
    <div class="cs-card cs-card-row">
        <div class="cs-card-text">
            <div class="cs-card-preview">{preview}</div>
            <div class="cs-card-date">{date_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cs-card-btn-wrap cs-card-btn-left">', unsafe_allow_html=True)

    left, _ = st.columns([2, 8])
    with left:
        col_open, col_delete = st.columns([1, 1], gap="small")
        with col_open:
            if st.button("Open →", key=f"open_conv_{index}"):
                session_id = conv["session_id"]
                success = load_conversation_messages(session_id)
                if success:
                    st.session_state.session_id = session_id
                    st.session_state.view = "chat"
                    st.session_state.confirm_delete = False
                st.rerun()
        with col_delete:
            if st.button("🗑 Delete", key=f"delete_conv_{index}"):
                success = delete_conversation(conv["session_id"])
                if success:
                    st.session_state.conversations = [
                        c for c in st.session_state.conversations
                        if c["session_id"] != conv["session_id"]
                    ]
                st.rerun()
        
    
    st.markdown('</div>', unsafe_allow_html=True)


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
