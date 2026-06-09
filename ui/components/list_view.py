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

#  Helpers  


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


#  Sub-renders  


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
    st.markdown(
        """
    <div class="cs-empty">
        <div class="cs-empty-icon">⚠️</div>
        <div class="cs-empty-text">
            Unable to connect to the service.<br>
            Please check the backend and try again.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
    <div class="cs-empty">
        <div class="cs-empty-icon">💬</div>
        <div class="cs-empty-text">
            No previous conversations.<br>
            Start a new one below.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_new_chat_button()


def render_conversation_card(conv: dict, index: int) -> None:
    """
    Render a single past conversation card with Continue and Delete buttons
    pinned to the right side of the card.

    Layout:
        ┌─────────────────────────────────────────────────────┐
        │  preview text (truncated)       [Continue] [Delete] │
        │  date label                                          │
        └─────────────────────────────────────────────────────┘

    Delete triggers inline confirm flow. Confirmed delete calls
    delete_conversation() and refreshes the list.

    Args:
        conv  : Dict with keys: session_id, preview, created_at
        index : Unique index to key the Streamlit buttons per card
    """
    safe_preview = html.escape(conv.get("preview", "No messages yet"))
    date_label = _format_date(conv.get("created_at", ""))
    session_id = conv["session_id"]
    confirm_key = f"confirm_delete_card_{index}"

    #  Card shell — flex row: text on left, buttons on right 
    # We render the card HTML, then use an overlapping columns trick:
    # a 1-px-height negative-margin container pulls Streamlit buttons
    # up into the card's visual space using CSS positioning.
    st.markdown(
        f"""
<div class="cs-card cs-card-row">
    <div class="cs-card-text">
        <div class="cs-card-preview">{safe_preview}</div>
        <div class="cs-card-date">{date_label}</div>
    </div>
    <div class="cs-card-actions-placeholder"></div>
</div>
<style>
  /* Card flex layout */
  .cs-card.cs-card-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.65rem 0.85rem;
  }}
  .cs-card-text {{
      flex: 1;
      min-width: 0;   /* allow text truncation inside flex child */
  }}
  /* Pull the immediately following Streamlit column block up into the card */
  .cs-card-row + div[data-testid="stHorizontalBlock"] {{
      margin-top: -3.05rem;
      margin-bottom: 0.4rem;
      padding-right: 0.5rem;
      display: flex;
      justify-content: flex-end;
  }}
  /* Make each button column as narrow as its content */
  .cs-card-row + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
      flex: 0 0 auto !important;
      width: auto !important;
      min-width: 0 !important;
      padding: 0 3px !important;
  }}
  /* Compact button sizing for card actions */
  .cs-card-row + div[data-testid="stHorizontalBlock"] button {{
      padding: 0.28rem 0.7rem !important;
      font-size: 0.75rem !important;
      width: auto !important;
      min-width: 0 !important;
      border-radius: 2px !important;
  }}
  /* Delete button — use VSCode error red */
  .cs-card-row + div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:last-child button {{
      background: transparent !important;
      border: 1px solid #f44747 !important;
      color: #f48771 !important;
  }}
  .cs-card-row + div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:last-child button:hover {{
      background: #1f1a1a !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )

    #  Inline delete confirmation 
    if st.session_state.get(confirm_key):
        st.markdown(
            '<div class="cs-confirm">⚠ Delete this conversation? This cannot be undone.</div>',
            unsafe_allow_html=True,
        )
        _, cy, cn = st.columns([6, 2, 2])
        with cy:
            if st.button("Yes, Delete", key=f"del_yes_{index}"):
                from api_client import delete_conversation

                delete_conversation(session_id)
                st.session_state[confirm_key] = False
                st.rerun()
        with cn:
            if st.button("Cancel", key=f"del_no_{index}"):
                st.session_state[confirm_key] = False
                st.rerun()
    else:
        # Spacer + two right-aligned action buttons
        _, col_cont, col_del = st.columns([7, 2, 2])
        with col_cont:
            if st.button("▶  Continue", key=f"open_conv_{index}"):
                success = load_conversation_messages(session_id)
                if success:
                    st.session_state.session_id = session_id
                    st.session_state.view = "chat"
                    st.session_state.confirm_delete = False
                st.rerun()
        with col_del:
            if st.button("🗑 Delete", key=f"delete_conv_{index}"):
                st.session_state[confirm_key] = True
                st.rerun()


#  Main list view render 


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
