"""
ui/app.py

Entry point for the Credit Card Spend Summarizer Streamlit UI.
Responsibilities here are limited to:
  - Page config
  - Global CSS
  - Session state initialisation
  - Top-level view routing (list | chat)

Run with:
  streamlit run ui/app.py
"""

import streamlit as st

from ui.state import init_session_state
from ui.components.header import render_header, render_error_banner
from ui.components.list_view import render_list_view
from ui.components.chat import (
    render_chat_controls,
    render_conversation,
    render_input_bar,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Spend Summarizer",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section.main,
.main .block-container {
    background-color: #0d0f14 !important;
}
[data-testid="stSidebar"] {
    background-color: #0d0f14 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 760px; }

/* ── Header ── */
.cs-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #1e2130;
}
.cs-header-icon {
    font-size: 1.6rem;
    background: linear-gradient(135deg, #4f6ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.cs-header-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: -0.3px;
}
.cs-header-sub {
    font-size: 0.72rem;
    color: #555870;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ── Conversation cards ── */
.cs-card {
    background: #161924;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.65rem;
    transition: border-color 0.15s;
}
.cs-card:hover { border-color: #3b5bdb; }
.cs-card-preview {
    font-size: 0.87rem;
    color: #c8cce0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
}
.cs-card-date {
    font-size: 0.68rem;
    color: #3a3e52;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section label ── */
.cs-section-label {
    font-size: 0.72rem;
    color: #3a3e52;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    margin-top: 0.4rem;
}

/* ── Message bubbles ── */
.cs-msg-row {
    display: flex;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.25s ease;
}
.cs-msg-row.user  { justify-content: flex-end; }
.cs-msg-row.assistant { justify-content: flex-start; }
.cs-bubble {
    max-width: 82%;
    padding: 0.75rem 1rem;
    border-radius: 16px;
    font-size: 0.88rem;
    line-height: 1.6;
    overflow-wrap: break-word;
    min-width: 60px;
}
.cs-bubble.user {
    background: linear-gradient(135deg, #3b5bdb, #4c6ef5);
    color: #fff;
    border-bottom-right-radius: 4px;
}
.cs-bubble.assistant {
    background: #161924;
    border: 1px solid #1e2130;
    color: #d4d8e8;
    border-bottom-left-radius: 4px;
}
.cs-ts {
    font-size: 0.65rem;
    color: #3a3e52;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
    text-align: right;
}
.cs-msg-row.assistant .cs-ts { text-align: left; }

/* ── Typing indicator ── */
.cs-typing {
    display: flex;
    gap: 5px;
    padding: 0.75rem 1rem;
    background: #161924;
    border: 1px solid #1e2130;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    width: fit-content;
}
.cs-dot {
    width: 7px; height: 7px;
    background: #4c6ef5;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.cs-dot:nth-child(2) { animation-delay: 0.2s; }
.cs-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Error banner ── */
.cs-error {
    background: #1e1218;
    border: 1px solid #5c1e2e;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #f87171;
    margin-bottom: 1rem;
}

/* ── Empty / info state ── */
.cs-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: #2e3248;
}
.cs-empty-icon { font-size: 2.8rem; margin-bottom: 0.8rem; }
.cs-empty-text { font-size: 0.85rem; line-height: 1.7; color: #3a3e52; }

/* ── Confirmation box ── */
.cs-confirm {
    background: #1a1220;
    border: 1px solid #5c1e2e;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.83rem;
    color: #f4a0a0;
    margin-top: 0.5rem;
}

/* ── Divider ── */
.cs-divider {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 1.2rem 0;
}

/* ── Input overrides ── */
.stTextInput > div > div > input {
    background: #161924 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b5bdb !important;
    box-shadow: 0 0 0 2px rgba(59,91,219,0.18) !important;
}

/* ── Button overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #3b5bdb, #4c6ef5) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    white-space: nowrap !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40%           { transform: translateY(-6px); }
}
</style>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Top-level entry point.

    Order:
      1. Initialise session state
      2. Render persistent header and error banner
      3. Route to correct view:
           list → render_list_view()
           chat → render_chat_controls() + render_conversation() + render_input_bar()
    """
    init_session_state()
    render_header()
    render_error_banner()

    if st.session_state.view == "list":
        render_list_view()
    else:
        render_chat_controls()
        render_conversation()
        render_input_bar()


if __name__ == "__main__":
    main()
