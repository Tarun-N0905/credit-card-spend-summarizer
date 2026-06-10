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

from state import init_session_state
from components.header import render_header, render_error_banner
from components.list_view import render_list_view
from components.chat import (
    render_chat_controls,
    render_conversation,
    render_input_bar,
)

#  Page config

st.set_page_config(
    page_title="North Star Agent",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

#  Global CSS

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;500;600&family=Consolas&display=swap');

/*
 * VSCode Dark+ theme
 
 * bg:          #1e1e1e   (editor background)
 * sidebar-bg:  #252526   (side bar)
 * panel-bg:    #2d2d2d   (input / panel areas)
 * border:      #3e3e42   (widget border)
 * accent:      #007acc   (VS blue — focus rings, links)
 * accent-alt:  #0e639c   (button hover)
 * token-blue:  #569cd6   (keyword blue)
 * token-green: #4ec9b0   (class / type teal)
 * token-str:   #ce9178   (string orange)
 * token-cmt:   #6a9955   (comment green)
 * fg:          #d4d4d4   (default text)
 * fg-muted:    #858585   (line numbers / hints)
 * fg-inactive: #3e3e42   (disabled)
 * error-bg:    #1f1a1a
 * error-border:#f44747
 * error-fg:    #f44747
 
*/

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #1e1e1e;
    color: #d4d4d4;
}
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section.main,
.main .block-container {
    background-color: #1e1e1e !important;
}
[data-testid="stSidebar"] {
    background-color: #252526 !important;
    border-right: 1px solid #3e3e42;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 1100px; }

/* ── Header — mimics VSCode title bar ── */
.cs-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid #3e3e42;
}
.cs-header-icon {
    font-size: 1.4rem;
    /* VSCode blue icon tint */
    filter: drop-shadow(0 0 6px #007acc88);
}
.cs-header-title {
    font-size: 1.05rem;
    font-weight: 400;
    color: #cccccc;
    letter-spacing: 0.01em;
    font-family: 'Segoe UI', sans-serif;
}
/* Subtle breadcrumb separator after title */
.cs-header-title::before {
    content: "EXPLORER  ›  ";
    font-size: 0.68rem;
    color: #858585;
    font-family: 'Consolas', monospace;
    letter-spacing: 0.06em;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Conversation cards — like open editor tabs / file list ── */
.cs-card {
    background: #252526;
    border: 1px solid #3e3e42;
    border-left: 3px solid transparent;
    border-radius: 2px;          /* VSCode uses sharp corners */
    padding: 0.75rem 1rem;
    margin-bottom: 0.4rem;
    transition: border-left-color 0.12s, background 0.12s;
    cursor: pointer;
}
.cs-card:hover {
    background: #2a2d2e;
    border-left-color: #007acc;
}
.cs-card-preview {
    font-size: 0.85rem;
    color: #cccccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
    font-family: 'Consolas', monospace;
}
.cs-card-date {
    font-size: 0.67rem;
    color: #858585;
    font-family: 'Consolas', monospace;
}

/* ── Section label — like VSCode explorer section headers ── */
.cs-section-label {
    font-size: 0.68rem;
    color: #bbbbbb;
    font-family: 'Consolas', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.6rem;
    margin-top: 0.4rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid #3e3e42;
}

/* ── Message bubbles ── */
.cs-msg-row {
    display: flex;
    margin-bottom: 1.1rem;
    animation: fadeUp 0.2s ease;
}
.cs-msg-row.user     { justify-content: flex-end; }
.cs-msg-row.assistant { justify-content: flex-start; }

.cs-bubble {
    max-width: 82%;
    padding: 0.7rem 1rem;
    font-size: 0.875rem;
    line-height: 1.65;
    overflow-wrap: break-word;
    min-width: 60px;
    border-radius: 2px;
}

/* User bubble — VSCode blue accent (like a selected line highlight) */
.cs-bubble.user {
    background: #04395e;
    color: #d4d4d4;
    border: 1px solid #007acc;
    border-bottom-right-radius: 0;
}

/* Assistant bubble — editor panel style */
.cs-bubble.assistant {
    background: #252526;
    border: 1px solid #3e3e42;
    border-left: 3px solid #007acc;
    color: #d4d4d4 !important;
    border-bottom-left-radius: 0;
}

.cs-bubble p { margin: 0 !important; padding: 0 !important; }
.cs-bubble ul, .cs-bubble ol { margin: 0.3rem 0 0 1.2rem !important; padding: 0 !important; }
.cs-bubble.assistant p,
.cs-bubble.assistant li,
.cs-bubble.assistant h1,
.cs-bubble.assistant h2,
.cs-bubble.assistant h3,
.cs-bubble.assistant strong,
.cs-bubble.assistant em {
    color: #d4d4d4 !important;
    background: transparent !important;
}

/* Inline code — VSCode token style */
.cs-bubble.assistant code {
    background: #1e1e1e !important;
    color: #ce9178 !important;   /* string orange */
    padding: 1px 5px;
    border-radius: 2px;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.82rem;
    border: 1px solid #3e3e42;
}

/* Keyword / bold in assistant — VSCode blue token */
.cs-bubble.assistant strong {
    color: #569cd6 !important;   /* keyword blue */
}

.cs-ts {
    font-size: 0.63rem;
    color: #555555;
    font-family: 'Consolas', monospace;
    margin-top: 3px;
    text-align: right;
}
.cs-msg-row.assistant .cs-ts { text-align: left; }

/* ── Typing indicator — three dots like VSCode loading spinners ── */
.cs-typing {
    display: flex;
    gap: 5px;
    padding: 0.7rem 1rem;
    background: #252526;
    border: 1px solid #3e3e42;
    border-left: 3px solid #007acc;
    border-radius: 2px;
    width: fit-content;
}
.cs-dot {
    width: 6px; height: 6px;
    background: #007acc;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.cs-dot:nth-child(2) { animation-delay: 0.2s; }
.cs-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Error banner — matches VSCode error squiggle style ── */
.cs-error {
    background: #1f1a1a;
    border: 1px solid #f44747;
    border-left: 3px solid #f44747;
    border-radius: 2px;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #f48771;
    margin-bottom: 1rem;
    font-family: 'Consolas', monospace;
}
.cs-error::before {
    content: "⛔  ";
}

/* ── Empty / info state ── */
.cs-empty {
    text-align: center;
    padding: 3rem 1rem;
}
.cs-empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.4; }
.cs-empty-text {
    font-size: 0.83rem;
    line-height: 1.8;
    color: #858585;
    font-family: 'Consolas', monospace;
}

/* ── Confirmation box ── */
.cs-confirm {
    background: #1f1a1a;
    border: 1px solid #f44747;
    border-left: 3px solid #f44747;
    border-radius: 2px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #f48771;
    margin-top: 0.5rem;
    font-family: 'Consolas', monospace;
}

/* ── Divider — like VSCode panel separator ── */
.cs-divider {
    border: none;
    border-top: 1px solid #3e3e42;
    margin: 1rem 0;
}

/* ── Fixed input bar — VSCode terminal / panel bottom strip ── */
.cs-input-bar-fixed {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #252526;
    border-top: 1px solid #3e3e42;
    padding: 0.75rem 1.5rem;
    z-index: 999;
    display: flex;
    align-items: center;
    gap: 10px;
    max-width: 1100px;
    margin: 0 auto;
}

/* ── Input overrides — VSCode terminal input style ── */
.stTextInput > div > div > input {
    background: #3c3c3c !important;
    border: 1px solid #3e3e42 !important;
    border-radius: 2px !important;
    color: #cccccc !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1rem !important;
    caret-color: #007acc;
}
.stTextInput > div > div > input:focus {
    border-color: #007acc !important;
    box-shadow: 0 0 0 1px #007acc !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: #555555 !important;
}

/* ── Button overrides — VSCode primary button ── */
.stButton > button {
    background: #0e639c !important;
    border: 1px solid #1177bb !important;
    border-radius: 2px !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 1rem !important;
    white-space: nowrap !important;
    width: 100% !important;
    transition: background 0.12s !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: #1177bb !important;
    border-color: #1177bb !important;
}
.stButton > button:active {
    background: #007acc !important;
}

/* ── Back / ghost button — VSCode secondary button ── */
button[kind="secondary"], .cs-back-btn button {
    background: transparent !important;
    border: 1px solid #3e3e42 !important;
    color: #cccccc !important;
    border-radius: 2px !important;
}
button[kind="secondary"]:hover, .cs-back-btn button:hover {
    background: #2a2d2e !important;
    border-color: #007acc !important;
    color: #ffffff !important;
}

/* ── Scrollbar — VSCode-style thin scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #1e1e1e; }
::-webkit-scrollbar-thumb { background: #424242; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #4f4f4f; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40%           { transform: translateY(-5px); }
}

/* ───────── Markdown rendering fixes ───────── */

.cs-md-bubble {
    overflow-x: auto;
}

.cs-md-bubble table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.75rem 0;
    font-size: 0.85rem;
}

.cs-md-bubble th,
.cs-md-bubble td {
    border: 1px solid #3e3e42;
    padding: 8px 12px;
    text-align: left;
}

.cs-md-bubble th {
    background: #2d2d2d;
    font-weight: 600;
}

.cs-md-bubble tr:nth-child(even) {
    background: #2a2d2a;
}

.cs-md-bubble pre {
    overflow-x: auto;
    padding: 12px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 2px;
}

.cs-md-bubble blockquote {
    border-left: 3px solid #007acc;
    margin: 0.75rem 0;
    padding-left: 12px;
    color: #bbbbbb;
}

.cs-md-bubble h1,
.cs-md-bubble h2,
.cs-md-bubble h3,
.cs-md-bubble h4,
.cs-md-bubble h5,
.cs-md-bubble h6 {
    margin-top: 0.8rem;
    margin-bottom: 0.5rem;
}

.cs-md-bubble p + table {
    margin-top: 0.75rem;
}

</style>
""",
    unsafe_allow_html=True,
)


#  Main


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
