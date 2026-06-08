"""
ui/components/chat.py
"""

import streamlit as st

from state import add_message, clear_error, go_to_list
from api_client import send_chat_message, delete_conversation


def render_chat_controls() -> None:
    if st.session_state.confirm_delete:
        st.markdown(
            '<div class="cs-confirm">Are you sure? This cannot be undone.</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, Delete", key="confirm_yes_btn"):
                success = delete_conversation(st.session_state.session_id)
                if success:
                    go_to_list()
                st.rerun()
        with c2:
            if st.button("Cancel", key="confirm_cancel_btn"):
                st.session_state.confirm_delete = False
                st.rerun()
        st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)


def render_message(role: str, content: str, timestamp: str) -> None:
    st.markdown(f"""
    <div class="cs-msg-row {role}">
        <div>
            <div class="cs-bubble {role}">{content}</div>
            <div class="cs-ts">{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_typing_indicator() -> None:
    st.markdown("""
    <div class="cs-msg-row assistant">
        <div class="cs-typing">
            <div class="cs-dot"></div>
            <div class="cs-dot"></div>
            <div class="cs-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_chat_state() -> None:
    st.markdown("""
    <div class="cs-empty">
        <div class="cs-empty-icon">💬</div>
        <div class="cs-empty-text">
            Ask me about your credit card spend,<br>
            reward points, billing statements, or card benefits.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_conversation() -> None:
    messages = st.session_state.messages
    if not messages and not st.session_state.is_loading:
        render_empty_chat_state()
        return
    for msg in messages:
        render_message(msg["role"], msg["content"], msg["timestamp"])
    if st.session_state.is_loading:
        render_typing_indicator()


def render_input_bar() -> None:
    if "input_reset_counter" not in st.session_state:
        st.session_state.input_reset_counter = 0

    input_key = f"chat_input_{st.session_state.input_reset_counter}"

    # Fix the input row to the bottom of the screen
    st.markdown(f"""
    <style>
    div[data-testid="stHorizontalBlock"]:has(input[aria-label="message"]) {{
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1000 !important;
        background: #0d0f14 !important;
        border-top: 1px solid #1e2130 !important;
        padding: 0.75rem 2rem !important;
        margin: 0 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Layout: [text input  |  Send  |  ← Back]
    col_input, col_send, col_back = st.columns([6, 1, 1])

    with col_input:
        user_input = st.text_input(
            label="message",
            label_visibility="collapsed",
            placeholder="Ask about your spend, rewards, or card benefits…",
            key=input_key,
        )

    with col_send:
        send_clicked = st.button("Send", key="send_btn")

    with col_back:
        if st.button("← Back", key="input_back_btn"):
            go_to_list()
            st.rerun()

    if send_clicked and user_input.strip():
        clear_error()
        add_message("user", user_input.strip())
        st.session_state.is_loading = True
        st.session_state.input_reset_counter += 1
        st.rerun()

    if st.session_state.is_loading:
        reply = send_chat_message(st.session_state.messages[-1]["content"])
        st.session_state.is_loading = False
        if reply:
            add_message("assistant", reply)
        st.rerun()
