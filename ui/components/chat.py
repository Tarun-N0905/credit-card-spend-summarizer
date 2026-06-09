"""
ui/components/chat.py

All rendering for the chat view — controls bar, message bubbles,
typing indicator, empty state, and the input bar.

Imported by:
  - ui/app.py
"""

import html
import os

import streamlit as st

from state import add_message, clear_error, go_to_list
from api_client import send_chat_message, delete_conversation

# Chat controls


def render_chat_controls() -> None:
    """
    Thin divider below header in chat view.
    Back button and Delete are now in the fixed input bar / list view respectively.
    """
    st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)


# Message renders


def render_message(role: str, content: str, timestamp: str) -> None:
    """
    Render a single chat bubble with correct alignment and styling.

    User messages are HTML-escaped (plain text input, no markdown expected).
    Assistant messages are rendered with st.markdown so bold, bullets, and
    other formatting from the LLM display correctly instead of showing raw **.

    Args:
        role      : "user" → right-aligned blue bubble
                    "assistant" → left-aligned dark bubble
        content   : message text to display inside the bubble
        timestamp : HH:MM string shown below the bubble
    """
    if role == "user":
        safe_content = html.escape(content)
        st.markdown(
            f"""<div class="cs-msg-row {role}">
                <div>
                    <div class="cs-bubble {role}">{safe_content}</div>
                    <div class="cs-ts">{timestamp}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="cs-msg-row assistant">
                <div>
                    <div class="cs-bubble assistant">
                        {content}
                    </div>
                    <div class="cs-ts">{timestamp}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_message_images(image_paths: list[str]) -> None:
    """
    Render images returned alongside a knowledge-base answer.

    Only paths that exist on disk are displayed — missing paths are
    skipped silently so a stale or moved file never causes an error.
    Images are shown below the text bubble using st.image(), which
    handles PNG/JPEG natively without any additional dependencies.

    Args:
        image_paths : List of absolute file-system paths from the agent.
    """
    for path in image_paths or []:
        if os.path.exists(path):
            st.image(path, use_container_width=True)


def render_typing_indicator() -> None:
    """
    Render an animated three-dot bounce indicator while is_loading is True,
    showing the user the assistant is processing their message.
    """
    st.markdown(
        """
    <div class="cs-msg-row assistant">
        <div class="cs-typing">
            <div class="cs-dot"></div>
            <div class="cs-dot"></div>
            <div class="cs-dot"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_empty_chat_state() -> None:
    """
    Render a placeholder prompt when no messages exist yet in a new
    conversation. Shown only when messages list is empty and not loading.
    """
    st.markdown(
        """
    <div class="cs-empty">
        <div class="cs-empty-icon">💬</div>
        <div class="cs-empty-text">
            Ask me about your credit card spend,<br>
            reward points, billing statements, or card benefits.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_conversation() -> None:
    """
    Iterate over all messages in st.session_state.messages and render
    each as a bubble. Appends the typing indicator when is_loading is True.
    Shows the empty chat state if no messages exist and not loading.

    For assistant messages that carry image_paths, images are rendered
    directly below the text bubble via render_message_images().
    """
    messages = st.session_state.messages

    if not messages and not st.session_state.is_loading:
        render_empty_chat_state()
        return

    for msg in messages:
        render_message(msg["role"], msg["content"], msg["timestamp"])

        # Render images only on assistant KB replies that returned paths
        if msg["role"] == "assistant" and msg.get("image_paths"):
            render_message_images(msg["image_paths"])

    if st.session_state.is_loading:
        render_typing_indicator()


#  Input bar


def render_input_bar() -> None:
    """
    Fixed bottom bar: [← Back] [text input (flex)] [Send]

    Delete confirmation (when triggered from list view open) is shown
    inline above the bar. Back button is left of the text input.

    Two-pass pattern unchanged for loading indicator.
    """
    if "input_reset_counter" not in st.session_state:
        st.session_state.input_reset_counter = 0

    # Confirmation state (triggered if confirm_delete is True)
    if st.session_state.confirm_delete:
        st.markdown(
            '<div class="cs-confirm">Are you sure you want to delete this chat? This cannot be undone.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([2, 2, 6])
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
        return

    col_back, col_input, col_send = st.columns([1, 7, 1])

    with col_back:
        if st.button("← Back", key="back_btn"):
            go_to_list()
            st.rerun()

    with col_input:
        user_input = st.text_input(
            label="message",
            label_visibility="collapsed",
            placeholder="Ask about your spend, rewards, or card benefits…",
            key=f"chat_input_{st.session_state.input_reset_counter}",
        )

    with col_send:
        send_clicked = st.button("Send", key="send_btn")

    if send_clicked and user_input.strip():
        clear_error()
        add_message("user", user_input.strip())
        st.session_state.is_loading = True
        st.session_state.input_reset_counter += 1
        st.rerun()

    if st.session_state.is_loading:
        reply, image_paths = send_chat_message(st.session_state.messages[-1]["content"])
        st.session_state.is_loading = False
        if reply:
            add_message("assistant", reply, image_paths=image_paths)
        st.rerun()
