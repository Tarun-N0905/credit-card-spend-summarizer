import html
import os
import re
import time

import streamlit as st

from state import add_message, clear_error, go_to_list
from api_client import stream_chat_message, delete_conversation

# Rotating status messages shown while waiting for the first token
_COOKING_MESSAGES = [
    "🔍 Scanning your statements…",
    "✨ Crunching the numbers…",
    "🍳 Cooking your response…",
]

#  Chat controls


def render_chat_controls() -> None:
    """Thin divider below header in chat view."""
    st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
    st.markdown(
        """<style>
        /* ── Zero Streamlit's flex gap on the chat message column ── */
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .cs-msg-row),
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .cs-assistant-wrap) {
            gap: 0 !important;
        }

        div[data-testid="element-container"]:has(.cs-msg-row),
        div[data-testid="element-container"]:has(.cs-assistant-wrap) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        .cs-msg-row {
            margin-bottom: 6px !important;
        }
        .cs-ts {
            margin-bottom: 4px !important;
        }

        /* ── Assistant bubble ── */
        .cs-assistant-wrap {
            max-width: 82%;
            margin-bottom: 6px;
        }
        .cs-assistant-bubble {
            background: #252526;
            border: 1px solid #3e3e42;
            border-left: 3px solid #007acc;
            border-radius: 2px;
            border-bottom-left-radius: 0;
            padding: 0.7rem 1rem;
            color: #d4d4d4;
            font-size: 0.875rem;
            line-height: 1.65;
            overflow-wrap: break-word;
        }
        /* Typography inside the bubble */
        .cs-assistant-bubble p  { margin: 0 0 0.4rem !important; color: #d4d4d4 !important; }
        /* Remove bottom margin on the last paragraph — avoids dead space at bubble bottom */
        .cs-assistant-bubble p:last-child { margin-bottom: 0 !important; }
        .cs-assistant-bubble ul,
        .cs-assistant-bubble ol { margin: 0.3rem 0 0 1.2rem !important; color: #d4d4d4 !important; }
        .cs-assistant-bubble li { color: #d4d4d4 !important; }
        .cs-assistant-bubble li:last-child { margin-bottom: 0 !important; }
        .cs-assistant-bubble strong { color: #569cd6 !important; }
        .cs-assistant-bubble em    { color: #d4d4d4 !important; }
        .cs-assistant-bubble h1,
        .cs-assistant-bubble h2,
        .cs-assistant-bubble h3 { margin: 0.8rem 0 0.5rem; color: #cccccc !important; }
        .cs-assistant-bubble code {
            background: #1e1e1e !important;
            color: #ce9178 !important;
            padding: 1px 5px;
            border-radius: 2px;
            font-family: 'Consolas','Courier New',monospace !important;
            font-size: 0.82rem;
            border: 1px solid #3e3e42;
        }
        .cs-assistant-bubble pre {
            overflow-x: auto;
            padding: 12px;
            background: #1e1e1e !important;
            border: 1px solid #3e3e42;
            border-radius: 2px;
            margin: 0.5rem 0 0 0;
        }
        .cs-assistant-bubble pre:last-child { margin-bottom: 0 !important; }
        .cs-assistant-bubble pre code {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
        }
        .cs-assistant-bubble table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 0.75rem 0 0 0 !important;
            font-size: 0.85rem !important;
            color: #d4d4d4 !important;
        }
        .cs-assistant-bubble table:last-child { margin-bottom: 0 !important; }
        .cs-assistant-bubble th,
        .cs-assistant-bubble td {
            border: 1px solid #3e3e42 !important;
            padding: 8px 12px !important;
            text-align: left !important;
            color: #d4d4d4 !important;
        }
        .cs-assistant-bubble th {
            background: #2d2d2d !important;
            font-weight: 600 !important;
            color: #cccccc !important;
        }
        .cs-assistant-bubble tr:nth-child(even) { background: #2a2a2a !important; }
        .cs-assistant-bubble blockquote {
            border-left: 3px solid #007acc;
            margin: 0.75rem 0 0 0;
            padding-left: 12px;
            color: #bbbbbb;
        }
        .cs-assistant-bubble blockquote:last-child { margin-bottom: 0 !important; }
        </style>""",
        unsafe_allow_html=True,
    )


#  Message renders


def render_message(role: str, content: str, timestamp: str) -> None:
    """
    Render a single chat bubble.

    User      → right-aligned blue bubble via HTML (safe escaped plain text).
    Assistant → left-aligned bubble via _render_assistant_bubble().
    """
    if role == "user":
        safe_content = html.escape(content)
        st.markdown(
            f"""<div class="cs-msg-row user">
                <div>
                    <div class="cs-bubble user">{safe_content}</div>
                    <div class="cs-ts">{timestamp}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        _render_assistant_bubble(content, timestamp)


def _render_assistant_bubble(content: str, timestamp: str) -> None:
    """
    Render an assistant message in a SINGLE st.markdown() call.

    Three sources of extra space are eliminated here:

    1. Streamlit gap   — the parent stVerticalBlock uses CSS `gap` (not margins)
                         to space element-containers. Gap is zeroed via the
                         :has() rule in render_chat_controls().

    2. p:last-child    — the markdown library wraps every response in <p> tags.
                         The generic rule gives each <p> margin-bottom:0.4rem,
                         which would add dead space at the bubble bottom.
                         A p:last-child { margin-bottom:0 } rule cancels it.

    3. Trailing <br>   — nl2br converts every trailing newline in the content
                         into a <br> inside the last <p>, adding an extra line
                         of height. content.strip() + _strip_trailing_br()
                         remove these before rendering.
    """
    clean = _md_to_html_safe(content.strip())
    st.markdown(
        f'<div class="cs-assistant-wrap">'
        f'<div class="cs-assistant-bubble cs-md-bubble">{clean}</div>'
        f"</div>"
        f'<div class="cs-ts" style="font-size:0.63rem;color:#555555;'
        f'font-family:Consolas,monospace;margin-top:2px;">{timestamp}</div>',
        unsafe_allow_html=True,
    )


def render_message_images(image_paths: list[str]) -> None:
    """Render images attached to an assistant message."""
    for path in image_paths or []:
        if os.path.exists(path):
            st.image(path, use_container_width=True)


def render_typing_indicator() -> None:
    """Three-dot bounce indicator (legacy fallback, not used during streaming)."""
    st.markdown(
        """<div class="cs-msg-row assistant">
            <div class="cs-typing">
                <div class="cs-dot"></div>
                <div class="cs-dot"></div>
                <div class="cs-dot"></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_empty_chat_state() -> None:
    """Empty-conversation placeholder."""
    st.markdown(
        """<div class="cs-empty">
            <div class="cs-empty-icon">💬</div>
            <div class="cs-empty-text">
                Ask me about your credit card spend,<br>
                reward points, billing statements, or card benefits.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_conversation() -> None:
    """
    Render all messages in session state.
    Shows empty state if no messages, typing indicator if loading.
    """
    messages = st.session_state.messages

    if not messages and not st.session_state.is_loading:
        render_empty_chat_state()
        return

    for msg in messages:
        render_message(msg["role"], msg["content"], msg["timestamp"])
        if msg["role"] == "assistant" and msg.get("image_paths"):
            render_message_images(msg["image_paths"])

    if st.session_state.is_loading:
        st.session_state._stream_placeholder = st.empty()


#  Streaming helpers


def _status_bubble_html(status_text: str) -> str:
    """HTML for the pre-token status bubble with animated dots."""
    return f"""<div class="cs-msg-row assistant">
        <div style="max-width:82%">
            <div class="cs-bubble assistant">
                <div class="cs-stream-status">{status_text}</div>
                <div class="cs-typing cs-typing-inline">
                    <div class="cs-dot"></div>
                    <div class="cs-dot"></div>
                    <div class="cs-dot"></div>
                </div>
            </div>
        </div>
    </div>"""


def _streaming_bubble_html(text: str, done: bool = False) -> str:
    """
    HTML for the in-progress (or just-finished) streaming bubble.
    After streaming ends st.rerun() hands off to _render_assistant_bubble().
    """
    rendered = _md_to_html_safe(text)
    cursor = "" if done else '<span class="cs-cursor">▍</span>'
    extra_cls = "" if done else " cs-streaming"
    return f"""<div class="cs-msg-row assistant">
        <div style="max-width:82%">
            <div class="cs-bubble assistant cs-md-bubble{extra_cls}">
                {rendered}{cursor}
            </div>
        </div>
    </div>"""


def _md_to_html_safe(text: str) -> str:
    """
    Convert markdown → HTML.

    Strips trailing <br> tags that nl2br injects when content ends with
    newlines — these add a blank line of height inside the bubble.
    """
    try:
        import markdown as md_lib

        rendered = md_lib.markdown(
            text,
            extensions=["fenced_code", "tables", "nl2br"],
        )
        # Remove trailing <br> / <br /> that nl2br appends before closing </p>
        rendered = re.sub(r"(<br\s*/?>)+(\s*</p>)", r"\2", rendered)
        return rendered
    except ImportError:
        pass

    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe, flags=re.DOTALL)
    safe = re.sub(r"\*(.+?)\*", r"<em>\1</em>", safe)
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    safe = safe.replace("\n\n", "</p><p>")
    safe = safe.replace("\n", "<br>")
    return f"<p>{safe}</p>"


#  Input bar


def render_input_bar() -> None:
    """
    Fixed bottom bar: [← Back] [text input] [Send]

    Streaming flow:
      1. User hits Send → message saved, is_loading=True, rerun.
      2. On rerun render_conversation() creates an st.empty() placeholder.
      3. render_input_bar() sees is_loading=True → streams tokens into the
         placeholder using _streaming_bubble_html() (HTML, transient).
      4. Stream ends → raw markdown saved to session state, rerun.
      5. render_conversation() now calls _render_assistant_bubble() which
         uses a single st.markdown() call — no gaps, no height issues.
    """
    if "input_reset_counter" not in st.session_state:
        st.session_state.input_reset_counter = 0

    #  Delete confirmation overlay
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

    #  Input row
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

    #  Step 1: capture new message
    if send_clicked and user_input.strip():
        clear_error()
        add_message("user", user_input.strip())
        st.session_state.is_loading = True
        st.session_state.pending_user_message = user_input.strip()
        st.session_state.input_reset_counter += 1
        st.rerun()

    #  Step 2: stream the assistant response
    if st.session_state.is_loading:
        user_msg = st.session_state.get("pending_user_message", "")

        stream_placeholder = st.session_state.get(
            "_stream_placeholder",
            st.empty(),
        )

        accumulated: list[str] = []
        got_first_token = False
        status_idx = 0
        last_swap = time.monotonic()
        last_render = time.monotonic()
        RENDER_INTERVAL = 0.05

        stream_placeholder.markdown(
            _status_bubble_html(_COOKING_MESSAGES[0]),
            unsafe_allow_html=True,
        )

        for token in stream_chat_message(user_msg):
            now = time.monotonic()

            # Rotate status bubble until first token arrives
            if not got_first_token:
                if now - last_swap >= 2.0:
                    status_idx = (status_idx + 1) % len(_COOKING_MESSAGES)
                    last_swap = now
                    stream_placeholder.markdown(
                        _status_bubble_html(_COOKING_MESSAGES[status_idx]),
                        unsafe_allow_html=True,
                    )

            accumulated.append(token)
            got_first_token = True

            if now - last_render >= RENDER_INTERVAL:
                last_render = now
                partial = "".join(accumulated)
                stream_placeholder.markdown(
                    _streaming_bubble_html(partial, done=False),
                    unsafe_allow_html=True,
                )

        if not got_first_token:
            stream_placeholder.empty()
        else:
            full_text = "".join(accumulated)
            stream_placeholder.markdown(
                _streaming_bubble_html(full_text, done=True),
                unsafe_allow_html=True,
            )

        #  Step 3: persist and rerun
        full_reply = "".join(accumulated).strip()
        image_paths = st.session_state.get("stream_image_paths", [])

        st.session_state.is_loading = False
        st.session_state.pending_user_message = ""
        st.session_state.stream_image_paths = []

        if full_reply:
            add_message("assistant", full_reply, image_paths=image_paths)
        elif not st.session_state.get("error"):
            st.session_state.error = "⚠ No response received. Please try again."

        st.rerun()
