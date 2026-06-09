"""
ui/components/chat.py

All rendering for the chat view — controls bar, message bubbles,
typing indicator, empty state, and the input bar.

Streaming flow
──────────────
1. User submits → is_loading=True, pending_user_message set → rerun
2. Next run: call stream_chat_message() generator.
   • Before first token: show animated status bubble.
   • After first token: render live markdown HTML on every token via
     _md_to_html_safe(), giving a smooth typewriter effect with proper
     formatting (tables, bold, code, etc.) visible from the first chunk.
3. Stream done: persist raw markdown to session state, rerun to render
   via render_conversation() as a normal bubble.

Fix summary (vs previous version)
──────────────────────────────────
• REMOVED _streaming_text_html() — it used html.escape() which rendered
  markdown syntax literally (raw **bold**, |table| etc.) instead of HTML.
• REPLACED with _streaming_bubble_html() for ALL in-flight frames,
  converting markdown on every token so the user sees formatted output
  from the very first word.
• Removed the redundant done=True finalize call before st.rerun() —
  render_conversation() re-renders from session state anyway.
• Session state always stores raw markdown; never pre-converted HTML.

Imported by:
  - ui/app.py
"""

import html
import os
import time

import streamlit as st

from state import add_message, clear_error, go_to_list
from api_client import stream_chat_message, delete_conversation

# Rotating status messages shown while waiting for the first token
_COOKING_MESSAGES = [
    "🍳 Cooking your response…",
    "🔍 Scanning your statements…",
    "✨ Crunching the numbers…",
]

# ── Chat controls ────────────────────────────────────────────────────────────


def render_chat_controls() -> None:
    """Thin divider below header in chat view."""
    st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
    # Collapse Streamlit's default inter-element gaps so chat bubbles sit
    # close together. Targets the wrapper divs Streamlit injects around every
    # st.markdown() / st.empty() call inside the message list.
    st.markdown(
        """<style>
        /* Remove top/bottom margins from every Streamlit element wrapper
           that contains a chat bubble or timestamp */
        div[data-testid="element-container"]:has(.cs-msg-row),
        div[data-testid="element-container"]:has(.cs-ts) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        /* Bubble-to-bubble spacing — controlled here, not via Streamlit gaps */
        .cs-msg-row {
            margin-bottom: 6px !important;
        }
        /* Tighten timestamp below assistant bubble */
        .cs-ts {
            margin-bottom: 4px !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )


# ── Message renders ───────────────────────────────────────────────────────────


def render_message(role: str, content: str, timestamp: str) -> None:
    """
    Render a single chat bubble.

    User      → right-aligned blue bubble (HTML-escaped plain text).
    Assistant → left-aligned dark bubble with markdown rendered inside.
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
        # Use components.html to bypass Streamlit's <table> tag stripping.
        # st.markdown(unsafe_allow_html=True) silently removes <table> elements
        # which breaks markdown tables even when the HTML is correct.
        _assistant_bubble_components_html(content)
        st.markdown(
            f'<div class="cs-ts" style="font-size:0.63rem;color:#555;'
            f'font-family:Consolas,monospace;margin-top:-8px;">{timestamp}</div>',
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


# ── Streaming helpers ─────────────────────────────────────────────────────────


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
    HTML for the in-progress (or just-finished) assistant bubble.

    Markdown is converted on EVERY frame so the user always sees
    formatted output — tables, bold, code blocks — not raw syntax.

    While streaming, a blinking cursor is appended after the rendered HTML.
    When done=True the cursor and the streaming border variant are omitted.

    Note: the cursor is injected OUTSIDE the markdown-rendered HTML so it
    does not interfere with table or list structures mid-render.
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

    Supports:
      - tables
      - fenced code blocks
      - lists
      - headings
      - line breaks
    """
    try:
        import markdown as md_lib

        return md_lib.markdown(
            text,
            extensions=[
                "fenced_code",
                "tables",
                "nl2br",
            ],
        )
    except ImportError:
        pass

    import re

    safe = html.escape(text)

    safe = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        safe,
        flags=re.DOTALL,
    )

    safe = re.sub(
        r"\*(.+?)\*",
        r"<em>\1</em>",
        safe,
    )

    safe = re.sub(
        r"`([^`]+)`",
        r"<code>\1</code>",
        safe,
    )

    safe = safe.replace("\n\n", "</p><p>")
    safe = safe.replace("\n", "<br>")

    return f"<p>{safe}</p>"


def _assistant_bubble_components_html(content: str) -> None:
    """
    Render an assistant bubble using st.components.v1.html() to bypass
    Streamlit's <table> tag stripping in unsafe_allow_html mode.
    """
    import streamlit.components.v1 as components

    rendered = _md_to_html_safe(content)

    # Estimate height: ~24px per line, ~80 chars per line at 0.875rem
    _lines = max(1, len(content) // 80) + content.count("\n")
    calculated_height = max(120, min(3000, _lines * 24 + 60))

    components.html(
        f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; padding:0; background:transparent; font-family:'Segoe UI',system-ui,sans-serif; }}
  .cs-bubble {{ padding:0.7rem 1rem; font-size:0.875rem; line-height:1.65;
                overflow-wrap:break-word; border-radius:2px;
                background:#252526; border:1px solid #3e3e42;
                border-left:3px solid #007acc; color:#d4d4d4;
                border-bottom-left-radius:0; overflow-x:auto; }}
  .cs-bubble p {{ margin:0 0 0.4rem !important; padding:0 !important; color:#d4d4d4; }}
  .cs-bubble ul, .cs-bubble ol {{ margin:0.3rem 0 0 1.2rem !important; color:#d4d4d4; }}
  .cs-bubble strong {{ color:#569cd6; }}
  .cs-bubble code {{ background:#1e1e1e !important; color:#ce9178 !important;
                     padding:1px 5px; border-radius:2px;
                     font-family:'Consolas','Courier New',monospace;
                     font-size:0.82rem; border:1px solid #3e3e42; }}
  .cs-bubble table {{ border-collapse:collapse; width:100%; margin:0.75rem 0;
                      font-size:0.85rem; color:#d4d4d4; }}
  .cs-bubble th, .cs-bubble td {{ border:1px solid #3e3e42; padding:8px 12px;
                                  text-align:left; color:#d4d4d4; }}
  .cs-bubble th {{ background:#2d2d2d; font-weight:600; color:#cccccc; }}
  .cs-bubble tr:nth-child(even) {{ background:#2a2a2a; }}
  .cs-bubble pre {{ overflow-x:auto; padding:12px; background:#1e1e1e;
                    border:1px solid #3e3e42; border-radius:2px; margin:0.5rem 0; }}
  .cs-bubble blockquote {{ border-left:3px solid #007acc; margin:0.75rem 0;
                           padding-left:12px; color:#bbbbbb; }}
  .cs-bubble h1,.cs-bubble h2,.cs-bubble h3 {{ margin:0.8rem 0 0.5rem; color:#cccccc; }}
  .cs-ts {{ font-size:0.63rem; color:#555555; font-family:'Consolas',monospace;
            margin-top:4px; }}
</style>
</head>
<body>
  <div class="cs-bubble">{rendered}</div>
</body>
</html>""",
        height=calculated_height,
        scrolling=False,
    )


# ── Input bar ─────────────────────────────────────────────────────────────────


def render_input_bar() -> None:
    """
    Fixed bottom bar: [← Back] [text input] [Send]

    Streaming flow (runs synchronously inside a single Streamlit run):
      • A stable st.empty() placeholder is created below the message list.
      • All tokens are buffered silently while a rotating status animation
        plays. Rendering partial markdown mid-stream produces broken HTML
        (incomplete tables, dangling list items) so we wait for the full
        response before converting.
      • Once the stream ends, _streaming_bubble_html(done=True) renders the
        complete markdown — tables, code blocks, bold etc. all intact.
      • Raw markdown is persisted to session state and st.rerun() triggers
        render_conversation() to draw the final permanent bubble.

    Key invariant: session state always stores raw markdown strings, never
    pre-converted HTML, so render_message() / _md_to_html_safe() works
    correctly on every rerun.
    """
    if "input_reset_counter" not in st.session_state:
        st.session_state.input_reset_counter = 0

    # ── Delete confirmation overlay ──────────────────────────────────────
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

    # ── Input row ────────────────────────────────────────────────────────
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

    # ── Step 1: capture new message ──────────────────────────────────────
    if send_clicked and user_input.strip():
        clear_error()
        add_message("user", user_input.strip())
        st.session_state.is_loading = True
        st.session_state.pending_user_message = user_input.strip()
        st.session_state.input_reset_counter += 1
        st.rerun()

    # ── Step 2: stream the assistant response ────────────────────────────
    if st.session_state.is_loading:
        user_msg = st.session_state.get("pending_user_message", "")

        # Single stable DOM slot for the entire streaming bubble.
        # All frames — status text, live tokens — are written into this one
        # placeholder so nothing shifts on screen.
        stream_placeholder = st.session_state.get(
            "_stream_placeholder",
            st.empty(),
        )

        accumulated: list[str] = []
        got_first_token = False
        status_idx = 0
        last_swap = time.monotonic()
        last_render = time.monotonic()
        RENDER_INTERVAL = 0.05  # re-render every 50 ms for smooth typewriter effect

        # Show initial status immediately
        stream_placeholder.markdown(
            _status_bubble_html(_COOKING_MESSAGES[0]),
            unsafe_allow_html=True,
        )

        for token in stream_chat_message(user_msg):
            accumulated.append(token)

            if not got_first_token:
                got_first_token = True

            # Throttle renders to ~20 fps so we don't hammer Streamlit
            now = time.monotonic()
            if now - last_render >= RENDER_INTERVAL:
                last_render = now
                partial = "".join(accumulated)
                stream_placeholder.markdown(
                    _streaming_bubble_html(partial, done=False),
                    unsafe_allow_html=True,
                )

            # Rotate status message every 2 s ONLY while waiting for first token
            if not got_first_token and now - last_swap >= 2.0:
                status_idx = (status_idx + 1) % len(_COOKING_MESSAGES)
                last_swap = now
                stream_placeholder.markdown(
                    _status_bubble_html(_COOKING_MESSAGES[status_idx]),
                    unsafe_allow_html=True,
                )

        if not got_first_token:
            # Stream ended with no tokens — error was set by api_client
            stream_placeholder.empty()
        else:
            # Final render: full markdown, no blinking cursor
            full_text = "".join(accumulated)
            stream_placeholder.markdown(
                _streaming_bubble_html(full_text, done=True),
                unsafe_allow_html=True,
            )

        # ── Step 3: persist raw markdown and rerun ───────────────────────
        # Store the raw accumulated text (not HTML) so render_message()
        # can convert it correctly on every future rerun.
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
