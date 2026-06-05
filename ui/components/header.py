"""
ui/components/header.py

Persistent UI elements shown in both list and chat views.

Imported by:
  - ui/app.py
"""

import streamlit as st

from ui.state import clear_error


def render_header() -> None:
    """
    Render the top header bar. Always visible in both list and chat views.

    In list view: subtitle shows "conversations"
    In chat view: subtitle shows first 8 chars of the active session UUID
    """
    if st.session_state.view == "list":
        sub = "conversations"
    else:
        sid = st.session_state.session_id or ""
        sub = sid[:8] + "…" if sid else ""

    st.markdown(f"""
    <div class="cs-header">
        <div class="cs-header-icon">💳</div>
        <div>
            <div class="cs-header-title">Credit Spend Summarizer</div>
            <div class="cs-header-sub">{sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_error_banner() -> None:
    """
    Display a styled dismissible error banner when st.session_state.error
    is set. Dismiss button clears the error and reruns.
    Shown in both list and chat views.
    """
    if st.session_state.error:
        st.markdown(
            f'<div class="cs-error">{st.session_state.error}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Dismiss", key="dismiss_error"):
            clear_error()
            st.rerun()
