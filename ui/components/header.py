"""
ui/components/header.py

Persistent UI elements shown in both list and chat views.

Imported by:
  - ui/app.py
"""

import streamlit as st

from state import clear_error


def render_header() -> None:
    """
    Render the top header bar. Always visible in both list and chat views.
    Title is centered; no session sub shown.
    """
    st.markdown(
        f"""
    <div class="cs-header">
        <div class="cs-header-icon">💳</div>
        <div>
            <div class="cs-header-title">Credit Spend Summarizer</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


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
