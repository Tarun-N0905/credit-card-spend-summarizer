"""
ui/components/header.py
"""

import streamlit as st
from state import clear_error


def render_header() -> None:

    if st.session_state.view == "list":
        st.markdown("""
        <div class="cs-header-fixed">
            <div class="cs-header-title">💳 Credit Card Summarizer</div>
            <div class="cs-header-sub">conversations</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Fixed title bar (pure HTML, always visible) — no Back/Delete here
        st.markdown("""
        <div class="cs-header-fixed">
            <div class="cs-header-title">💳 Credit Card Summarizer</div>
        </div>
        """, unsafe_allow_html=True)


def render_error_banner() -> None:
    if st.session_state.error:
        st.markdown(
            f'<div class="cs-error">{st.session_state.error}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Dismiss", key="dismiss_error"):
            clear_error()
            st.rerun()
