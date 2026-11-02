# views/home.py
import streamlit as st


def render():
    st.title("🏎️ PitWall")
    st.caption("Your F1 race companion")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📡 Livestream Mode", use_container_width=True, type="primary"):
            st.session_state.page = "live"
            st.session_state.live_lap = 1
            st.session_state.strategy_text = None
            st.rerun()
    with col2:
        if st.button("🎬 Race Recap (what you missed)", use_container_width=True):
            st.session_state.page = "recap_setup"
            st.session_state.recap_text = None
            st.session_state.verdict_text = None
            st.rerun()