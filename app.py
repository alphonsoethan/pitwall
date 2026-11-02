# app.py
import os
import streamlit as st
import fastf1

from config import CACHE_DIR

# ---------- FastF1 cache: enabled ONCE, before any session load ----------
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.configure(CACHE_DIR)

st.set_page_config(page_title="PitWall", page_icon="🏎️", layout="wide")

# ---------- Session state defaults ----------
for key, default in [
    ("page", "home"),
    ("live_lap", 0),
    ("strategy_text", None),
    ("recap_text", None),
    ("verdict_text", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Router ----------
from views import home, live, recap

page = st.session_state.page
if page == "home":
    home.render()
elif page == "live":
    live.render()
elif page == "recap_setup":
    recap.render_setup()
elif page == "recap":
    recap.render_display()
else:
    st.error(f"Unknown page: {page}")