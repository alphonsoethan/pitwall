# config.py
import os
import streamlit as st
import fastf1

RACE_MAP = {
    "São Paulo 2024": {"year": 2024, "race": "São Paulo", "laps": 71},
}

DRIVER_CHOICES = [
    "VER", "HAM", "LEC", "NOR", "PIA", "RUS",
    "SAI", "PER", "TSU", "ALO", "OCO", "GAS",
]

CACHE_DIR = "f1_cache"


def f1_credentials():
    try:
        return st.secrets["F1_USERNAME"], st.secrets["F1_PASSWORD"]
    except Exception:
        return os.environ.get("F1_USERNAME"), os.environ.get("F1_PASSWORD")


@st.cache_data(show_spinner="Loading race data...")
def get_shared_session():
    """One cached session, shared by live and recap."""
    user, pw = f1_credentials()
    if user and pw:
        os.environ["F1_USERNAME"] = user
        os.environ["F1_PASSWORD"] = pw

    race_info = RACE_MAP["São Paulo 2024"]
    session = fastf1.get_session(race_info["year"], race_info["race"], "R")
    session.load(telemetry=False, weather=True, messages=False)

    if session.laps is None or len(session.laps) == 0:
        raise RuntimeError(
            "F1 timing data came back empty. Check F1_USERNAME / F1_PASSWORD "
            "in Streamlit secrets, or try again in a minute."
        )
    return session