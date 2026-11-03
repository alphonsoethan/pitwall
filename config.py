# config.py
import os
import streamlit as st
import fastf1

from data_store import load_session

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


@st.cache_resource(show_spinner="Loading race data...")
def get_shared_session():
    return load_session()  
   