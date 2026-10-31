import os
import streamlit as st
import fastf1

# Automatically create the cache folder if it doesn't exist
os.makedirs('cache', exist_ok=True)

# Enable the cache
fastf1.Cache.enable_cache('cache')

st.title("PitWall Test")

session = fastf1.get_session(2024, "Monza", "R")
session.load(telemetry=False, weather=False, messages=False)

st.write("Laps loaded:", len(session.laps))
st.dataframe(session.laps[["Driver", "LapNumber", "LapTime", "Compound"]].head(20))