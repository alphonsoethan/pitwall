# views/recap.py
import streamlit as st
import pandas as pd
import fastf1

from config import RACE_MAP, DRIVER_CHOICES, get_shared_session
from gemini_client import get_recap, get_verdict, get_race_recap, get_race_verdict


# def _f1_credentials():
#     try:
#         return st.secrets["F1_USERNAME"], st.secrets["F1_PASSWORD"]
#     except Exception:
#         return os.environ.get("F1_USERNAME"), os.environ.get("F1_PASSWORD")


# @st.cache_data(show_spinner="Loading race data...")
# def _load_session(year, race):
    user, pw = _f1_credentials()
    if user and pw:
        os.environ["F1_USERNAME"] = user
        os.environ["F1_PASSWORD"] = pw

    session = fastf1.get_session(year, race, "R")
    session.load(telemetry=False, weather=False, messages=False)

    if session.laps is None or len(session.laps) == 0:
        raise RuntimeError(
            "F1 timing data came back empty. Check F1_USERNAME / F1_PASSWORD "
            "in Streamlit secrets, or try again in a minute."
        )
    return session


def render_setup():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🎬 Race Recap")
    st.caption("São Paulo 2024 · Fixed for this prototype")
    st.write("Pick a driver and the lap you left at. Following a specific driver is optional.")

    max_laps = int(RACE_MAP["São Paulo 2024"]["laps"])

    driver = st.selectbox(
        "Driver to focus on (optional)",
        ["Whole race (no specific driver)"] + DRIVER_CHOICES,
    )
    from_lap = st.slider(
        "I left at lap (0 = missed the entire race)",
        min_value=0,
        max_value=max_laps,
        value=0,
        help=f"São Paulo 2024 has {max_laps} laps. Set to 0 to recap the entire race.",
    )

    if st.button("Generate Recap →", type="primary"):
        st.session_state.recap_race_label = "São Paulo 2024"
        st.session_state.recap_driver = driver
        st.session_state.recap_from_lap = from_lap
        st.session_state.recap_text = None
        st.session_state.verdict_text = None
        st.session_state.page = "recap"
        st.rerun()
        

def _render_classification_table(session):
    """Final classification: position, driver, team, points."""
    res = session.results.copy()
    if "Abbreviation" in res.columns and "Driver" not in res.columns:
        res = res.rename(columns={"Abbreviation": "Driver"})
    if "TeamName" not in res.columns:
        res["TeamName"] = ""
    cols_needed = ["Position", "Driver", "FullName", "TeamName", "Points"]
    res = res[[c for c in cols_needed if c in res.columns]].dropna(subset=["Position"])
    res = res.sort_values("Position")

    display = pd.DataFrame({
        "P": res["Position"].astype(int).values,
        "Driver": res["FullName"].values if "FullName" in res.columns else res["Driver"].values,
        "Team": res["TeamName"].values if "TeamName" in res.columns else [""] * len(res),
        "Points": res["Points"].fillna(0).astype(int).values if "Points" in res.columns else [0] * len(res),
    })
    st.dataframe(display, hide_index=True, use_container_width=True, height=340)



def render_display():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.session_state.recap_text = None
        st.session_state.verdict_text = None
        st.rerun()

    race_label = st.session_state.recap_race_label
    from_lap = st.session_state.recap_from_lap
    driver_choice = st.session_state.recap_driver
    race_info = RACE_MAP[race_label]

    focus_label = (
        "Whole race" if driver_choice == "Whole race (no specific driver)" else driver_choice
    )
    lap_label = "the full race" if from_lap == 0 else f"lap {from_lap}"

    st.title(f"Recap: {focus_label} · {race_label}")
    st.caption(f"You left at {lap_label}")

    session = get_shared_session()
    # --- Top row: highlights box (left) | classification + standings (right) ---
    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.subheader("Continue Watching")
        st.markdown(
            f"""
            <div style="max-width:640px;margin:0 auto;">
              <div style="aspect-ratio:16/9;width:100%;background-color:#1a1a2e;
                   border:2px dashed #e94560;border-radius:10px;
                   display:flex;flex-direction:column;align-items:center;justify-content:center;
                   color:#e94560;text-align:center;padding:16px;box-sizing:border-box;">
                <div style="font-size:20px;">🎞️ EDITED HIGHLIGHTS</div>
                <div style="font-size:13px;color:#bbb;margin-top:8px;max-width:480px;">
                  Edited highlights of {race_label} from {lap_label} onwards will be displayed here.<br>
                  The clip will pick up right where you left off.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"▶ Will resume from {lap_label} of {race_label}.")

    with right_col:
        st.subheader("Final classification")
        _render_classification_table(session)


    st.divider()

    # --- Recap + verdict ---
    if st.session_state.recap_text is None or st.session_state.verdict_text is None:
        with st.spinner("Building your recap from real race data..."):
            from race_data import build_recap_data, build_race_summary

            if driver_choice == "Whole race (no specific driver)":
                recap_data = build_race_summary(session, from_lap)
                recap_text = get_race_recap(recap_data["summary"])
                verdict_text = get_race_verdict(recap_data["summary"])
            else:
                recap_data = build_recap_data(session, driver_choice, from_lap)
                recap_text = get_recap(recap_data["summary"])
                verdict_text = get_verdict(recap_data["summary"])

            st.session_state.recap_text = recap_text
            st.session_state.verdict_text = verdict_text

    st.subheader("Your Recap")
    st.write(st.session_state.recap_text)

    st.subheader("Strategy Verdict")
    st.info(st.session_state.verdict_text)