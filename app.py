import streamlit as st
import pandas as pd
import fastf1
import os
import time
import random
from google import genai
from google.genai import types

st.set_page_config(page_title="PitWall", page_icon="🏎️", layout="wide")

# ---------- session state ----------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "live_lap" not in st.session_state:
    st.session_state.live_lap = 0
if "strategy_text" not in st.session_state:
    st.session_state.strategy_text = None
if "recap_text" not in st.session_state:
    st.session_state.recap_text = None
if "verdict_text" not in st.session_state:
    st.session_state.verdict_text = None

# ---------- Race map ----------
RACE_MAP = {
    "São Paulo 2024": {"year": 2024, "race": "São Paulo", "laps": 71},
    "Monza 2024": {"year": 2024, "race": "Monza", "laps": 53},
    "Silverstone 2024": {"year": 2024, "race": "Silverstone", "laps": 52},
}


# ---------- Cached session loaders ----------
@st.cache_data(show_spinner="Loading race data...")
def get_live_session():
    CACHE_DIR = "fastf1_cache"
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(2024, "São Paulo", "R")
    session.load(telemetry=False, weather=True, messages=False)
    return session


@st.cache_data(show_spinner="Loading race data...")
def get_session_for_race(year, race):
    CACHE_DIR = "fastf1_cache"
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(year, race, "R")
    session.load(telemetry=False, weather=False, messages=False)
    return session


# ---------- Gemini client ----------
def get_gemini_client():
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_gemini(prompt):
    client = get_gemini_client()
    if client is None:
        return "Gemini API key not configured. Add GEMINI_API_KEY to .streamlit/secrets.toml."

    models_to_try = ["gemini-3.6-flash", "gemini-3.8-flash"]
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=2000),
                )
                return response.text
            except Exception as e:
                err = str(e)
                if ("503" in err or "429" in err) and attempt < 2:
                    time.sleep(min(2 ** attempt + random.uniform(0, 1), 10))
                    continue
                break
    return "Gemini is currently under high demand. Please try again in a few seconds."


def get_gemini_strategy(weather_data, leaderboard_text):
    prompt = f"""You are an F1 race strategist. Based on the following live race conditions, suggest the best tyre strategy and expected pit stops.

Weather: Air {weather_data['air']}°C, Track {weather_data['track']}°C, Humidity {weather_data['humidity']}%, Rain: {weather_data['rain']}

Current leaderboard (top 10):
{leaderboard_text}

Give a concise recommendation (2-3 sentences) covering:
1. Which tyre compound is best suited right now and why
2. Expected number of pit stops for the remaining race
3. Any risk factors (weather, degradation)"""
    return _call_gemini(prompt)


def get_gemini_recap(summary):
    prompt = f"""You are a knowledgeable F1 race companion for a fan who follows ONE driver.
Here is a structured summary of what happened after a specific lap, from real race timing data.

{summary}

Write a fan-friendly recap of 150-250 words focused on the chosen driver.
Rules:
- Do NOT invent events not in the data.
- If a RED FLAG is mentioned, explain it as race context, not a driver incident.
- Mention strategy, tyre life, position changes.
- Tone: like a friend explaining the race. Avoid superlatives like "greatest ever".
- End with one sentence on what to watch next."""
    return _call_gemini(prompt)


def get_gemini_verdict(summary):
    prompt = f"""You are an F1 strategy analyst. Given this race summary, write a 60-100 word verdict on whether the driver's strategy worked after the cutoff lap.

{summary}

Rules:
- Cite position changes and tyre usage from the data.
- If the red flag affected strategy, mention it.
- Do NOT invent pit stops or incidents.
- Analytical but accessible tone."""
    return _call_gemini(prompt)


# ============================================================
# Page 1: Home
# ============================================================
if st.session_state.page == "home":
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

# ============================================================
# Page 2: Livestream mode
# ============================================================
elif st.session_state.page == "live":
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.session_state.strategy_text = None
        st.rerun()

    st.title("📡 Livestream Mode")

    session = get_live_session()
    total_laps = int(session.laps["LapNumber"].max())

    @st.fragment(run_every="5s")
    def live_dashboard():
        current_lap = st.session_state.live_lap
        st.caption(f"Simulated live playback · Lap {current_lap} / {total_laps}")

        # --- Safe results build ---
        res = session.results.copy()
        if "Abbreviation" in res.columns and "Driver" not in res.columns:
            res = res.rename(columns={"Abbreviation": "Driver"})
        if "DriverNumber" not in res.columns:
            if "DriverNumber" in session.laps.columns:
                num_map = (
                    session.laps[["Driver", "DriverNumber"]]
                    .dropna()
                    .drop_duplicates("Driver")
                    .set_index("Driver")["DriverNumber"]
                    .to_dict()
                )
                res["DriverNumber"] = res["Driver"].map(num_map)
            else:
                res["DriverNumber"] = ""
        if "FullName" not in res.columns:
            res["FullName"] = res.get("Driver", "?")
        if "TeamName" not in res.columns:
            res["TeamName"] = ""
        if "TeamColor" not in res.columns:
            res["TeamColor"] = "#888888"
        results = res[["Driver", "FullName", "DriverNumber", "TeamName", "TeamColor"]].copy()

        # --- Flag status ---
        flag_status, flag_color = "GREEN", "#00cc44"
        if not session.track_status.empty:
            lap_times = session.laps[session.laps["LapNumber"] == current_lap]["Time"]
            if not lap_times.empty:
                current_time = lap_times.max()
                ts_before = session.track_status[session.track_status["Time"] <= current_time]
                if not ts_before.empty:
                    status_code = str(ts_before.iloc[-1]["Status"])
                    status_map = {
                        "1": ("GREEN", "#00cc44"),
                        "2": ("YELLOW", "#ffcc00"),
                        "4": ("SAFETY CAR", "#ff9900"),
                        "5": ("RED FLAG", "#ff3333"),
                        "6": ("VSC", "#ffcc00"),
                    }
                    flag_status, flag_color = status_map.get(status_code, ("GREEN", "#00cc44"))

        left_col, right_col = st.columns([1, 1.6])

        with left_col:
            st.markdown(
                f"""
                <div style="background-color:#1a1a2e;border:2px solid {flag_color};border-radius:10px;
                    height:200px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;color:#e94560;font-size:20px;position:relative;">
                    🎥 LIVE STREAM (Dummy)
                    <div style="position:absolute;top:10px;left:10px;background:{flag_color};
                        color:#000;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold;">
                        {flag_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Weather**")
            weather = session.weather_data
            weather_dict = {"air": "N/A", "track": "N/A", "humidity": "N/A", "rain": "No"}
            if not weather.empty:
                lap_times = session.laps[session.laps["LapNumber"] == current_lap]["Time"]
                current_lap_time = lap_times.max() if not lap_times.empty else None
                if pd.notna(current_lap_time):
                    w = weather.iloc[(weather["Time"] - current_lap_time).abs().argsort()[:1]].iloc[0]
                else:
                    w = weather.iloc[0]
                weather_dict = {
                    "air": f"{w['AirTemp']:.1f}",
                    "track": f"{w['TrackTemp']:.1f}",
                    "humidity": f"{w['Humidity']:.0f}",
                    "rain": "Yes" if w["Rainfall"] else "No",
                }
                w1, w2 = st.columns(2)
                w1.metric("Air", f"{w['AirTemp']:.1f}°C")
                w2.metric("Track", f"{w['TrackTemp']:.1f}°C")
                w3, w4 = st.columns(2)
                w3.metric("Humidity", f"{w['Humidity']:.0f}%")
                w4.metric("Rain", "Yes" if w["Rainfall"] else "No")
            else:
                st.caption("Weather unavailable.")

        with right_col:
            st.subheader("Leaderboard")
            laps_to_now = session.laps[session.laps["LapNumber"] <= current_lap].copy()
            active_drivers = laps_to_now["Driver"].dropna().unique().tolist() if not laps_to_now.empty else []
            all_registered = results["Driver"].dropna().unique().tolist()

            display_rows = []
            merged = pd.DataFrame()
            if laps_to_now.empty:
                st.info(f"Waiting for lap {current_lap} data...")
            else:
                latest = (
                    laps_to_now.sort_values("LapNumber")
                    .groupby("Driver", as_index=False)
                    .last()
                )
                merged = latest.merge(results, on="Driver", how="left")
                merged = merged.sort_values("Position").reset_index(drop=True)
                leader_time = merged.iloc[0]["Time"] if "Time" in merged.columns else None

                for i, row in merged.iterrows():
                    pos = int(row["Position"]) if pd.notna(row["Position"]) else i + 1
                    driver_name = row.get("FullName") or row.get("Driver", "?")
                    number = row.get("DriverNumber", "")
                    number = "" if pd.isna(number) else (str(int(number)) if isinstance(number, (int, float)) else str(number))
                    team = row.get("TeamName", "") or ""
                    colour = row.get("TeamColor", "#888888") or "#888888"

                    gap_str = "Leader"
                    if i > 0 and leader_time is not None and pd.notna(row.get("Time")):
                        gap_seconds = (row["Time"] - leader_time).total_seconds()
                        gap_str = f"+{gap_seconds:.1f}s"

                    tyre = row.get("Compound", "")
                    tyre_life = row.get("TyreLife", "")
                    tyre_str = f"{tyre} ({int(tyre_life)})" if pd.notna(tyre_life) else (str(tyre) if tyre else "-")

                    display_rows.append({
                        "P": pos, "No": number, "Driver": driver_name,
                        "Team": team, "Gap": gap_str, "Tyre": tyre_str,
                        "Colour": colour, "Status": "RUNNING",
                    })

                for drv in all_registered:
                    if drv not in active_drivers:
                        d_row = results[results["Driver"] == drv]
                        if d_row.empty:
                            continue
                        d = d_row.iloc[0]
                        d_num = d.get("DriverNumber", "")
                        d_num = "" if pd.isna(d_num) else (str(int(d_num)) if isinstance(d_num, (int, float)) else str(d_num))
                        display_rows.append({
                            "P": "—", "No": d_num,
                            "Driver": d.get("FullName") or drv,
                            "Team": d.get("TeamName", "") or "",
                            "Gap": "DNF", "Tyre": "-",
                            "Colour": d.get("TeamColor", "#555555") or "#555555",
                            "Status": "DNF",
                        })

                with st.container(height=400, border=False):
                    for r in display_rows:
                        c1, c2, c3, c4, c5 = st.columns([0.4, 3, 1.8, 1.2, 1.8])
                        c1.markdown(
                            f"<div style='background:{r['Colour']};width:8px;height:24px;border-radius:2px'></div>",
                            unsafe_allow_html=True,
                        )
                        if r["Status"] == "DNF":
                            c2.markdown(f"~~**{r['No']} {r['Driver']}**~~")
                        else:
                            c2.markdown(f"**P{r['P']}** · {r['No']} {r['Driver']}")
                        c3.caption(r["Team"])
                        c4.write(r["Gap"])
                        c5.caption(r["Tyre"])

        # --- Gemini strategy ---
        st.subheader("🤖 Strategy Suggestion")
        if st.button("Get AI strategy for current conditions"):
            lb_lines = [f"P{r['P']} {r['Driver']} ({r['Tyre']})" for r in display_rows[:10]]
            leaderboard_text = "\n".join(lb_lines)
            with st.spinner("Gemini is thinking..."):
                st.session_state.strategy_text = get_gemini_strategy(weather_dict, leaderboard_text)
        if st.session_state.strategy_text:
            st.info(st.session_state.strategy_text)

        st.divider()

        # --- Favourites ---
        st.subheader("Favourite Drivers")
        all_driver_names = results.sort_values("FullName")["FullName"].tolist()
        fav_col1, fav_col2 = st.columns(2)
        with fav_col1:
            fav1 = st.selectbox("Favourite 1 (optional)", ["None"] + all_driver_names, key="fav1")
        with fav_col2:
            fav2 = st.selectbox("Favourite 2 (optional)", ["None"] + all_driver_names, key="fav2")

        def driver_details(driver_name, container):
            row = results[results["FullName"] == driver_name]
            if row.empty:
                container.warning("Driver not found.")
                return
            abbrev = row.iloc[0]["Driver"]
            driver_laps = laps_to_now[laps_to_now["Driver"] == abbrev].sort_values("LapNumber")
            if driver_laps.empty:
                container.info("No lap data yet (DNF or not started).")
                return
            last = driver_laps.iloc[-1]
            pos = int(last["Position"]) if pd.notna(last["Position"]) else "-"
            tyre = last.get("Compound", "-")
            tyre_life = last.get("TyreLife", "-")

            ahead_info, behind_info = "—", "—"
            if isinstance(pos, int) and pos > 1 and not merged.empty:
                car_ahead = merged[merged["Position"] == pos - 1]
                if not car_ahead.empty and pd.notna(last.get("Time")):
                    gap = (last["Time"] - car_ahead.iloc[0]["Time"]).total_seconds()
                    ahead_info = f"{car_ahead.iloc[0].get('FullName', '?')} (+{gap:.1f}s)"
            if isinstance(pos, int) and not merged.empty and pos < len(merged):
                car_behind = merged[merged["Position"] == pos + 1]
                if not car_behind.empty and pd.notna(last.get("Time")):
                    gap = (car_behind.iloc[0]["Time"] - last["Time"]).total_seconds()
                    behind_info = f"{car_behind.iloc[0].get('FullName', '?')} (+{gap:.1f}s)"

            container.markdown(f"**{driver_name}** ({abbrev})")
            container.write(f"Position: P{pos}")
            container.write(f"Tyre: {tyre} ({int(tyre_life) if pd.notna(tyre_life) else '-'} laps)")
            container.write(f"Ahead: {ahead_info}")
            container.write(f"Behind: {behind_info}")

        if fav1 != "None" or fav2 != "None":
            st.subheader("Your Drivers in Detail")
            d1, d2 = st.columns(2)
            with d1:
                if fav1 != "None":
                    driver_details(fav1, d1)
            with d2:
                if fav2 != "None":
                    driver_details(fav2, d2)

        if current_lap < total_laps:
            st.session_state.live_lap += 1

    live_dashboard()

# ============================================================
# Page 3: Recap setup
# ============================================================
elif st.session_state.page == "recap_setup":
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🎬 Race Recap")
    st.write("Pick a race, a driver, and the lap you left at. We'll recap everything after that.")

    race_label = st.selectbox("Race", list(RACE_MAP.keys()))
    race_info = RACE_MAP[race_label]
    max_laps = int(race_info["laps"])   # instant — no session load

    driver = st.selectbox(
        "Driver to focus on",
        ["VER", "HAM", "LEC", "NOR", "PIA", "RUS", "SAI", "PER", "TSU", "ALO", "OCO", "GAS"],
    )
    from_lap = st.slider(
        "I left at lap",
        min_value=1,
        max_value=max_laps,
        value=min(10, max_laps),
        help=f"{race_label} has {max_laps} laps.",
    )

    if st.button("Generate Recap →", type="primary"):
        st.session_state.recap_race_label = race_label
        st.session_state.recap_driver = driver
        st.session_state.recap_from_lap = from_lap
        st.session_state.recap_text = None
        st.session_state.verdict_text = None
        st.session_state.page = "recap"
        st.rerun()
        
# ============================================================
# Page 4: Recap display
# ============================================================
elif st.session_state.page == "recap":
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.session_state.recap_text = None
        st.session_state.verdict_text = None
        st.rerun()

    race_label = st.session_state.recap_race_label
    from_lap = st.session_state.recap_from_lap
    driver_abbrev = st.session_state.recap_driver
    race_info = RACE_MAP[race_label]
    year = race_info["year"]
    race_query = race_info["race"]
    
    st.title(f"Recap: {driver_abbrev} · {race_label}")
    st.caption(f"You left at lap {from_lap}")

    # --- Dummy highlights box ---
    st.subheader("Continue Watching")
    st.markdown(
        f"""
        <div style="
            max-width: 640px;
            margin: 0 auto;
        ">
            <div style="
                aspect-ratio: 16 / 9;
                width: 100%;
                background-color: #1a1a2e;
                border: 2px dashed #e94560;
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: #e94560;
                text-align: center;
                padding: 16px;
                box-sizing: border-box;
            ">
                <div style="font-size: 20px;">🎞️ EDITED HIGHLIGHTS</div>
                <div style="font-size: 13px; color: #bbb; margin-top: 8px; max-width: 480px;">
                    Edited highlights of {race_label} from lap {from_lap} onwards
                    will be displayed here.<br>
                    The clip will pick up right where you left off.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"▶ Will resume from lap {from_lap} of {race_label}.")

    st.divider()

    # --- Auto-generate recap + verdict on first load, based on user's selections ---
    if st.session_state.recap_text is None or st.session_state.verdict_text is None:
        with st.spinner(f"Building recap for {driver_abbrev} in {race_label} from lap {from_lap}..."):
            session = get_session_for_race(year, race_query)
            from race_data import build_recap_data
            recap_data = build_recap_data(session, driver_abbrev, from_lap)
            st.session_state.recap_text = get_gemini_recap(recap_data["summary"])
            st.session_state.verdict_text = get_gemini_verdict(recap_data["summary"])

    st.subheader("Your Recap")
    st.write(st.session_state.recap_text)

    st.subheader("Strategy Verdict")
    st.info(st.session_state.verdict_text)