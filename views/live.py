# views/live.py
import pandas as pd
import streamlit as st
import fastf1
import os

from gemini_client import get_strategy
from config import get_shared_session

# def _f1_credentials():
#     try:
#         return st.secrets["F1_USERNAME"], st.secrets["F1_PASSWORD"]
#     except Exception:
#         return os.environ.get("F1_USERNAME"), os.environ.get("F1_PASSWORD")


# @st.cache_data(show_spinner="Loading race data...")
# def _load_session():
#     user, pw = _f1_credentials()
#     if user and pw:
#         os.environ["F1_USERNAME"] = user
#         os.environ["F1_PASSWORD"] = pw

#     session = fastf1.get_session(2024, "São Paulo", "R")
#     session.load(telemetry=False, weather=True, messages=False)

#     if session.laps is None or len(session.laps) == 0:
#         raise RuntimeError(
#             "F1 timing data came back empty. Check F1_USERNAME / F1_PASSWORD "
#             "in Streamlit secrets, or try again in a minute."
#         )
#     return session


def _build_results(session):
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
    return res[["Driver", "FullName", "DriverNumber", "TeamName", "TeamColor"]].copy()


def _flag_status(session, current_lap):
    if session.track_status.empty:
        return "GREEN", "#00cc44"
    lap_times = session.laps[session.laps["LapNumber"] == current_lap]["Time"]
    if lap_times.empty:
        return "GREEN", "#00cc44"
    ts_before = session.track_status[session.track_status["Time"] <= lap_times.max()]
    if ts_before.empty:
        return "GREEN", "#00cc44"
    status_code = str(ts_before.iloc[-1]["Status"])
    status_map = {
        "1": ("GREEN", "#00cc44"),
        "2": ("YELLOW", "#ffcc00"),
        "4": ("SAFETY CAR", "#ff9900"),
        "5": ("RED FLAG", "#ff3333"),
        "6": ("VSC", "#ffcc00"),
    }
    return status_map.get(status_code, ("GREEN", "#00cc44"))


def _format_laptime(td):
    """Format a pandas Timedelta as M:SS.mmm, or '—' if missing."""
    if pd.isna(td):
        return "—"
    try:
        total_seconds = td.total_seconds()
    except Exception:
        return "—"
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


def _tyre_expected_life():
    """Approximate expected tyre life per compound (in laps). Simple heuristic."""
    return {
        "SOFT": "~15–20 laps",
        "MEDIUM": "~25–30 laps",
        "HARD": "~35–40 laps",
        "INTERMEDIATE": "~25–35 laps (wet-dependent)",
        "WET": "~20–30 laps (heavy rain)",
    }


def _fastest_lap_so_far(laps_to_now):
    """Return (driver, formatted_laptime) for the fastest lap set by ANY driver so far."""
    if laps_to_now.empty or "LapTime" not in laps_to_now.columns:
        return None, None
    valid = laps_to_now.dropna(subset=["LapTime"])
    if valid.empty:
        return None, None
    fastest_row = valid.loc[valid["LapTime"].idxmin()]
    return fastest_row.get("Driver", "?"), _format_laptime(fastest_row["LapTime"])


def _render_leaderboard(display_rows):
    with st.container(height=440, border=False):
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


def _render_favourites(laps_to_now, results, merged):
    st.markdown("**Select drivers to follow closely**")
    all_driver_names = results.sort_values("FullName")["FullName"].tolist()

    # Preserve previous selections across reruns
    prev_fav1 = st.session_state.get("fav1", "None")
    prev_fav2 = st.session_state.get("fav2", "None")

    # Options for Driver 2 exclude whatever is picked in Driver 1
    options1 = ["None"] + all_driver_names
    options2 = ["None"] + [d for d in all_driver_names if d != prev_fav1]

    fav_col1, fav_col2 = st.columns(2)
    with fav_col1:
        fav1 = st.selectbox("Driver 1", options1, key="fav1")
    with fav_col2:
        fav2 = st.selectbox("Driver 2", options2, key="fav2")

    # Guard: if somehow both end up the same (e.g. old state), reset Driver 2
    if fav1 != "None" and fav1 == fav2:
        st.session_state["fav2"] = "None"
        fav2 = "None"

    if fav1 == "None" and fav2 == "None":
        st.caption("Select a driver above to see their live details here.")
        return

    def details(driver_name):
        row = results[results["FullName"] == driver_name]
        if row.empty:
            return f"**{driver_name}** — driver not found."
        abbrev = row.iloc[0]["Driver"]
        team = row.iloc[0].get("TeamName", "") or ""
        driver_laps = laps_to_now[laps_to_now["Driver"] == abbrev].sort_values("LapNumber")
        if driver_laps.empty:
            return f"**{driver_name}** ({abbrev}) — no lap data yet (DNF or not started)."

        last = driver_laps.iloc[-1]
        pos = int(last["Position"]) if pd.notna(last["Position"]) else "-"
        tyre = last.get("Compound", "-")
        tyre_life = last.get("TyreLife", "-")

        current_laptime = _format_laptime(last.get("LapTime"))
        valid_laptimes = driver_laps["LapTime"].dropna()
        fastest_laptime = _format_laptime(valid_laptimes.min()) if not valid_laptimes.empty else "—"

        gap_leader = "—"
        if not merged.empty and "Time" in merged.columns and pd.notna(last.get("Time")):
            leader_time = merged.iloc[0]["Time"]
            if pd.notna(leader_time):
                gap_leader = f"+{(last['Time'] - leader_time).total_seconds():.1f}s"

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

        return (
            f"**{driver_name}** ({abbrev}) — {team}\n\n"
            f"- Position: **P{pos}** (gap to leader: {gap_leader})\n"
            f"- Tyre: {tyre} ({int(tyre_life) if pd.notna(tyre_life) else '-'} laps)\n"
            f"- Current lap time: **{current_laptime}**\n"
            f"- Fastest lap so far: **{fastest_laptime}**\n"
            f"- Ahead: {ahead_info}\n"
            f"- Behind: {behind_info}"
        )

    d1, d2 = st.columns(2)
    with d1:
        if fav1 != "None":
            st.markdown(details(fav1))
    with d2:
        if fav2 != "None":
            st.markdown(details(fav2))

def render():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.session_state.strategy_text = None
        st.rerun()

    st.title("📡 Livestream Mode")

    session = get_shared_session()
    total_laps = int(session.laps["LapNumber"].max())

    @st.fragment(run_every="5s")
    def live_dashboard():
        current_lap = st.session_state.live_lap
        st.caption(f"Simulated live playback · Lap {current_lap} / {total_laps}")

        results = _build_results(session)
        flag_status, flag_color = _flag_status(session, current_lap)

        # Compute once — used by both columns
        laps_to_now = session.laps[session.laps["LapNumber"] <= current_lap].copy()

        left_col, right_col = st.columns([1, 1.6])

        # --- Left column ---
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
            
            st.markdown("**Fastest lap so far**")
            fl_driver, fl_time = _fastest_lap_so_far(laps_to_now)
            if fl_driver is None:
                st.caption("No valid lap times yet.")
            else:
                st.info(f"🟣 **{fl_time}** — {fl_driver}")


            st.markdown("**Weather**")
            weather_dict = {"air": "N/A", "track": "N/A", "humidity": "N/A", "rain": "No"}

            weather = None
            try:
                weather = session.weather_data
            except Exception:
                weather = None

            if weather is not None and not weather.empty:
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
            
        # --- Right column ---
        display_rows = []
        merged = pd.DataFrame()

        with right_col:
            tab_lb, tab_fav = st.tabs(["🏁 Leaderboard", "⭐ My Drivers"])

            with tab_lb:
                active_drivers = (
                    laps_to_now["Driver"].dropna().unique().tolist()
                    if not laps_to_now.empty else []
                )
                all_registered = results["Driver"].dropna().unique().tolist()

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
                            gap_str = f"+{(row['Time'] - leader_time).total_seconds():.1f}s"

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

                    _render_leaderboard(display_rows)

            with tab_fav:
                _render_favourites(laps_to_now, results, merged)

        # --- Strategy ---
        st.subheader("🤖 Strategy Suggestion")
        if st.button("Get AI strategy for current conditions"):
            lb_lines = [f"P{r['P']} {r['Driver']} ({r['Tyre']})" for r in display_rows[:10]]
            with st.spinner("Gemini is thinking..."):
                st.session_state.strategy_text = get_strategy(weather_dict, "\n".join(lb_lines))
        if st.session_state.strategy_text:
            st.info(st.session_state.strategy_text)

        if current_lap < total_laps:
            st.session_state.live_lap += 1

    live_dashboard()