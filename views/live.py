# views/live.py
import time

import pandas as pd
import streamlit as st

from config import get_shared_session
from gemini_client import get_answer, get_strategy
from glossary import GLOSSARY, find_terms
from race_context import (
    build_context, compute_standings, fastest_lap_so_far, flag_status,
    format_laptime, gap_seconds, weather_at,
)

TICK_SECONDS = 5
SUGGESTED_QUESTIONS = [
    "What just happened?",
    "Why did the race stop?",
    "How are my drivers doing?",
    "What is an undercut?",
]


@st.cache_data(show_spinner=False, max_entries=200)
def _standings(_session, lap):
    # _session starts with an underscore so Streamlit does not try to hash it.
    return compute_standings(_session, lap)


def _reset_replay():
    st.session_state.live_lap = 0
    st.session_state.live_last_tick = None
    st.session_state.live_final_rerun = False
    st.session_state.strategy_text = None


def _advance_lap(total_laps):
    """
    Move the replay forward by one lap, but only if a full tick has passed.
    Time-based, so a click on any widget inside the live panel does not skip laps.
    """
    if st.session_state.live_paused or st.session_state.live_lap >= total_laps:
        return
    now = time.time()
    last = st.session_state.live_last_tick
    if last is None or now - last >= TICK_SECONDS - 0.5:
        st.session_state.live_lap += 1
        st.session_state.live_last_tick = now


def _render_leaderboard(standings):
    with st.container(height=440, border=False):
        for r in standings.to_dict("records"):
            c1, c2, c3, c4, c5 = st.columns([0.4, 3, 1.8, 1.2, 1.8])
            c1.markdown(
                f"<div style='background:{r['TeamColor']};width:8px;height:24px;border-radius:2px'></div>",
                unsafe_allow_html=True,
            )
            if r["Status"] == "DNF":
                c2.markdown(f"~~**{r['DriverNumber']} {r['FullName']}**~~")
            else:
                c2.markdown(f"**P{r['Position']}** · {r['DriverNumber']} {r['FullName']}")
            c3.caption(r["TeamName"])
            c4.write(r["Gap"])
            c5.caption(r["TyreText"] if r["Status"] == "RUNNING" else "-")


def _favourite_details(name, standings, session, lap):
    row = standings[standings["FullName"] == name]
    if row.empty:
        return f"**{name}** — driver not found."
    r = row.iloc[0]
    head = f"**{name}** ({r['Driver']}) — {r['TeamName']}"
    if r["Status"] == "DNF":
        return f"{head}\n\n- Out of the race after lap {int(r['RetiredAfterLap'])}"

    running = standings[standings["Status"] == "RUNNING"]
    pos = int(r["Position"])
    L = int(r["LastLap"])

    def neighbour(offset):
        other = running[running["Position"] == pos + offset]
        if other.empty:
            return "—"
        o = other.iloc[0]
        g = gap_seconds(session, r["Driver"], o["Driver"], min(L, int(o["LastLap"])))
        gap = "—" if g is None else f"{abs(g):.1f}s"
        return f"{o['FullName']} (+{gap})"

    valid = session.laps[(session.laps["Driver"] == r["Driver"]) & (session.laps["LapNumber"] <= lap)]["LapTime"].dropna()
    best = format_laptime(valid.min()) if not valid.empty else "—"
    tyre = f"{r['Compound'] or '-'} ({r['TyreLife']} laps)" if r["TyreLife"] is not None else (r["Compound"] or "-")
    return (
        f"{head}\n\n"
        f"- Position: **P{pos}** (gap to leader: {r['Gap']})\n"
        f"- Tyre: {tyre}\n"
        f"- Current lap time: **{format_laptime(r['LapTime'])}**\n"
        f"- Fastest lap so far: **{best}**\n"
        f"- Ahead: {neighbour(-1)}\n"
        f"- Behind: {neighbour(+1)}"
    )


def _render_favourites(standings, session, lap):
    st.markdown("**Select drivers to follow closely**")
    names = sorted(standings["FullName"].tolist())

    prev_fav1 = st.session_state.get("fav1", "None")
    options1 = ["None"] + names
    options2 = ["None"] + [d for d in names if d != prev_fav1]

    col1, col2 = st.columns(2)
    with col1:
        fav1 = st.selectbox("Driver 1", options1, key="fav1")
    with col2:
        fav2 = st.selectbox("Driver 2", options2, key="fav2")

    if fav1 != "None" and fav1 == fav2:
        st.session_state["fav2"] = "None"
        fav2 = "None"

    if fav1 == "None" and fav2 == "None":
        st.caption("Select a driver above to see their live details here.")
        return

    d1, d2 = st.columns(2)
    with d1:
        if fav1 != "None":
            st.markdown(_favourite_details(fav1, standings, session, lap))
    with d2:
        if fav2 != "None":
            st.markdown(_favourite_details(fav2, standings, session, lap))


def _render_qa(session):
    """'What did I miss?' box. Lives outside the auto-refreshing panel so typing is never interrupted."""
    st.divider()
    st.subheader("💬 What did I miss?")
    st.caption("Ask anything about the race so far. Answers come only from the lap data, and jargon gets explained.")

    clicked = None
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(q, key=f"suggest_{q}", use_container_width=True):
            clicked = q

    with st.form("qa_form", clear_on_submit=True):
        typed = st.text_input(
            "Your question",
            max_chars=300,
            placeholder="e.g. Why is there a safety car? Is Hamilton on a good strategy?",
        )
        submitted = st.form_submit_button("Ask")

    question = clicked or (typed.strip() if submitted else "")
    if question:
        lap = int(st.session_state.live_lap)
        if lap < 1:
            st.warning("The replay has not started yet. Ask again in a moment.")
        else:
            standings = _standings(session, lap)
            favourites = [st.session_state.get("fav1"), st.session_state.get("fav2")]
            context = build_context(session, lap, standings, favourites)
            with st.spinner("Checking the race data..."):
                answer = get_answer(question, context)
            st.session_state.qa_history.insert(0, {"q": question, "a": answer, "lap": lap})
            del st.session_state.qa_history[6:]

    for item in st.session_state.qa_history:
        with st.container(border=True):
            st.markdown(f"**🙋 {item['q']}**")
            st.markdown(item["a"])
            st.caption(f"Based on the race as of lap {item['lap']}")
            terms = find_terms(item["a"] + " " + item["q"])
            if terms:
                with st.expander("📖 Jargon in this answer"):
                    for term, meaning in terms:
                        st.markdown(f"**{term}**: {meaning}")

    with st.expander("📖 F1 jargon buster"):
        for term, (_, meaning) in GLOSSARY.items():
            st.markdown(f"**{term}**: {meaning}")


def render():
    for key, default in [
        ("live_lap", 0), ("live_paused", False), ("live_last_tick", None),
        ("live_final_rerun", False), ("strategy_text", None), ("qa_history", []),
    ]:
        st.session_state.setdefault(key, default)

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.session_state.strategy_text = None
        st.rerun()

    st.title("📡 Livestream Mode")

    session = get_shared_session()
    total_laps = int(session.laps["LapNumber"].max())
    finished = st.session_state.live_lap >= total_laps

    # Controls live outside the fragment so a click re-decorates it with the right timer
    c_pause, c_restart, _ = st.columns([1, 1, 6])
    if c_pause.button("▶ Resume" if st.session_state.live_paused else "⏸ Pause", disabled=finished):
        st.session_state.live_paused = not st.session_state.live_paused
        st.rerun()
    if c_restart.button("↺ Restart"):
        _reset_replay()
        st.session_state.live_paused = False
        st.rerun()

    # No timer once the race is over or paused, so the page stops re-running every 5 seconds
    run_every = None if (st.session_state.live_paused or finished) else f"{TICK_SECONDS}s"

    @st.fragment(run_every=run_every)
    def live_dashboard():
        _advance_lap(total_laps)
        lap = st.session_state.live_lap

        status_note = " · ⏸ paused" if st.session_state.live_paused else (" · 🏁 finished" if lap >= total_laps else "")
        st.caption(f"Simulated replay of a past race · Lap {lap} / {total_laps}{status_note}")

        if lap < 1:
            st.info("Waiting for lap 1 data...")
            return

        standings = _standings(session, lap)
        flag, flag_color = flag_status(session, lap)
        weather = weather_at(session, lap)

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
                        {flag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Fastest lap so far**")
            fl_driver, fl_time = fastest_lap_so_far(session, lap)
            if fl_driver is None:
                st.caption("No valid lap times yet.")
            else:
                st.info(f"🟣 **{fl_time}** — {fl_driver}")

            st.markdown("**Weather**")
            w1, w2 = st.columns(2)
            w1.metric("Air", f"{weather['air']}°C")
            w2.metric("Track", f"{weather['track']}°C")
            w3, w4 = st.columns(2)
            w3.metric("Humidity", f"{weather['humidity']}%")
            w4.metric("Rain", weather["rain"])
            if weather["source"] == "fallback":
                st.caption("Weather data unavailable, showing typical values.")

        with right_col:
            tab_lb, tab_fav = st.tabs(["🏁 Leaderboard", "⭐ My Drivers"])
            with tab_lb:
                _render_leaderboard(standings)
            with tab_fav:
                _render_favourites(standings, session, lap)

        st.subheader("🤖 Strategy Suggestion")
        if st.button("Get AI strategy for current conditions"):
            top10 = standings[standings["Status"] == "RUNNING"].head(10)
            lb_lines = [f"P{r.Position} {r.FullName} ({r.TyreText})" for r in top10.itertuples()]
            with st.spinner("Gemini is thinking..."):
                st.session_state.strategy_text = get_strategy(weather, "\n".join(lb_lines))
        if st.session_state.strategy_text:
            st.info(st.session_state.strategy_text)

        # Race over: do one full-page rerun so the fragment is re-created without its timer
        if lap >= total_laps and not st.session_state.live_final_rerun:
            st.session_state.live_final_rerun = True
            st.rerun()

    live_dashboard()
    _render_qa(session)