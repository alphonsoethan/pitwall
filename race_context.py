# race_context.py
"""
Race-state logic shared by the live view and the Q&A feature:
  - compute_standings: leaderboard at a given lap, with real DNFs, DNS and correct gaps
  - detect_events:     safety cars, red flags, pit stops, retirements
  - build_context:     the text block Gemini answers questions from

No Streamlit import, so it can be tested on its own.
"""
import numpy as np
import pandas as pd

STATUS_MAP = {
    "1": ("GREEN", "#00cc44"),
    "2": ("YELLOW", "#ffcc00"),
    "4": ("SAFETY CAR", "#ff9900"),
    "5": ("RED FLAG", "#ff3333"),
    "6": ("VSC", "#ffcc00"),
    "7": ("VSC ENDING", "#ffcc00"),
}
STATUS_EVENT_TEXT = {
    "1": "Track clear, racing resumes (green flag)",
    "2": "Yellow flag: danger on track",
    "4": "Safety Car deployed",
    "5": "Red flag: race stopped",
    "6": "Virtual Safety Car (VSC) deployed",
    "7": "VSC ending",
}
FALLBACK_WEATHER = {"air": "22.0", "track": "28.0", "humidity": "78", "rain": "Yes", "source": "fallback"}

STANDINGS_COLUMNS = [
    "Driver", "FullName", "DriverNumber", "TeamName", "TeamColor", "Status", "Position",
    "LastLap", "RetiredAfterLap", "Gap", "GapSeconds", "Interval", "Compound", "TyreLife",
    "TyreText", "LapTime", "PitStatus",
]


# ---------------------------------------------------------------- helpers
def format_laptime(td):
    """Format a pandas Timedelta as M:SS.mmm, or a dash if missing."""
    if td is None or pd.isna(td):
        return "—"
    try:
        total = td.total_seconds()
    except Exception:
        return "—"
    minutes = int(total // 60)
    return f"{minutes}:{total - minutes * 60:06.3f}"


def _num_str(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(int(v))
    return str(v)


def _hex(v):
    """FastF1 gives team colours as 'E10600' without '#', which is invalid CSS."""
    if v is None or (not isinstance(v, str) and pd.isna(v)) or str(v).strip() == "":
        return "#888888"
    v = str(v).strip()
    return v if v.startswith("#") else "#" + v


def build_results(session):
    res = session.results.copy()
    if "Abbreviation" in res.columns and "Driver" not in res.columns:
        res = res.rename(columns={"Abbreviation": "Driver"})
    if "DriverNumber" not in res.columns:
        res["DriverNumber"] = ""
    if "FullName" not in res.columns:
        res["FullName"] = res["Driver"]
    if "TeamName" not in res.columns:
        res["TeamName"] = ""
    if "TeamColor" not in res.columns:
        res["TeamColor"] = "#888888"
    if "Status" not in res.columns:
        res["Status"] = ""
    res["DriverNumber"] = res["DriverNumber"].map(_num_str)
    res["TeamColor"] = res["TeamColor"].map(_hex)
    res["Status"] = res["Status"].fillna("").astype(str)
    return res[["Driver", "FullName", "DriverNumber", "TeamName", "TeamColor", "Status"]].copy()


def get_pivots(session):
    """(time, position, laptime) tables indexed by lap number, one column per driver."""
    cached = getattr(session, "_pitwall_pivots", None)
    if cached is not None:
        return cached
    laps = session.laps.copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    laps = laps.drop_duplicates(["Driver", "LapNumber"], keep="last")
    pivots = (
        laps.pivot(index="LapNumber", columns="Driver", values="Time"),
        laps.pivot(index="LapNumber", columns="Driver", values="Position"),
        laps.pivot(index="LapNumber", columns="Driver", values="LapTime"),
    )
    try:
        session._pitwall_pivots = pivots
    except Exception:
        pass
    return pivots


def gap_seconds(session, a, b, lap):
    """Seconds by which driver `a` crossed the line after driver `b` on the SAME lap number."""
    tp = get_pivots(session)[0]
    lap = int(lap)
    if a not in tp.columns or b not in tp.columns or lap not in tp.index:
        return None
    ta, tb = tp.at[lap, a], tp.at[lap, b]
    if pd.isna(ta) or pd.isna(tb):
        return None
    return (ta - tb).total_seconds()


def _leader_lap_seconds(session, leader, lap):
    lt = get_pivots(session)[2]
    if leader not in lt.columns:
        return None
    v = lt.at[lap, leader] if lap in lt.index else pd.NaT
    if pd.isna(v):
        v = lt[leader].median()
    return None if pd.isna(v) else v.total_seconds()


def _is_retired(status, final_lap, lap, total_laps):
    """
    A driver is out when the replay has moved past the last lap they ever completed,
    unless they went on to be classified as a finisher (finished, or lapped at the flag).
    """
    if final_lap >= lap:
        return False
    s = str(status or "").strip()
    if s == "":
        return True  # no result info: fall back to the lap-number rule
    if s == "Finished" or s == "Lapped" or s.startswith("+"):
        return False  # still running, just a lap or more down
    return True


def _laps_label(n):
    return f"+{n} LAP" if n == 1 else f"+{n} LAPS"


def _pit_status(row):
    """
    Returns 'IN' if the driver is stationary in the pits at the end of their last lap,
    'OUT' if they have just left the pits, or '' otherwise.
    FastF1 records PitInTime on the lap a driver enters, and PitOutTime on the next lap.
    """
    if row is None:
        return ""
    pit_in = row.get("PitInTime")
    pit_out = row.get("PitOutTime")
    if pd.notna(pit_in) and pd.isna(pit_out):
        return "IN"
    if pd.isna(pit_in) and pd.notna(pit_out):
        return "OUT"
    return ""


# --------------------------------------------------------------- standings
def compute_standings(session, lap):
    """
    Leaderboard as it stood at the end of `lap`.

    DNF: a driver whose last completed lap is behind the current lap (and who was not
    classified as a finisher). DNS: a driver in results but with no lap at all.
    Gaps: always measured between two cars on the same lap number; cars a lap or more
    down show "+1 LAP" instead of a misleading time.
    """
    lap = int(lap)
    results = build_results(session)
    laps = session.laps
    total = int(laps["LapNumber"].max())
    upto = laps[laps["LapNumber"] <= lap]
    if upto.empty:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    final_lap = laps.groupby("Driver")["LapNumber"].max()
    last = upto.sort_values("LapNumber").groupby("Driver").tail(1).set_index("Driver")
    laps_drivers = set(session.laps["Driver"].dropna().unique())

    running, out = [], []
    for r in results.itertuples(index=False):
        base = {
            "Driver": r.Driver, "FullName": r.FullName, "DriverNumber": r.DriverNumber,
            "TeamName": r.TeamName or "", "TeamColor": r.TeamColor,
        }

        # DNS: driver present in results but never completed a lap in this session
        if r.Driver not in laps_drivers:
            out.append({
                **base, "Status": "DNS", "RetiredAfterLap": None, "Position": None,
                "Gap": "DNS", "GapSeconds": None, "Interval": "—",
                "Compound": "", "TyreLife": None, "TyreText": "-",
                "LapTime": pd.NaT, "LastLap": 0, "PitStatus": "",
            })
            continue

        fl = int(final_lap.get(r.Driver, 0))
        retired = _is_retired(r.Status, fl, lap, total)
        row = last.loc[r.Driver] if r.Driver in last.index else None
        if row is not None:
            comp = row.get("Compound")
            comp = "" if comp is None or pd.isna(comp) else str(comp)
            age = row.get("TyreLife")
            age = int(age) if age is not None and pd.notna(age) else None
            base.update({
                "Compound": comp, "TyreLife": age,
                "TyreText": f"{comp} ({age})" if comp and age is not None else (comp or "-"),
                "LapTime": row.get("LapTime"), "LastLap": int(row["LapNumber"]),
                "PitStatus": _pit_status(row),
            })
        else:
            base.update({"Compound": "", "TyreLife": None, "TyreText": "-",
                         "LapTime": pd.NaT, "LastLap": 0, "PitStatus": ""})

        if retired:
            out.append({**base, "Status": "DNF", "RetiredAfterLap": fl, "Position": None,
                        "Gap": f"DNF (L{fl})", "GapSeconds": None, "Interval": "—"})
        elif row is not None:
            pos = row.get("Position")
            running.append({**base, "Status": "RUNNING", "RetiredAfterLap": None,
                            "_pos": float(pos) if pd.notna(pos) else 999.0,
                            "_time": row.get("Time")})

    run_df = pd.DataFrame(running)
    if not run_df.empty:
        run_df = run_df.sort_values(["_pos", "_time"], kind="stable").reset_index(drop=True)
        run_df["Position"] = np.arange(1, len(run_df) + 1)
        leader = run_df.iloc[0]["Driver"]
        gaps, gap_secs, intervals = [], [], []
        for i, r in run_df.iterrows():
            L = int(r["LastLap"])
            if i == 0:
                gaps.append("Leader"); gap_secs.append(0.0); intervals.append("—")
                continue
            # Gap to leader
            if lap > L:
                gaps.append(_laps_label(lap - L)); gap_secs.append(None)
            else:
                g = gap_seconds(session, r["Driver"], leader, L)
                lt = _leader_lap_seconds(session, leader, L)
                if g is None:
                    gaps.append("—"); gap_secs.append(None)
                elif lt and g >= lt:
                    gaps.append(_laps_label(max(1, int(g // lt)))); gap_secs.append(g)
                else:
                    gaps.append(f"+{g:.1f}s"); gap_secs.append(g)
            # Interval to the car directly ahead, compared on their common lap
            ahead = run_df.iloc[i - 1]
            common = min(L, int(ahead["LastLap"]))
            iv = gap_seconds(session, r["Driver"], ahead["Driver"], common)
            intervals.append("—" if iv is None else f"+{iv:.1f}s")
        run_df["Gap"], run_df["GapSeconds"], run_df["Interval"] = gaps, gap_secs, intervals
        run_df = run_df.drop(columns=["_pos", "_time"])

    # Build the result from plain records (running cars first, then retirements/DNS).
    records = run_df.to_dict("records") if not run_df.empty else []
    # DNFs sorted by retirement lap (latest first); DNS last
    dnfs = [d for d in out if d["Status"] == "DNF"]
    dnss = [d for d in out if d["Status"] == "DNS"]
    records += sorted(dnfs, key=lambda d: d["RetiredAfterLap"], reverse=True)
    records += dnss
    return pd.DataFrame(records, columns=STANDINGS_COLUMNS)


# ------------------------------------------------------ flag, weather, laps
def _leader_cross_time(session, lap):
    t = session.laps[session.laps["LapNumber"] == lap]["Time"]
    return t.min() if not t.empty else pd.NaT


def flag_status(session, lap):
    ts = getattr(session, "track_status", None)
    if ts is None or ts.empty:
        return "GREEN", "#00cc44"
    t = _leader_cross_time(session, lap)
    if pd.isna(t):
        return "GREEN", "#00cc44"
    before = ts[ts["Time"] <= t]
    if before.empty:
        return "GREEN", "#00cc44"
    return STATUS_MAP.get(str(before.iloc[-1]["Status"]), ("GREEN", "#00cc44"))


def weather_at(session, lap):
    """Weather nearest to the leader's lap end, or plausible fallback values."""
    w = getattr(session, "weather_data", None)
    if w is None or w.empty:
        return dict(FALLBACK_WEATHER)
    try:
        t = _leader_cross_time(session, lap)
        if pd.notna(t):
            idx = (w["Time"] - t).abs().to_numpy().argmin()
            row = w.iloc[idx]
        else:
            row = w.iloc[0]
        return {
            "air": f"{row['AirTemp']:.1f}", "track": f"{row['TrackTemp']:.1f}",
            "humidity": f"{row['Humidity']:.0f}",
            "rain": "Yes" if bool(row["Rainfall"]) else "No", "source": "data",
        }
    except Exception:
        return dict(FALLBACK_WEATHER)


def fastest_lap_so_far(session, lap):
    """(driver code, formatted time) for the quickest lap by anyone up to `lap`."""
    laps = session.laps
    valid = laps[laps["LapNumber"] <= lap].dropna(subset=["LapTime"])
    if valid.empty:
        return None, None
    row = valid.loc[valid["LapTime"].idxmin()]
    return row["Driver"], format_laptime(row["LapTime"])


# ------------------------------------------------------------------ events
def detect_events(session, upto_lap):
    """Key moments up to `upto_lap`: flags/safety cars, pit stops, retirements."""
    laps = session.laps
    results = build_results(session)
    total = int(laps["LapNumber"].max())
    events = []

    ts = getattr(session, "track_status", None)
    if ts is not None and not ts.empty:
        leader_end = laps.groupby("LapNumber")["Time"].min().sort_index()
        prev = None
        for r in ts.sort_values("Time").itertuples(index=False):
            code = str(r.Status)
            if code == prev:
                continue
            first = prev is None
            prev = code
            if first and code == "1":
                continue
            ahead = leader_end[leader_end >= r.Time]
            lap_no = int(ahead.index[0]) if not ahead.empty else total
            if lap_no <= upto_lap and code in STATUS_EVENT_TEXT:
                events.append({"lap": lap_no, "kind": "status", "text": STATUS_EVENT_TEXT[code]})

    for drv, g in laps.groupby("Driver"):
        g = g.sort_values("LapNumber").reset_index(drop=True)
        if "Stint" not in g.columns:
            continue
        stint = g["Stint"]
        changed = stint.ne(stint.shift()) & stint.shift().notna() & stint.notna()
        for j in g.index[changed]:
            in_lap = int(g.loc[j - 1, "LapNumber"])
            if in_lap > upto_lap:
                continue
            old = g.loc[j - 1].get("Compound")
            new = g.loc[j].get("Compound")
            old = "" if old is None or pd.isna(old) else old
            new = "" if new is None or pd.isna(new) else new
            swap = f": {old} to {new} tyres" if old and new else ""
            events.append({"lap": in_lap, "kind": "pit", "text": f"{drv} pitted{swap}"})

    final_lap = laps.groupby("Driver")["LapNumber"].max()
    for r in results.itertuples(index=False):
        fl = int(final_lap.get(r.Driver, 0))
        if _is_retired(r.Status, fl, upto_lap, total):
            why = f" ({r.Status})" if r.Status else ""
            events.append({"lap": fl + 1, "kind": "dnf",
                           "text": f"{r.Driver} retired{why}"})

    return sorted(events, key=lambda e: e["lap"])


def recent_position_changes(session, lap, window=5, min_move=2, limit=6):
    _, pp, _ = get_pivots(session)
    results = build_results(session).set_index("Driver")["FullName"].to_dict()
    start = max(1, lap - window)
    if lap not in pp.index or start not in pp.index:
        return []
    moves = []
    for drv in pp.columns:
        a, b = pp.at[start, drv], pp.at[lap, drv]
        if pd.notna(a) and pd.notna(b) and abs(a - b) >= min_move:
            moves.append((abs(a - b), drv, int(a), int(b)))
    moves.sort(reverse=True)
    out = []
    for _, drv, a, b in moves[:limit]:
        verb = "gained" if b < a else "lost"
        out.append(f"{results.get(drv, drv)} {verb} {abs(a - b)} places (P{a} to P{b})")
    return out


# ----------------------------------------------------------------- context
def build_context(session, lap, standings, favourites=None, race_label="São Paulo 2024"):
    """Text snapshot of the race up to `lap`. This is all Gemini is allowed to answer from."""
    lap = int(lap)
    total = int(session.laps["LapNumber"].max())
    flag, _ = flag_status(session, lap)
    w = weather_at(session, lap)
    fl_drv, fl_time = fastest_lap_so_far(session, lap)
    names = build_results(session).set_index("Driver")["FullName"].to_dict()

    lines = [
        f"RACE: {race_label}. Replay of a past race, currently at the end of lap {lap} of {total}.",
        f"TRACK STATUS NOW: {flag}",
        f"WEATHER NOW: air {w['air']}C, track {w['track']}C, humidity {w['humidity']}%, rain: {w['rain']}"
        + (" (approximate values, no weather data was available)" if w["source"] == "fallback" else ""),
    ]
    if fl_drv:
        lines.append(f"FASTEST LAP SO FAR: {names.get(fl_drv, fl_drv)} {fl_time}")

    lines.append("\nRUNNING ORDER (position, driver code, name, team, gap to leader, interval to car ahead, tyre and its age in laps):")
    for r in standings[standings["Status"] == "RUNNING"].itertuples(index=False):
        pit = " (currently in the pits)" if r.PitStatus == "IN" else ""
        lines.append(f"P{r.Position} {r.Driver} {r.FullName} ({r.TeamName}) gap {r.Gap}, "
                     f"interval {r.Interval}, tyre {r.TyreText}{pit}")
    dnfs = standings[standings["Status"] == "DNF"]
    if not dnfs.empty:
        lines.append("\nOUT OF THE RACE:")
        for r in dnfs.itertuples(index=False):
            lines.append(f"{r.Driver} {r.FullName} stopped after lap {r.RetiredAfterLap}")
    dnss = standings[standings["Status"] == "DNS"]
    if not dnss.empty:
        lines.append("\nDID NOT START:")
        for r in dnss.itertuples(index=False):
            lines.append(f"{r.Driver} {r.FullName} did not start")

    events = detect_events(session, lap)
    keep = [e for e in events if e["kind"] != "pit"] + [e for e in events if e["kind"] == "pit"][-15:]
    keep.sort(key=lambda e: e["lap"])
    if keep:
        lines.append("\nKEY EVENTS SO FAR (lap: event):")
        lines += [f"Lap {e['lap']}: {e['text']}" for e in keep]

    moves = recent_position_changes(session, lap)
    if moves:
        lines.append("\nPOSITION CHANGES OVER THE LAST 5 LAPS:")
        lines += moves

    favs = [f for f in (favourites or []) if f and f != "None"]
    if favs:
        lines.append("\nDRIVERS THE FAN IS FOLLOWING:")
        for name in favs:
            row = standings[standings["FullName"] == name]
            if row.empty:
                lines.append(f"{name}: not found")
                continue
            r = row.iloc[0]
            if r["Status"] == "DNF":
                lines.append(f"{name}: out of the race after lap {r['RetiredAfterLap']}")
            elif r["Status"] == "DNS":
                lines.append(f"{name}: did not start")
            else:
                pit = " (currently in the pits)" if r["PitStatus"] == "IN" else ""
                lines.append(f"{name}: P{r['Position']}, gap {r['Gap']}, interval {r['Interval']}, "
                             f"tyre {r['TyreText']}{pit}, last lap {format_laptime(r['LapTime'])}")
    return "\n".join(lines)