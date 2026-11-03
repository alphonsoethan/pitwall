# race_data.py
import fastf1
import pandas as pd
import numpy as np
import os

# --- Cache setup ---
CACHE_DIR = "f1_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_session(year, race, session_type="R"):
    session = fastf1.get_session(year, race, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    return session


def get_drivers(session):
    return sorted(session.laps["Driver"].dropna().unique().tolist())


def get_driver_laps(session, driver):
    # Works on both a FastF1 session and the cached session (plain DataFrame):
    # match the 3-letter code ("VER") or the car number ("1").
    all_laps = session.laps
    key = str(driver).strip()
    mask = all_laps["Driver"].astype(str).str.upper() == key.upper()
    if "DriverNumber" in all_laps.columns:
        mask = mask | (all_laps["DriverNumber"].astype(str) == key)
    return all_laps[mask].copy().sort_values("LapNumber")


def _build_status_windows(session):
    """
    Return list of (start_time, end_time, status) from session.track_status.
    Status codes: '1'=green, '2'=yellow, '4'=SC, '5'=red, '6'=VSC.
    """
    ts = session.track_status.copy()
    if ts.empty:
        return []

    ts = ts.sort_values("Time").reset_index(drop=True)
    windows = []
    for i, row in ts.iterrows():
        start = row["Time"]
        end = ts.iloc[i + 1]["Time"] if i + 1 < len(ts) else None
        windows.append((start, end, str(row["Status"])))
    return windows


def _lap_is_green(lap_start, lap_end, status_windows):
    """
    A lap is 'green' if most of its duration is under status '1'.
    We approximate: check if any non-green window overlaps this lap
    for more than 15% of the lap duration. If so, not green.
    """
    if lap_end is None or pd.isna(lap_start) or pd.isna(lap_end):
        return False
    lap_dur = (lap_end - lap_start).total_seconds()
    if lap_dur <= 0:
        return False

    non_green_overlap = 0.0
    for w_start, w_end, status in status_windows:
        if status == "1":
            continue
        overlap_start = max(lap_start, w_start)
        overlap_end = min(lap_end, w_end) if w_end is not None else lap_end
        if overlap_end > overlap_start:
            non_green_overlap += (overlap_end - overlap_start).total_seconds()

    return (non_green_overlap / lap_dur) < 0.15


def build_race_summary(session, from_lap):
    """
    Build a whole-race summary (no single driver focus).
    Used when the user picks 'Whole race (no specific driver)'.
    """
    laps = session.laps.copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    after = laps[laps["LapNumber"] > from_lap].copy()

    # Final classification from results
    res = session.results.copy()
    classification = res[["Abbreviation", "FullName", "Position", "Points"]].dropna(subset=["Position"])
    classification = classification.sort_values("Position")

    # Winner
    winner = classification.iloc[0] if not classification.empty else None

    # Podium
    podium = classification.head(3)

    # Fastest lap overall
    valid_laps = laps.dropna(subset=["LapTime"])
    fl = None
    if not valid_laps.empty:
        fl_row = valid_laps.loc[valid_laps["LapTime"].idxmin()]
        fl = (fl_row["Driver"], fl_row["LapTimeSec"])

    # Key events: red flags, safety cars from track status
    events = []
    if not session.track_status.empty:
        ts = session.track_status
        for i, row in ts.iterrows():
            code = str(row["Status"])
            if code in ("4", "5", "6"):
                label = {"4": "Safety Car", "5": "Red Flag", "6": "VSC"}[code]
                events.append(f"{label} at session time {row['Time']}")
        # Keep it short
        events = events[:6]

    summary_lines = [
        f"Race: {session.event['EventName']} {session.event.year}",
        f"User watched up to lap {from_lap} (0 = missed the entire race).",
        f"Total laps: {int(laps['LapNumber'].max())}",
    ]
    if winner is not None:
        summary_lines.append(
            f"Winner: {winner['FullName']} ({winner['Abbreviation']}), "
            f"{int(winner['Points'])} pts"
        )
    if not podium.empty:
        podium_str = ", ".join(
            f"P{int(r['Position'])} {r['FullName']}" for _, r in podium.iterrows()
        )
        summary_lines.append(f"Podium: {podium_str}")
    if fl is not None:
        summary_lines.append(f"Fastest lap: {fl[0]} — {fl[1]:.3f}s")
    if events:
        summary_lines.append("Race control events:")
        for e in events:
            summary_lines.append(f"- {e}")

    summary_lines.append("Final top 10:")
    for _, r in classification.head(10).iterrows():
        summary_lines.append(
            f"- P{int(r['Position'])} {r['FullName']} ({r['Abbreviation']}) — {int(r['Points'])} pts"
        )

    return {
        "driver": None,
        "from_lap": from_lap,
        "laps": after,
        "key_moments": [],
        "summary": "\n".join(summary_lines),
        "session": session,
    }
    

def build_recap_data(session, driver, from_lap):
    laps = get_driver_laps(session, driver)
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    after = laps[laps["LapNumber"] > from_lap].copy()

    if after.empty:
        return {
            "driver": driver,
            "from_lap": from_lap,
            "laps": after,
            "key_moments": [],
            "summary": f"No laps found for {driver} after lap {from_lap}.",
            "session": session,
        }

    if "TyreLife" not in after.columns:
        after["TyreLife"] = np.nan

    median_lap = after["LapTimeSec"].median()
    key_moments = []

    # --- Track status windows for green-lap detection ---
    status_windows = _build_status_windows(session)

    # Build lap start/end times for green check
    # FastF1 laps have 'Time' = time at end of lap (approx session time)
    # We derive lap start as the previous lap's Time, or Time - LapTime
    after_reset = after.reset_index(drop=True)
    lap_starts = []
    lap_ends = []
    for i, row in after_reset.iterrows():
        end = row["Time"] if "Time" in row and pd.notna(row["Time"]) else None
        if i > 0 and pd.notna(after_reset.iloc[i - 1]["Time"]):
            start = after_reset.iloc[i - 1]["Time"]
        else:
            start = end - row["LapTime"] if (end is not None and pd.notna(row["LapTime"])) else None
        lap_starts.append(start)
        lap_ends.append(end)

    after_reset["LapStart"] = lap_starts
    after_reset["LapEnd"] = lap_ends

    # 1. Position changes
    if "Position" in after_reset.columns:
        pos = after_reset[["LapNumber", "Position"]].dropna()
        prev_pos = laps[laps["LapNumber"] <= from_lap]["Position"].dropna()
        last_pos_before = prev_pos.iloc[-1] if not prev_pos.empty else None

        prev = last_pos_before
        for _, row in pos.iterrows():
            if prev is not None and row["Position"] != prev:
                direction = "gained" if row["Position"] < prev else "lost"
                key_moments.append({
                    "lap": int(row["LapNumber"]),
                    "type": "position",
                    "text": f"Lap {int(row['LapNumber'])}: {direction} a position "
                            f"(P{int(prev)} → P{int(row['Position'])})",
                })
            prev = row["Position"]

    # 2. Pit stops (compound change)
    if "Compound" in after_reset.columns:
        comp = after_reset[["LapNumber", "Compound", "TyreLife"]].dropna(subset=["Compound"])
        before_comp = laps[laps["LapNumber"] <= from_lap]["Compound"].dropna()
        prev_comp = before_comp.iloc[-1] if not before_comp.empty else None

        for _, row in comp.iterrows():
            if prev_comp is not None and row["Compound"] != prev_comp:
                key_moments.append({
                    "lap": int(row["LapNumber"]),
                    "type": "pit",
                    "text": f"Lap {int(row['LapNumber'])}: Pit stop — "
                            f"switched to {row['Compound']} tyres",
                })
            prev_comp = row["Compound"]

    # 3. Red flag / SC / VSC events (from track status, mapped to nearest lap)
    red_flag_added = False
    for w_start, w_end, status in status_windows:
        if status not in ("4", "5", "6"):
            continue
        # Find nearest lap in `after` to this event time
        candidates = after_reset[
            after_reset["LapStart"].notna() & after_reset["LapEnd"].notna()
        ]
        if candidates.empty:
            continue
        # Find lap whose interval contains w_start, else nearest
        containing = candidates[
            (candidates["LapStart"] <= w_start) & (candidates["LapEnd"] >= w_start)
        ]
        if not containing.empty:
            lap_no = int(containing.iloc[0]["LapNumber"])
        else:
            # nearest by start time
            nearest_idx = (candidates["LapStart"] - w_start).abs().idxmin()
            lap_no = int(candidates.loc[nearest_idx, "LapNumber"])

        if status == "5":
            key_moments.append({
                "lap": lap_no,
                "type": "red_flag",
                "text": f"Lap ~{lap_no}: RED FLAG — race neutralized. "
                        f"Tyre changes permitted during stoppage.",
            })
            red_flag_added = True
        elif status == "4":
            key_moments.append({
                "lap": lap_no,
                "type": "safety_car",
                "text": f"Lap ~{lap_no}: Safety Car deployed.",
            })
        elif status == "6":
            key_moments.append({
                "lap": lap_no,
                "type": "vsc",
                "text": f"Lap ~{lap_no}: Virtual Safety Car deployed.",
            })

    # 4. Incidents — only on green laps, and only isolated slow laps
    for i, row in after_reset.iterrows():
        lt = row["LapTimeSec"]
        if pd.isna(lt) or pd.isna(median_lap):
            continue
        if lt <= median_lap * 1.07:
            continue
        # Skip if this lap wasn't green
        if not _lap_is_green(row["LapStart"], row["LapEnd"], status_windows):
            continue
        # Skip if neighbors are also slow (block event, not isolated incident)
        prev_ok = True
        next_ok = True
        if i > 0:
            prev_lt = after_reset.iloc[i - 1]["LapTimeSec"]
            prev_ok = pd.isna(prev_lt) or abs(prev_lt - median_lap) < median_lap * 0.1
        if i < len(after_reset) - 1:
            next_lt = after_reset.iloc[i + 1]["LapTimeSec"]
            next_ok = pd.isna(next_lt) or abs(next_lt - median_lap) < median_lap * 0.1
        if not (prev_ok and next_ok):
            continue

        key_moments.append({
            "lap": int(row["LapNumber"]),
            "type": "incident",
            "text": f"Lap {int(row['LapNumber'])}: Slow lap "
                    f"({lt:.1f}s vs median {median_lap:.1f}s) — possible incident.",
        })

    # Deduplicate key moments by (lap, type, text prefix)
    seen = set()
    deduped = []
    for m in key_moments:
        key = (m["lap"], m["type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    key_moments = sorted(deduped, key=lambda m: (m["lap"] if m["lap"] is not None else 0))

    # --- Summary string ---
    summary_lines = [
        f"Race: {session.event['EventName']} {session.event.year}",
        f"Driver: {driver}",
        f"User watched up to lap {from_lap}.",
        f"Laps after cutoff: {len(after)}",
        f"Median lap time after cutoff: {median_lap:.2f}s",
    ]

    # Position AT cutoff (not first after)
    pos_at_cutoff = laps[laps["LapNumber"] == from_lap]["Position"].dropna()
    if not pos_at_cutoff.empty:
        start_pos = pos_at_cutoff.iloc[0]
    elif not after_reset["Position"].dropna().empty:
        start_pos = after_reset["Position"].dropna().iloc[0]
    else:
        start_pos = None

    if "Position" in after_reset.columns and not after_reset["Position"].dropna().empty:
        end_pos = after_reset["Position"].dropna().iloc[-1]
        if start_pos is not None:
            summary_lines.append(
                f"Position: P{int(start_pos)} at lap {from_lap} → P{int(end_pos)} at finish"
            )
        else:
            summary_lines.append(f"Finished P{int(end_pos)}")

    if "Compound" in after_reset.columns:
        compounds = after_reset["Compound"].dropna().unique().tolist()
        summary_lines.append(f"Tyres used after cutoff: {', '.join(compounds)}")

    if red_flag_added:
        summary_lines.append("Note: race was RED FLAGGED — this explains "
                             "the gap in lap times and is not a driver incident.")

    summary_lines.append("Key moments:")
    for m in key_moments:
        summary_lines.append(f"- {m['text']}")

    summary = "\n".join(summary_lines)

    return {
        "driver": driver,
        "from_lap": from_lap,
        "laps": after,
        "key_moments": key_moments,
        "summary": summary,
        "session": session,
    }


# --- Quick test ---
if __name__ == "__main__":
    print("Loading São Paulo 2024 race...")
    session = load_session(2024, "São Paulo", "R")
    print("Drivers:", get_drivers(session))

    driver = "VER"
    from_lap = 10

    data = build_recap_data(session, driver, from_lap)

    print("\n=== SUMMARY FOR LLM ===")
    print(data["summary"])

    print("\n=== KEY MOMENTS ===")
    for m in data["key_moments"]:
        print(f"[{m['type']}] {m['text']}")

    print("\n=== LAP DATA (first 5 rows after cutoff) ===")
    print(data["laps"][["LapNumber", "LapTimeSec", "Compound", "TyreLife", "Position"]].head())