# data_store.py
"""
Loads the race from pre-exported parquet files so the app starts fast and never
depends on the F1 timing servers at runtime.

Normal flow:   python scripts/preload_data.py   (run once, commit the data/ folder)
Safety net:    if the files are missing, the app downloads the race once via FastF1
               and writes the files, so the next start is fast.

This module deliberately has no Streamlit import so it can be tested on its own.
"""
from pathlib import Path

import pandas as pd

RACE_YEAR = 2024
RACE_EVENT = "São Paulo"
RACE_SESSION = "R"
DATA_DIR = Path(__file__).resolve().parent / "data" / "sao_paulo_2024"

# Only the columns the app actually uses, which keeps files small and avoids
# parquet trouble with FastF1's mixed-type columns.
LAP_COLS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Time", "Stint",
    "Compound", "TyreLife", "Position", "PitInTime", "PitOutTime", "TrackStatus",
    "IsAccurate",
]
RESULT_COLS = [
    "Abbreviation", "FullName", "DriverNumber", "TeamName", "TeamColor",
    "Position", "ClassifiedPosition", "GridPosition", "Status", "Points",
]
WEATHER_COLS = ["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]
TRACK_COLS = ["Time", "Status", "Message"]

FILES = {
    "laps": "laps.parquet",
    "results": "results.parquet",
    "weather": "weather.parquet",
    "track_status": "track_status.parquet",
}


class CachedSession:
    """Minimal stand-in for a FastF1 Session: same attribute names the app reads."""

    def __init__(self, laps, results, weather_data, track_status):
        laps = laps.dropna(subset=["LapNumber"]).copy()
        laps["LapNumber"] = laps["LapNumber"].astype(int)
        self.laps = laps
        self.results = results
        self.weather_data = weather_data
        self.track_status = track_status

    @property
    def total_laps(self):
        return int(self.laps["LapNumber"].max())

    @property
    def event(self):
        """Mimics FastF1's session.event: dict-style AND attribute-style access."""
        class _Event(dict):
            @property
            def year(self):
                return self.get("year")
        return _Event({"EventName": RACE_EVENT, "year": RACE_YEAR})

def _subset(df, cols):
    if df is None:
        return pd.DataFrame(columns=cols)
    out = df[[c for c in cols if c in df.columns]].copy()
    # Mixed-type object columns break parquet, so make every object column text.
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: None if pd.isna(v) else str(v))
    return out


def export_session(session, data_dir=DATA_DIR):
    """Write the four tables of a loaded FastF1 session to parquet files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "laps": _subset(session.laps, LAP_COLS),
        "results": _subset(session.results, RESULT_COLS),
        "weather": _subset(getattr(session, "weather_data", None), WEATHER_COLS),
        "track_status": _subset(getattr(session, "track_status", None), TRACK_COLS),
    }
    for key, df in tables.items():
        df.to_parquet(data_dir / FILES[key], index=False)
    return {k: len(v) for k, v in tables.items()}


def data_available(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    return all((data_dir / f).exists() for f in FILES.values())


def load_from_disk(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    read = lambda key: pd.read_parquet(data_dir / FILES[key])
    return CachedSession(
        laps=read("laps"),
        results=read("results"),
        weather_data=read("weather"),
        track_status=read("track_status"),
    )


def load_from_fastf1(data_dir=DATA_DIR):
    """Slow path: download the race, then save it so this only ever happens once."""
    import fastf1

    session = fastf1.get_session(RACE_YEAR, RACE_EVENT, RACE_SESSION)
    session.load(telemetry=False, weather=True, messages=False)
    if session.laps is None or len(session.laps) == 0:
        raise RuntimeError(
            "F1 timing data came back empty. Run scripts/preload_data.py locally "
            "and commit the data/ folder."
        )
    try:
        export_session(session, data_dir)
    except Exception:
        pass  # read-only filesystem is fine, we still have the data in memory
    return CachedSession(
        laps=_subset(session.laps, LAP_COLS),
        results=_subset(session.results, RESULT_COLS),
        weather_data=_subset(session.weather_data, WEATHER_COLS),
        track_status=_subset(session.track_status, TRACK_COLS),
    )


def load_session(data_dir=DATA_DIR):
    if data_available(data_dir):
        return load_from_disk(data_dir)
    return load_from_fastf1(data_dir)