# scripts/preload_data.py
"""
Run ONCE on your own machine, then commit the generated data/ folder:

    python scripts/preload_data.py

This downloads the race with FastF1 and saves laps, results, weather and track
status as parquet files, so the deployed app never has to call the F1 servers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fastf1  # noqa: E402

from data_store import (  # noqa: E402
    DATA_DIR, RACE_EVENT, RACE_SESSION, RACE_YEAR, export_session,
)


def main():
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    print(f"Loading {RACE_YEAR} {RACE_EVENT} ({RACE_SESSION}) ...")
    session = fastf1.get_session(RACE_YEAR, RACE_EVENT, RACE_SESSION)
    session.load(telemetry=False, weather=True, messages=False)

    if session.laps is None or len(session.laps) == 0:
        raise SystemExit("No lap data came back. Try again in a minute.")

    counts = export_session(session, DATA_DIR)
    print(f"Saved to {DATA_DIR}")
    for name, n in counts.items():
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()