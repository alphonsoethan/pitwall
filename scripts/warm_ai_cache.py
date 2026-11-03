# scripts/warm_ai_cache.py
"""
Pre-generate the recap and verdict texts and save them to data/ai_cache.json.
Commit that file and reviewers get instant answers that never touch your Gemini quota.

    GEMINI_API_KEY=your-key python scripts/warm_ai_cache.py
    GEMINI_API_KEY=your-key python scripts/warm_ai_cache.py --drivers VER,NOR,LEC --laps 0,20,40

--drivers must match what your Recap page dropdown sends (the values in DRIVER_CHOICES).
Calls are spaced out automatically, so ~30 texts take a few minutes.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gemini_client as g  # noqa: E402
from data_store import load_session  # noqa: E402
from race_data import build_race_summary, build_recap_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", help="comma-separated, same format as DRIVER_CHOICES")
    ap.add_argument("--laps", default="0,25,50", help="comma-separated 'I left at lap' values")
    args = ap.parse_args()

    session = load_session()
    laps = [int(x) for x in args.laps.split(",")]

    if args.drivers:
        drivers = [d.strip() for d in args.drivers.split(",")]
    else:
        from config import DRIVER_CHOICES
        drivers = list(DRIVER_CHOICES)[:5]

    ok = failed = 0

    def check(text, label):
        nonlocal ok, failed
        if text.startswith("⚠️") or text.startswith("Gemini API key"):
            failed += 1
            print(f"  FAILED  {label}: {text.splitlines()[0]}")
        else:
            ok += 1
            print(f"  saved   {label}")

    for from_lap in laps:
        summary = build_race_summary(session, from_lap)["summary"]
        check(g.get_race_recap(summary), f"whole race, lap {from_lap}: recap")
        check(g.get_race_verdict(summary), f"whole race, lap {from_lap}: verdict")
        for d in drivers:
            summary = build_recap_data(session, d, from_lap)["summary"]
            check(g.get_recap(summary), f"{d}, lap {from_lap}: recap")
            check(g.get_verdict(summary), f"{d}, lap {from_lap}: verdict")

    print(f"\nDone: {ok} saved, {failed} failed. Commit data/ai_cache.json.")
    if failed:
        print("Re-run the script to fill in the failures, it skips anything already saved.")


if __name__ == "__main__":
    main()