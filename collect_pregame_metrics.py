"""Bounded enrichment of upcoming games already in the DB; no radar or Telegram."""
import argparse
import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from pregame_metric_enrichment import collect_recent_metric_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="ia_sports_v5.db")
    parser.add_argument("--max-lookups", type=int, default=8)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new report path")
    with closing(sqlite3.connect(Path(args.db).resolve().as_uri()+"?mode=ro", uri=True)) as conn:
        games = [{"ID": mid, "Timestamp": timestamp} for mid, timestamp in conn.execute(
            "SELECT match_id,start_timestamp FROM ml_prediction_snapshots WHERE start_timestamp>? ORDER BY start_timestamp",
            (int(time.time()),))]
    fallback = None
    if not args.no_fallback:
        # Import does not start the scheduler/menu; only the existing limited
        # API client is used. No new credentials or additional subscriptions.
        from robo_auto import RAPIDAPI_HOST, safe_api_get
        from allsports_api import fetch_allsports_postmatch_resources
        fallback = lambda mid, resources: fetch_allsports_postmatch_resources(
            RAPIDAPI_HOST, mid, resources, safe_api_get)
    print(f"Pregame enrichment: {len(games)} upcoming targets; at most {args.max_lookups} prior-fixture lookups", flush=True)
    summary = collect_recent_metric_profiles(args.db, games, fallback=fallback,
                                             max_lookups=args.max_lookups)
    report = {"collected_at": datetime.now(timezone.utc).isoformat(),
              "upcoming_targets": len(games), "limit_prior_fixtures": args.max_lookups,
              "note": "Existing picks/snapshots were not changed; new data is only available from this capture onward",
              "summary": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
