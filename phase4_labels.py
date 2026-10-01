"""Blind outcome ingestion for the prospective Phase 4 holdout.

This module may persist final labels, but it never compares them with a pick,
computes a metric, or exposes class/score values in its returned summary.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from phase2_observations import append
from soccer_football_info import SoccerFootballInfoClient


FINISHED_STATUSES = {"ENDED", "FINISHED", "FT", "AFTER_PENALTIES", "AFTER_EXTRA_TIME"}


def _event_id(event):
    return event.get("id") or event.get("match_id") or event.get("matchId")


def _full_time_score(team):
    if not isinstance(team, dict):
        return None
    score = team.get("score")
    if isinstance(score, dict):
        for key in ("f", "ft", "full", "fullTime", "current"):
            if key in score:
                score = score[key]
                break
    try:
        value = int(score)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _actual_outcome(home_score, away_score):
    if home_score > away_score:
        return "HOME"
    if away_score > home_score:
        return "AWAY"
    return "DRAW"


def _load_pending(sidecar, now, grace_seconds):
    if not sidecar.exists():
        return []
    with closing(sqlite3.connect(sidecar, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "observations" not in tables:
            return []
        resolved = {
            (row[0], row[1]) for row in conn.execute(
                "SELECT run_id,match_id FROM observations WHERE stage='phase4_outcome_v1'"
            )
        }
        rows = conn.execute(
            """SELECT run_id,match_id,payload FROM observations
               WHERE stage='phase4_prefilter_v1' ORDER BY observed_at"""
        ).fetchall()
    pending = []
    for row in rows:
        key = (str(row["run_id"]), str(row["match_id"]))
        if key in resolved:
            continue
        try:
            payload = json.loads(row["payload"])
            kickoff = float(payload.get("event_time") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        source_id = payload.get("source_event_id")
        if (not source_id or not kickoff or kickoff + grace_seconds > now
                or not payload.get("available_before_decision")):
            continue
        pending.append({
            "run_id": key[0], "match_id": key[1], "kickoff": kickoff,
            "source_event_id": str(source_id),
            "eligibility_group": payload.get("eligibility_group"),
            "provider": payload.get("provider"),
        })
    return pending


def ingest_blind_outcomes(db_path, *, client=None, now=None, grace_seconds=3 * 3600,
                          max_days=2):
    """Store final labels and return counts/request health only.

    The return object deliberately contains neither scores nor outcome classes.
    Exact provider IDs are mandatory; there is no fuzzy team-name fallback.
    """
    now = float(now if now is not None else time.time())
    sidecar = Path(db_path).resolve().with_name("phase2_observations.db")
    pending = _load_pending(sidecar, now, int(grace_seconds))
    grouped = defaultdict(list)
    for item in pending:
        day = datetime.fromtimestamp(item["kickoff"], tz=timezone.utc).date()
        grouped[day].append(item)
    selected_days = sorted(grouped)[:max(0, int(max_days))]
    if not selected_days:
        return {"resolved_new": 0, "pending_candidates": len(pending),
                "days_checked": 0, "http_requests": 0, "pages_missing": 0,
                "status": "IDLE"}

    client = client or SoccerFootballInfoClient(str(db_path))
    resolved_new = 0
    pages_missing = 0
    incomplete_days = 0
    http_before = int(getattr(client, "http_requests", 0))
    for utc_day in selected_days:
        events, meta = client.fetch_day(utc_day, force_refresh=True)
        pages_missing += int(meta.get("pages_missing", 0) or 0)
        incomplete_days += int(not bool(meta.get("complete")))
        by_id = {str(_event_id(event)): event for event in events if _event_id(event) is not None}
        for item in grouped[utc_day]:
            event = by_id.get(item["source_event_id"])
            if not isinstance(event, dict):
                continue
            status = str(event.get("status") or "").strip().upper()
            if status not in FINISHED_STATUSES:
                continue
            home_score = _full_time_score(event.get("teamA"))
            away_score = _full_time_score(event.get("teamB"))
            if home_score is None or away_score is None:
                continue
            saved = append(db_path, item["run_id"], item["match_id"],
                           "phase4_outcome_v1", {
                "label_version": "phase4-blind-label-v1",
                "source": "soccer-football-info",
                "source_event_id": item["source_event_id"],
                "source_status": status,
                "eligibility_group": item["eligibility_group"],
                "home_score": home_score,
                "away_score": away_score,
                "actual_outcome": _actual_outcome(home_score, away_score),
                "resolved_at": now,
                "performance_analysis_opened": False,
            })
            resolved_new += int(bool(saved))

    http_requests = int(getattr(client, "http_requests", 0)) - http_before
    return {
        "resolved_new": resolved_new,
        "pending_candidates": len(pending),
        "days_checked": len(selected_days),
        "http_requests": max(0, http_requests),
        "pages_missing": pages_missing,
        "incomplete_days": incomplete_days,
        "status": "PASS" if not incomplete_days else "PARTIAL",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão cega de labels da Fase 4B")
    parser.add_argument("--db", default="ia_sports_v5.db")
    parser.add_argument("--max-days", type=int, default=2)
    args = parser.parse_args()
    # A saída é deliberadamente limitada a volume/saúde.
    print(json.dumps(ingest_blind_outcomes(args.db, max_days=args.max_days),
                     ensure_ascii=False, sort_keys=True))
