"""Leakage-free reporting from predictions frozen before kickoff."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from contextlib import closing


OUTCOMES = {"MANDANTE", "EMPATE", "VISITANTE"}


def operational_snapshot_report(db_path: str) -> dict:
    """Score issued snapshots and their original four-leg ticket membership."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)) as conn:
        rows = conn.execute("""
            SELECT s.match_id,s.predicted_outcome,m.actual_outcome,s.source_version
            FROM ml_prediction_snapshots s
            JOIN match_postmortems m ON m.match_id=s.match_id
            WHERE m.actual_outcome IN ('MANDANTE','EMPATE','VISITANTE')
              AND s.predicted_outcome IN ('MANDANTE','EMPATE','VISITANTE')
              AND s.start_timestamp>0 AND s.captured_at<s.start_timestamp
        """).fetchall()
        ticket_rows = conn.execute("""
            SELECT p.radar_run_id,p.ticket_id,p.match_id,
                   s.predicted_outcome,m.actual_outcome
            FROM previsoes p
            JOIN ml_prediction_snapshots s ON s.match_id=p.match_id
            JOIN match_postmortems m ON m.match_id=p.match_id
            WHERE p.radar_run_id IS NOT NULL AND TRIM(p.radar_run_id)!=''
              AND p.ticket_id IS NOT NULL AND TRIM(p.ticket_id)!=''
              AND UPPER(TRIM(p.ticket_id))!='SEM_TICKET'
              AND m.actual_outcome IN ('MANDANTE','EMPATE','VISITANTE')
              AND s.predicted_outcome IN ('MANDANTE','EMPATE','VISITANTE')
              AND s.start_timestamp>0 AND s.captured_at<s.start_timestamp
            ORDER BY p.id
        """).fetchall()

    correct = sum(predicted == actual for _, predicted, actual, _ in rows)
    actual_counts = Counter(actual for _, _, actual, _ in rows)
    predicted_counts = Counter(predicted for _, predicted, _, _ in rows)
    recalls = {
        outcome: (
            sum(predicted == actual == outcome for _, predicted, actual, _ in rows)
            / actual_counts[outcome]
            if actual_counts[outcome] else 0.0
        )
        for outcome in sorted(OUTCOMES)
    }

    # A rerun can persist the same leg more than once. It must count once in
    # its issued ticket, while a leg legitimately reused in another ticket is
    # kept in both groups.
    grouped = defaultdict(dict)
    for run_id, ticket_id, match_id, predicted, actual in ticket_rows:
        grouped[(str(run_id), str(ticket_id))][str(match_id)] = (
            str(predicted), str(actual)
        )
    distributions = Counter()
    complete = green = 0
    for legs in grouped.values():
        if len(legs) != 4:
            continue
        hits = sum(predicted == actual for predicted, actual in legs.values())
        complete += 1
        green += int(hits == 4)
        distributions[hits] += 1
    return {
        "resolved_predictions": len(rows),
        "correct_predictions": int(correct),
        "individual_accuracy": (correct / len(rows)) if rows else 0.0,
        "actual_counts": dict(actual_counts),
        "predicted_counts": dict(predicted_counts),
        "recall_by_outcome": recalls,
        "complete_tickets": complete,
        "green_tickets": green,
        "ticket_green_rate": (green / complete) if complete else 0.0,
        "ticket_hits_distribution": {
            str(hits): int(distributions.get(hits, 0)) for hits in range(5)
        },
    }
