"""Fill real statistics for previous fixtures before a new radar predicts.

Only finished fixtures already listed in the pregame recent-form snapshot are
eligible. Current fixture statistics are never queried. Shared persistent cache
and a bounded per-radar fixture-lookup budget avoid a history-wide crawl.
"""
from __future__ import annotations

import json
import os
import logging
import time
from contextlib import closing
from collections import Counter

from sofascore_intelligence import (
    init_sofascore_db, _connect, _fetch_resource, _statistics_metrics,
    _save_team_profiles, _dominance,
)

DEFAULT_MAX_LOOKUPS = max(0, int(os.getenv("PREGAME_STATS_MAX_LOOKUPS", "40")))


def collect_recent_metric_profiles(db_path, games, fallback=None, max_lookups=DEFAULT_MAX_LOOKUPS,
                                  recent_per_team=5):
    init_sofascore_db(db_path)
    now = int(time.time())
    candidates, seen = [], set()
    summary = Counter()
    with closing(_connect(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS pregame_metric_attempts (
            match_id TEXT PRIMARY KEY, attempted_at INTEGER NOT NULL,
            status TEXT NOT NULL, data_coverage REAL NOT NULL DEFAULT 0)""")
        conn.commit()
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pregame_recent_form_snapshots'").fetchone():
            return {"missing_recent_snapshots": 1}
        for game in games:
            target = str(game.get("ID") or game.get("match_id") or "")
            kickoff = int(game.get("Timestamp") or game.get("start_timestamp") or 0)
            if not target or kickoff <= now:
                continue
            for side, provider, raw in conn.execute(
                "SELECT side,provider,games_json FROM pregame_recent_form_snapshots WHERE match_id=?",
                (target,),
            ).fetchall():
                # Only the two providers with fixture IDs in the same verified
                # Sofa/AllSports namespace are accepted here.
                if provider not in {"sofascore", "allsports"}:
                    summary["unknown_provider"] += 1; continue
                try:
                    history = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(history, list):
                    continue
                history = [row for row in history if isinstance(row, dict)
                           and isinstance(row.get("start_timestamp"), (float, int))]
                for rank, previous in enumerate(sorted(history, key=lambda r: r.get("start_timestamp",0), reverse=True)[:recent_per_team]):
                    mid = str(previous.get("match_id") or "")
                    start = int(previous.get("start_timestamp") or 0)
                    if not mid or mid == target or start <= 0 or start+3*3600 >= min(now,kickoff):
                        summary["unsafe_fixture_skipped"] += 1; continue
                    if mid in seen:
                        continue
                    seen.add(mid)
                    if (not previous.get("team_name") or not previous.get("opponent_name")
                            or previous.get("goals_for") is None or previous.get("goals_against") is None):
                        summary["incomplete_identity_or_score"] += 1; continue
                    # Preserve already measured profiles, including real zeros.
                    profiles = conn.execute("SELECT xg_for,xg_against,metric_presence_json FROM sofascore_team_match_profiles WHERE match_id=?", (mid,)).fetchall()
                    def measured_xg(row):
                        try:
                            presence = json.loads(row[2]) if row[2] else []
                        except (ValueError, TypeError):
                            presence = []
                        return ({"xg_for", "xg_against"}.issubset(presence)
                                or (row[0] is not None and row[1] is not None
                                    and max(row[0], row[1]) > 0))
                    if len(profiles) == 2 and all(measured_xg(row) for row in profiles):
                        summary["profile_cache_hits"] += 1; continue
                    last = conn.execute("SELECT attempted_at,status FROM pregame_metric_attempts WHERE match_id=?",(mid,)).fetchone()
                    ttl = 365*86400 if last and last[1] == "measured" else 86400
                    if last and now - int(last[0]) < ttl:
                        summary["profile_cache_hits"] += 1; continue
                    candidates.append((rank, previous))
    for rank, previous in sorted(candidates, key=lambda item: (item[0], -item[1]["start_timestamp"])):
        if summary["lookups"] >= max(0, int(max_lookups)):
            summary["deferred"] += 1; continue
        mid = str(previous["match_id"])
        summary["lookups"] += 1
        try:
            payload, hit, status = _fetch_resource(db_path, mid, "statistics", 365*86400)
        except Exception:
            logging.warning("Estatísticas pré-jogo: fonte SofaScore indisponível para %s", mid)
            payload, hit, status = None, False, 0
            summary["source_errors"] += 1
        summary["sofa_cache_hits"] += int(hit)
        metrics, coverage = _statistics_metrics(payload)
        if not metrics or coverage < .3:
            try:
                alternative = fallback(mid, ["statistics"]) if fallback else None
            except Exception:
                logging.warning("Estatísticas pré-jogo: fallback indisponível para %s", mid)
                alternative = None
                summary["source_errors"] += 1
            alternative = alternative or {}
            alt_metrics, alt_coverage = _statistics_metrics((alternative.get("resources") or {}).get("statistics"))
            summary["allsports_http"] += int(alternative.get("http_requests", 0))
            summary["allsports_cache_hits"] += int(alternative.get("cache_hits", 0))
            # Complement coverage. A shots-only fallback must not erase xG
            # already reported by the first source, even if its coverage is higher.
            for key, present in alt_metrics.items():
                if key.endswith("_available") and present and not metrics.get(key):
                    value_key = key[:-len("_available")]
                    metrics[value_key] = alt_metrics[value_key]
                    metrics[key] = 1.0
            coverage = sum(value for key, value in metrics.items() if key.endswith("_available")) / 20
        has_metrics = any(metrics.get(key, 0) for key in (
            "home_xg_available", "away_xg_available", "home_shots_on_target_available",
            "away_shots_on_target_available", "home_big_chances_available", "away_big_chances_available"))
        if has_metrics:
            own = {"id": previous.get("team_id"), "name": previous.get("team_name")}
            opponent = {"id": previous.get("opponent_id"), "name": previous.get("opponent_name")}
            is_home = bool(previous.get("is_home"))
            gf, ga = previous.get("goals_for"), previous.get("goals_against")
            event = {
                "id": mid, "startTimestamp": previous["start_timestamp"], "status": {"type": "finished"},
                "homeTeam": own if is_home else opponent, "awayTeam": opponent if is_home else own,
                "homeScore": {"normaltime": gf if is_home else ga},
                "awayScore": {"normaltime": ga if is_home else gf},
            }
            _save_team_profiles(db_path, mid, event, metrics, _dominance(metrics)[0])
            summary["measured_matches"] += 1
            summary["xg_matches"] += int(bool(metrics.get("home_xg_available") and metrics.get("away_xg_available")))
        else:
            summary["unavailable"] += 1
        with closing(_connect(db_path)) as conn:
            conn.execute("INSERT OR REPLACE INTO pregame_metric_attempts VALUES (?,?,?,?)",
                         (mid, now, "measured" if has_metrics else "unavailable", coverage))
            conn.commit()
    return dict(summary)


def enrich_recent_metrics_safely(db_path, games, **kwargs):
    """An optional source/database failure must not discard radar games."""
    try:
        return collect_recent_metric_profiles(db_path, games, **kwargs)
    except Exception:
        logging.exception("Enriquecimento estatístico indisponível; radar continuará com os dados existentes")
        return {"collection_failed": 1}
