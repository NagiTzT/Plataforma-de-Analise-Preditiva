"""Chronological opponent-strength values shared by training and inference."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from historical_identity import resolve_training_team_name
from soccer_football_info import normalize_team_name


@dataclass(frozen=True)
class _CachedIndex:
    signature: tuple[Any, ...]
    by_match_id: dict[str, tuple[float, float]]
    team_ratings: dict[str, float]
    checked_at: float


_cache: dict[str, _CachedIndex] = {}
_lock = threading.RLock()


def _database_key(conn) -> str:
    for _, name, path in conn.execute("PRAGMA database_list"):
        if name == "main":
            return str(path or id(conn))
    return str(id(conn))


def _signature(conn) -> tuple[Any, ...]:
    return tuple(conn.execute(
        """SELECT COUNT(*), COALESCE(MAX(rowid),0), COALESCE(MAX(data_jogo),''),
                  COALESCE(SUM(home_score),0), COALESCE(SUM(away_score),0),
                  COALESCE(SUM(home_score*home_score + away_score*away_score),0)
           FROM training_data"""
    ).fetchone())


def _ppg(state: list[float] | None, prior: float = 1.35) -> float:
    games, points = state or (0.0, 0.0)
    return (float(points) + prior * 8.0) / (float(games) + 8.0)


def _build(conn) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    rows = conn.execute(
        """SELECT match_id,home_team,away_team,home_score,away_score
           FROM training_data
           ORDER BY data_jogo ASC, match_id ASC"""
    ).fetchall()
    names = {str(value) for row in rows for value in row[1:3] if value}
    identities = {
        name: resolve_training_team_name(conn, name)[0]
        for name in names
    }
    states: dict[str, list[float]] = {}
    ratings: dict[str, float] = {}
    result: dict[str, tuple[float, float]] = {}
    for match_id, raw_home, raw_away, home_score, away_score in rows:
        home = identities.get(str(raw_home), raw_home)
        away = identities.get(str(raw_away), raw_away)
        # Um confronto do time contra ele próprio indica erro de identidade da
        # fonte. Não pode atualizar forma, força do adversário nem Elo.
        if (not home or not away
                or normalize_team_name(home) == normalize_team_name(away)):
            continue
        result[str(match_id)] = (_ppg(states.get(away)), _ppg(states.get(home)))
        try:
            hs, aws = float(home_score), float(away_score)
        except (TypeError, ValueError):
            continue
        if hs > aws:
            home_points, away_points = 3.0, 0.0
        elif hs == aws:
            home_points = away_points = 1.0
        else:
            home_points, away_points = 0.0, 3.0
        for team, points in ((home, home_points), (away, away_points)):
            state = states.setdefault(team, [0.0, 0.0])
            state[0] += 1.0
            state[1] += points
        elo_home = float(ratings.get(home, 1500.0))
        elo_away = float(ratings.get(away, 1500.0))
        expected_home = 1.0 / (
            1.0 + 10 ** ((elo_away - (elo_home + 70.0)) / 400.0)
        )
        actual_home = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
        margin = max(1.0, math.log1p(abs(hs - aws)))
        ratings[home] = elo_home + 24.0 * margin * (actual_home - expected_home)
        ratings[away] = elo_away + 24.0 * margin * (
            (1.0 - actual_home) - (1.0 - expected_home)
        )
    return result, ratings


def historical_opponent_strengths(conn) -> dict[str, tuple[float, float]]:
    """Map match id to (home-opponent PPG, away-opponent PPG), pre-match."""
    key = _database_key(conn)
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached.checked_at <= 60.0:
            return cached.by_match_id
        signature = _signature(conn)
        if cached and cached.signature == signature:
            _cache[key] = _CachedIndex(
                cached.signature, cached.by_match_id, cached.team_ratings,
                time.monotonic(),
            )
            return cached.by_match_id
        built, ratings = _build(conn)
        _cache[key] = _CachedIndex(signature, built, ratings, time.monotonic())
        return built


def current_elo_ratings(conn) -> dict[str, float]:
    """Return Elo through the latest audited result, with training semantics."""
    key = _database_key(conn)
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached.checked_at <= 60.0:
            return dict(cached.team_ratings)
        signature = _signature(conn)
        if not cached or cached.signature != signature:
            built, ratings = _build(conn)
            cached = _CachedIndex(signature, built, ratings, time.monotonic())
            _cache[key] = cached
        else:
            cached = _CachedIndex(
                cached.signature, cached.by_match_id, cached.team_ratings,
                time.monotonic(),
            )
            _cache[key] = cached
        return dict(cached.team_ratings)


def clear_opponent_strength_cache() -> None:
    with _lock:
        _cache.clear()
