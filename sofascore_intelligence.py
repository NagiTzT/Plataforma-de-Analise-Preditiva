"""Contexto pré-jogo e autópsia pós-jogo com dados públicos do SofaScore.

O módulo mantém uma fronteira temporal rígida:

* ``capture_pregame_contexts`` grava somente informações disponíveis antes do
  início da partida;
* ``audit_match_postmortem`` grava estatísticas finais em tabelas separadas;
* estatísticas finais nunca são devolvidas por ``get_pregame_features`` para a
  própria partida. Elas só alimentam o perfil móvel de jogos futuros.

O endpoint usado pelo site não é uma API contratada. Por isso toda chamada é
best-effort, limitada, cacheada e incapaz de interromper o radar/auditoria.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter
from contextlib import closing
from datetime import datetime
from typing import Any, Callable, Iterable

from soccer_football_info import team_name_similarity

try:
    from curl_cffi import requests as browser_requests
except ImportError:  # pragma: no cover - fallback documentado e testado indiretamente
    import requests as browser_requests


SOFASCORE_BASE_URL = os.getenv(
    "SOFASCORE_BASE_URL", "https://www.sofascore.com/api/v1"
).rstrip("/")
SOFASCORE_ENABLED = os.getenv("SOFASCORE_ENABLED", "1") == "1"
SOFASCORE_RADAR_ENABLED = os.getenv("SOFASCORE_RADAR_ENABLED", "1") == "1"
SOFASCORE_AUDIT_ENABLED = os.getenv("SOFASCORE_AUDIT_ENABLED", "1") == "1"
SOFASCORE_MIN_INTERVAL_SECONDS = max(
    0.25, float(os.getenv("SOFASCORE_MIN_INTERVAL_SECONDS", "0.35"))
)
SOFASCORE_TIMEOUT_SECONDS = max(
    5.0, float(os.getenv("SOFASCORE_TIMEOUT_SECONDS", "18"))
)
SOFASCORE_CONTEXT_BLEND = max(
    0.0, min(0.45, float(os.getenv("SOFASCORE_CONTEXT_BLEND", "0.30")))
)
LIVE_RECENT_TTL_SECONDS = max(
    1800, min(12 * 3600, int(os.getenv("LIVE_RECENT_TTL_SECONDS", str(4 * 3600))))
)
LIVE_RECENT_MATCHES = max(
    3, min(10, int(os.getenv("LIVE_RECENT_MATCHES", "10")))
)
ANALYSIS_VERSION = "sofa-context-v10-goal-profile"
# Perfil operacional escolhido após os replays v5 x v8. O algoritmo v8
# permanece disponível para testes em sombra; radar e app usam este perfil.
ACTIVE_ANALYSIS_PROFILE = "active-v5"
ACTIVE_ANALYSIS_VERSION = "sofa-context-v5"

_schema_lock = threading.RLock()
_initialized: set[str] = set()
_rate_lock = threading.Lock()
_last_request_at = 0.0
_circuit_lock = threading.Lock()
_sofa_blocked_until = 0.0
_consecutive_blocked = 0


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_sofascore_db(db_path: str) -> None:
    """Cria tabelas isoladas para contexto, snapshots e autópsias."""
    absolute = os.path.abspath(db_path)
    with _schema_lock:
        if absolute in _initialized:
            return
        with closing(_connect(absolute)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sofascore_http_cache (
                    cache_key TEXT PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    payload_json TEXT,
                    http_status INTEGER,
                    fetched_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sofa_cache_match
                    ON sofascore_http_cache(match_id, resource);

                CREATE TABLE IF NOT EXISTS sofascore_match_links (
                    match_id TEXT PRIMARY KEY,
                    provider_event_id TEXT,
                    home_team_id TEXT,
                    away_team_id TEXT,
                    home_name TEXT,
                    away_name TEXT,
                    start_timestamp INTEGER,
                    validated INTEGER DEFAULT 0,
                    similarity REAL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sofascore_pregame_context (
                    match_id TEXT PRIMARY KEY,
                    captured_at INTEGER NOT NULL,
                    start_timestamp INTEGER,
                    home_name TEXT,
                    away_name TEXT,
                    features_json TEXT NOT NULL,
                    coverage REAL NOT NULL DEFAULT 0,
                    source_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sofascore_historical_backfill (
                    match_id TEXT PRIMARY KEY,
                    processed_at INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    has_pregame_form INTEGER NOT NULL DEFAULT 0,
                    source_version TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sofa_backfill_status
                    ON sofascore_historical_backfill(status, processed_at);

                CREATE TABLE IF NOT EXISTS sofascore_historical_detail_backfill (
                    match_id TEXT PRIMARY KEY,
                    processed_at INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    statistics_coverage REAL NOT NULL DEFAULT 0,
                    source_version TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sofa_detail_backfill_status
                    ON sofascore_historical_detail_backfill(status, processed_at);

                CREATE TABLE IF NOT EXISTS ml_prediction_snapshots (
                    match_id TEXT PRIMARY KEY,
                    captured_at INTEGER NOT NULL,
                    start_timestamp INTEGER,
                    feature_version INTEGER NOT NULL DEFAULT 6,
                    features_json TEXT NOT NULL,
                    base_probabilities_json TEXT,
                    final_probabilities_json TEXT,
                    predicted_outcome TEXT,
                    information_quality REAL,
                    draw_risk REAL,
                    context_conflict REAL,
                    analysis_json TEXT,
                    source_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS match_postmortems (
                    match_id TEXT PRIMARY KEY,
                    audited_at INTEGER NOT NULL,
                    prediction_status TEXT,
                    predicted_outcome TEXT,
                    actual_outcome TEXT,
                    scoreline TEXT,
                    verdict TEXT NOT NULL,
                    process_score REAL,
                    chosen_dominance REAL,
                    opponent_dominance REAL,
                    learning_weight REAL NOT NULL DEFAULT 1,
                    error_margin REAL NOT NULL DEFAULT 0,
                    data_coverage REAL NOT NULL DEFAULT 0,
                    metrics_json TEXT,
                    reasons_json TEXT,
                    source_version TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_postmortem_verdict
                    ON match_postmortems(verdict, audited_at);

                CREATE TABLE IF NOT EXISTS ml_context_source_monitor (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluated_at INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    snapshots INTEGER NOT NULL,
                    resolved INTEGER NOT NULL,
                    correct INTEGER NOT NULL,
                    accuracy REAL,
                    details_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_context_source_monitor_time
                    ON ml_context_source_monitor(source, evaluated_at DESC);

                CREATE TABLE IF NOT EXISTS sofascore_team_match_profiles (
                    match_id TEXT NOT NULL,
                    team_key TEXT NOT NULL,
                    team_id TEXT,
                    team_name TEXT,
                    opponent_key TEXT,
                    start_timestamp INTEGER NOT NULL,
                    is_home INTEGER NOT NULL,
                    goals_for REAL,
                    goals_against REAL,
                    xg_for REAL,
                    xg_against REAL,
                    shots_for REAL,
                    shots_against REAL,
                    shots_on_target_for REAL,
                    shots_on_target_against REAL,
                    big_chances_for REAL,
                    big_chances_against REAL,
                    box_shots_for REAL,
                    box_shots_against REAL,
                    possession REAL,
                    dominance REAL,
                    errors_to_goal REAL,
                    result_points REAL,
                    is_draw INTEGER,
                    captured_at INTEGER NOT NULL,
                    metric_presence_json TEXT,
                    PRIMARY KEY (match_id, team_key)
                );
                CREATE INDEX IF NOT EXISTS idx_sofa_team_profile_time
                    ON sofascore_team_match_profiles(team_key, start_timestamp DESC);

                CREATE TABLE IF NOT EXISTS pregame_recent_form_snapshots (
                    match_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    provider_team_id TEXT,
                    captured_at INTEGER NOT NULL,
                    cutoff_timestamp INTEGER NOT NULL,
                    freshest_match_timestamp INTEGER,
                    games_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    coverage REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (match_id, side)
                );
                CREATE INDEX IF NOT EXISTS idx_recent_form_team_time
                    ON pregame_recent_form_snapshots(
                        provider, provider_team_id, captured_at DESC
                    );
                """
            )
            profile_columns = {
                row[1] for row in conn.execute(
                    "PRAGMA table_info(sofascore_team_match_profiles)"
                )
            }
            if "metric_presence_json" not in profile_columns:
                conn.execute(
                    "ALTER TABLE sofascore_team_match_profiles "
                    "ADD COLUMN metric_presence_json TEXT"
                )
            from pregame_snapshot_store import init_archive
            init_archive(conn)
            conn.commit()
        _initialized.add(absolute)


def normalize_team_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(fc|sc|cf|afc|ac|fk|cd|club|deportivo)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _team_identity_name(value: Any) -> str:
    """Normalização estrita para separar homônimos como Lugano e FC Lugano."""
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _similarity(left: Any, right: Any) -> float:
    # A similaridade compartilhada preserva marcadores de categoria. Assim
    # ``Sunderland U21`` jamais valida um evento do ``Sunderland`` principal.
    return float(team_name_similarity(left, right))


def _safe_timestamp(value: Any, default: int = 0) -> int:
    """Decode provider timestamps without letting malformed JSON abort a job."""
    try:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            return int(default or 0)
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return int(default or 0)


def _fixture_identity_matches(
    expected_home: str,
    expected_away: str,
    expected_start: Any,
    actual_home: str,
    actual_away: str,
    actual_start: Any,
    name_floor: float = 0.68,
    time_tolerance_seconds: int = 12 * 3600,
) -> tuple[bool, float]:
    scores = [
        _similarity(expected_home, actual_home),
        _similarity(expected_away, actual_away),
    ]
    similarity = sum(scores) / 2.0 if expected_home and expected_away else 0.0
    names_ok = bool(expected_home and expected_away and min(scores) >= name_floor)
    expected_ts = _safe_timestamp(expected_start)
    actual_ts = _safe_timestamp(actual_start)
    time_ok = not (expected_ts and actual_ts) or abs(expected_ts - actual_ts) <= time_tolerance_seconds
    return bool(names_ok and time_ok), float(similarity)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", ".").strip()
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _ratio(value: Any) -> float:
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(value or ""))
    if not match or int(match.group(2)) <= 0:
        return 0.0
    return _clamp(int(match.group(1)) / int(match.group(2)))


def _request_json(url: str) -> tuple[int, dict[str, Any] | None, str | None]:
    """Faz uma requisição parecida com navegador; nunca propaga exceção."""
    global _last_request_at, _sofa_blocked_until, _consecutive_blocked
    if not SOFASCORE_ENABLED:
        return 0, None, "integração desativada"
    with _circuit_lock:
        if time.monotonic() < _sofa_blocked_until:
            return 403, None, "circuito SofaScore temporariamente aberto"
    with _rate_lock:
        wait = SOFASCORE_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
    kwargs = {
        "headers": {
            "accept": "application/json, text/plain, */*",
            "referer": "https://www.sofascore.com/",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/136 Safari/537.36"
            ),
        },
        "timeout": SOFASCORE_TIMEOUT_SECONDS,
    }
    # curl_cffi aceita impersonate; requests tradicional não.
    if browser_requests.__name__.startswith("curl_cffi"):
        kwargs["impersonate"] = "chrome"
    last_error = None
    for attempt in range(2):
        try:
            response = browser_requests.get(url, **kwargs)
            status = int(response.status_code)
            if status == 200:
                payload = response.json()
                with _circuit_lock:
                    _consecutive_blocked = 0
                    _sofa_blocked_until = 0.0
                return status, payload if isinstance(payload, dict) else None, None
            last_error = f"HTTP {status}"
            if status in {403, 429}:
                with _circuit_lock:
                    _consecutive_blocked += 1
                    if _consecutive_blocked >= 2:
                        _sofa_blocked_until = time.monotonic() + 15 * 60
            else:
                with _circuit_lock:
                    _consecutive_blocked = 0
            if status not in {429, 500, 502, 503, 504}:
                return status, None, last_error
        except Exception as exc:  # rede/JSON/proteção do site
            status = 0
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt == 0:
            time.sleep(1.0)
    return status, None, last_error


def _resource_path(match_id: str, resource: str) -> str:
    suffix = {
        "event": "",
        "statistics": "/statistics",
        "shotmap": "/shotmap",
        "graph": "/graph",
        "lineups": "/lineups",
        "average-positions": "/average-positions",
        "pregame-form": "/pregame-form",
        "team-streaks": "/team-streaks",
        "h2h": "/h2h/events",
        "incidents": "/incidents",
    }[resource]
    return f"/event/{match_id}{suffix}"


def _fetch_resource(
    db_path: str,
    match_id: Any,
    resource: str,
    ttl_seconds: int,
    force_refresh: bool = False,
) -> tuple[dict[str, Any] | None, bool, int]:
    """Retorna (payload, cache_hit, http_status). Cacheia inclusive 404."""
    init_sofascore_db(db_path)
    match_id = str(match_id or "")
    if not match_id or resource not in {
        "event", "statistics", "shotmap", "graph", "lineups", "average-positions",
        "pregame-form", "team-streaks", "h2h", "incidents",
    }:
        return None, False, 0
    now = int(time.time())
    key = f"{match_id}:{resource}"
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT payload_json, http_status, expires_at FROM sofascore_http_cache WHERE cache_key=?",
            (key,),
        ).fetchone()
    if row and not force_refresh and int(row[2] or 0) > now:
        try:
            payload = json.loads(row[0]) if row[0] else None
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        status = int(row[1] or 0)
        # O corpo antigo é preservado no disco após 403/429 para diagnóstico,
        # mas nunca é devolvido como se fosse a resposta nova.
        if status != 200:
            return None, True, status
        return payload if isinstance(payload, dict) else None, True, status

    status, payload, error = _request_json(SOFASCORE_BASE_URL + _resource_path(match_id, resource))
    # Falhas transitórias duram pouco; 404 fica seis horas para não martelar ligas sem cobertura.
    if status == 200:
        expires = now + max(300, int(ttl_seconds))
    elif status == 404:
        expires = now + 6 * 3600
    else:
        expires = now + 15 * 60
    with closing(_connect(db_path)) as conn:
        # Uma falha transitória (403/429/rede) não pode apagar uma resposta
        # válida já coletada. O status/TTL ainda impedem que conteúdo vencido
        # seja tratado como uma resposta HTTP nova.
        conn.execute(
            """INSERT INTO sofascore_http_cache
               (cache_key, match_id, resource, payload_json, http_status,
                fetched_at, expires_at, last_error) VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 match_id=excluded.match_id,
                 resource=excluded.resource,
                 payload_json=COALESCE(excluded.payload_json,
                                       sofascore_http_cache.payload_json),
                 http_status=excluded.http_status,
                 fetched_at=excluded.fetched_at,
                 expires_at=excluded.expires_at,
                 last_error=excluded.last_error""",
            (
                key, match_id, resource,
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                int(status), now, expires, str(error or "")[:300],
            ),
        )
        conn.commit()
    return payload, False, status


def _fetch_team_recent_sofascore(
    db_path: str, team_id: Any, ttl_seconds: int = LIVE_RECENT_TTL_SECONDS,
) -> tuple[dict[str, Any] | None, bool, int]:
    """Busca a página recente do time com cache curto e isolado por provedor."""
    init_sofascore_db(db_path)
    team_id = str(team_id or "").strip()
    if not team_id:
        return None, False, 0
    now = int(time.time())
    key = f"team:{team_id}:events:last:0"
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            """SELECT payload_json,http_status,expires_at
               FROM sofascore_http_cache WHERE cache_key=?""",
            (key,),
        ).fetchone()
    if row and int(row[2] or 0) > now:
        try:
            payload = json.loads(row[0]) if row[0] else None
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        status = int(row[1] or 0)
        return (payload if status == 200 and isinstance(payload, dict) else None,
                True, status)

    status, payload, error = _request_json(
        f"{SOFASCORE_BASE_URL}/team/{team_id}/events/last/0"
    )
    expires = now + (
        int(ttl_seconds) if status == 200 else (6 * 3600 if status == 404 else 15 * 60)
    )
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """INSERT INTO sofascore_http_cache
               (cache_key,match_id,resource,payload_json,http_status,
                fetched_at,expires_at,last_error) VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 payload_json=COALESCE(excluded.payload_json,
                                       sofascore_http_cache.payload_json),
                 http_status=excluded.http_status,
                 fetched_at=excluded.fetched_at,
                 expires_at=excluded.expires_at,
                 last_error=excluded.last_error""",
            (
                key, f"team:{team_id}", "team-last",
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                int(status), now, expires, str(error or "")[:300],
            ),
        )
        conn.commit()
    return payload if isinstance(payload, dict) else None, False, int(status)


def _payload_events(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("events", "matches", "data", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for nested in ("events", "matches", "items", "data"):
                items = value.get(nested)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
    return []


def _event_score(event: dict[str, Any], side: str) -> float | None:
    from football_results import regulation_score
    score = regulation_score(event, require_finished=False)
    return float(score[0 if side == "home" else 1]) if score is not None else None


def _event_is_finished(event: dict[str, Any]) -> bool:
    from football_results import event_finished
    return event_finished(event)


def _recent_event_rows(
    payload: dict[str, Any] | None,
    team_id: Any,
    team_name: str,
    cutoff_timestamp: int,
    limit: int = LIVE_RECENT_MATCHES,
) -> list[dict[str, Any]]:
    """Normaliza jogos concluídos estritamente anteriores ao confronto-alvo."""
    team_id = str(team_id or "").strip()
    rows: list[dict[str, Any]] = []
    for event in _payload_events(payload):
        from pregame_form_quality import number
        raw_ts = number(event.get("startTimestamp"))
        if raw_ts is None or not raw_ts.is_integer():
            continue
        event_ts = int(raw_ts)
        if event_ts <= 0 or event_ts >= int(cutoff_timestamp or time.time()):
            continue
        if not _event_is_finished(event):
            continue
        home = event.get("homeTeam") or event.get("home_team") or {}
        away = event.get("awayTeam") or event.get("away_team") or {}
        if not isinstance(home, dict) or not isinstance(away, dict):
            continue
        home_id, away_id = str(home.get("id") or ""), str(away.get("id") or "")
        home_name, away_name = str(home.get("name") or ""), str(away.get("name") or "")
        if team_id:
            # An explicit provider ID cannot be overridden by a similar name
            # (e.g. reserve/women's teams or unrelated namesakes).
            if team_id not in {home_id, away_id}:
                continue
            is_home = team_id == home_id
        else:
            home_similarity = _similarity(team_name, home_name)
            away_similarity = _similarity(team_name, away_name)
            if max(home_similarity, away_similarity) < 0.72:
                continue
            is_home = home_similarity >= away_similarity
        home_score, away_score = _event_score(event, "home"), _event_score(event, "away")
        if home_score is None or away_score is None:
            continue
        gf, ga = (home_score, away_score) if is_home else (away_score, home_score)
        opponent = away if is_home else home
        tournament = event.get("tournament") or event.get("competition") or {}
        rows.append({
            "match_id": str(event.get("id") or ""),
            "start_timestamp": event_ts,
            "is_home": float(is_home),
            "team_id": home_id if is_home else away_id,
            "team_name": home_name if is_home else away_name,
            "opponent_id": str(opponent.get("id") or ""),
            "opponent_name": str(opponent.get("name") or ""),
            "goals_for": float(gf), "goals_against": float(ga),
            "result_points": 3.0 if gf > ga else (1.0 if gf == ga else 0.0),
            "is_draw": float(gf == ga),
            "tournament_id": str(tournament.get("id") or "") if isinstance(tournament, dict) else "",
            "tournament_name": str(tournament.get("name") or "") if isinstance(tournament, dict) else "",
        })
    from recent_history_integrity import clean_recent_history
    # Raw rows above already require a final regulation score. Do not count
    # provider aliases as two games or choose one of contradictory score rows.
    clean, _ = clean_recent_history(
        rows, min(int(time.time()), int(cutoff_timestamp or time.time())),
        team_id=team_id, limit=max(1, int(limit)), end_buffer=0,
    )
    return clean


def _seed_recent_score_profiles(db_path: str, events: list[dict[str, Any]]) -> None:
    """Atualiza resultados recentes sem apagar xG/estatísticas já auditadas."""
    now = int(time.time())
    rows = []
    for event in events:
        team_name, opponent_name = event.get("team_name", ""), event.get("opponent_name", "")
        team_key = normalize_team_name(team_name)
        if not event.get("match_id") or not team_key:
            continue
        rows.append((
            str(event["match_id"]), team_key, str(event.get("team_id") or ""),
            str(team_name), normalize_team_name(opponent_name),
            int(event["start_timestamp"]), int(event.get("is_home") or 0),
            _safe_float(event.get("goals_for")), _safe_float(event.get("goals_against")),
            _safe_float(event.get("result_points")), int(event.get("is_draw") or 0), now,
        ))
    if not rows:
        return
    with closing(_connect(db_path)) as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO sofascore_team_match_profiles
               (match_id,team_key,team_id,team_name,opponent_key,start_timestamp,
                is_home,goals_for,goals_against,result_points,is_draw,captured_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()


def _opponent_recent_ppg(
    db_path: str, opponent_name: str, cutoff_timestamp: int, limit: int = 5,
) -> tuple[float, bool]:
    key = normalize_team_name(opponent_name)
    if not key:
        return 1.35, False
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            """SELECT result_points,team_name FROM sofascore_team_match_profiles
               WHERE team_key=? AND start_timestamp<?
               ORDER BY start_timestamp DESC LIMIT ?""",
            (key, int(cutoff_timestamp), max(20, int(limit) * 4)),
        ).fetchall()
    strict = _team_identity_name(opponent_name)
    exact = [row for row in rows if _team_identity_name(row[1]) == strict]
    rows = (exact or rows)[:int(limit)]
    if not rows:
        return 1.35, False
    return sum(_safe_float(row[0]) for row in rows) / len(rows), True


def _live_recent_features(
    db_path: str, prefix: str, events: list[dict[str, Any]], cutoff_timestamp: int,
) -> dict[str, float]:
    games = len(events)
    result = {f"live_recent_{prefix}_games": float(games)}
    # Capture richer sequence statistics for gated challengers. The incumbent
    # feature order does not automatically consume these newly named fields.
    from pregame_form_quality import sequence_features
    result.update(sequence_features(events, prefix, min(int(time.time()), cutoff_timestamp)))
    from recent_history_integrity import fresh_sequence_features
    fresh, _ = fresh_sequence_features(
        events, prefix, min(int(time.time()), cutoff_timestamp)
    )
    result.update(fresh)
    if not games:
        return result
    weights = [1.0, 0.90, 0.82, 0.74, 0.68, 0.55, 0.48, 0.42, 0.37, 0.33][:games]
    weight_sum = sum(weights)
    points = [_safe_float(item.get("result_points")) for item in events]
    gf = [_safe_float(item.get("goals_for")) for item in events]
    ga = [_safe_float(item.get("goals_against")) for item in events]
    opponent_ppg = []
    opponent_known = 0
    for item in events:
        value, known = _opponent_recent_ppg(
            db_path, str(item.get("opponent_name") or ""), cutoff_timestamp
        )
        opponent_ppg.append(value)
        opponent_known += int(known)
    raw_ppg = sum(points) / games
    weighted_ppg = sum(value * weight for value, weight in zip(points, weights)) / weight_sum
    schedule_ppg = sum(opponent_ppg) / games
    # Resultado contra adversário acima/abaixo da média recebe ajuste pequeno;
    # nunca permitimos que a estimativa de força suplante o placar real.
    adjusted_ppg = _clamp(
        weighted_ppg + 0.30 * (schedule_ppg - 1.35), 0.0, 3.0
    )
    latest_three = points[:3]
    older = points[3:]
    trend = ((sum(latest_three) / len(latest_three))
             - (sum(older) / len(older) if older else raw_ppg)) / 3.0
    same_venue = [item for item in events if bool(item.get("is_home")) == (prefix == "home")]
    venue_weights = weights[:len(same_venue)]
    venue_weight_sum = sum(venue_weights) or 1.0
    venue_weighted_ppg = (
        sum(_safe_float(item.get("result_points")) * weight
            for item, weight in zip(same_venue, venue_weights)) / venue_weight_sum
        if same_venue else raw_ppg
    )
    venue_reliability = _clamp(len(same_venue) / 5.0)
    venue_shrunk_ppg = (
        venue_reliability * venue_weighted_ppg
        + (1.0 - venue_reliability) * raw_ppg
    )
    result.update({
        f"live_recent_{prefix}_ppg": raw_ppg,
        f"live_recent_{prefix}_weighted_ppg": weighted_ppg,
        f"live_recent_{prefix}_strength_adjusted_ppg": adjusted_ppg,
        f"live_recent_{prefix}_opponent_ppg": schedule_ppg,
        f"live_recent_{prefix}_opponent_coverage": opponent_known / games,
        f"live_recent_{prefix}_goals_for_avg": sum(gf) / games,
        f"live_recent_{prefix}_goals_against_avg": sum(ga) / games,
        f"live_recent_{prefix}_goal_difference_avg": (sum(gf) - sum(ga)) / games,
        f"live_recent_{prefix}_win_rate": sum(value == 3 for value in points) / games,
        f"live_recent_{prefix}_draw_rate": sum(value == 1 for value in points) / games,
        f"live_recent_{prefix}_loss_rate": sum(value == 0 for value in points) / games,
        f"live_recent_{prefix}_clean_sheet_rate": sum(value == 0 for value in ga) / games,
        f"live_recent_{prefix}_failed_to_score_rate": sum(value == 0 for value in gf) / games,
        f"live_recent_{prefix}_trend": _clamp(trend, -1.0, 1.0),
        f"live_recent_{prefix}_same_venue_games": float(len(same_venue)),
        f"live_recent_{prefix}_same_venue_ppg": venue_shrunk_ppg,
        f"live_recent_{prefix}_same_venue_weighted_ppg": venue_weighted_ppg,
        f"live_recent_{prefix}_same_venue_reliability": venue_reliability,
        f"live_recent_{prefix}_games_14d": float(sum(
            int(cutoff_timestamp) - int(item["start_timestamp"]) <= 14 * 86400
            for item in events
        )),
    })
    return result


def _capture_live_recent_side(
    db_path: str,
    match_id: str,
    game: dict[str, Any],
    side: str,
    team_id: str,
    team_name: str,
    cutoff_timestamp: int,
    fallback_loader: Callable[[dict[str, Any], str, str], dict[str, Any] | None] | None,
) -> tuple[dict[str, float], str, bool, int]:
    payload, cache_hit, status = _fetch_team_recent_sofascore(db_path, team_id)
    provider = "sofascore"
    provider_team_id = team_id
    events = _recent_event_rows(payload, team_id, team_name, cutoff_timestamp)
    fallback_requests = 0
    if len(events) < 3 and fallback_loader is not None:
        try:
            fallback = fallback_loader(game, side, team_id) or {}
        except Exception:
            fallback = {}
        fallback_payload = fallback.get("payload") if isinstance(fallback, dict) else None
        fallback_team_id = str(fallback.get("team_id") or team_id) if isinstance(fallback, dict) else team_id
        fallback_events = _recent_event_rows(
            fallback_payload, fallback_team_id, team_name, cutoff_timestamp
        )
        fallback_requests = 1
        if len(fallback_events) > len(events):
            payload, events = fallback_payload, fallback_events
            provider = str(fallback.get("provider") or "allsports")
            provider_team_id = fallback_team_id
    _seed_recent_score_profiles(db_path, events)
    features = _live_recent_features(db_path, side, events, cutoff_timestamp)
    coverage = _clamp(len(events) / float(LIVE_RECENT_MATCHES))
    now = int(time.time())
    with closing(_connect(db_path)) as conn:
        from pregame_snapshot_store import save_recent_snapshot
        save_recent_snapshot(conn,
            (
                match_id, side, provider, provider_team_id, now,
                int(cutoff_timestamp), int(events[0]["start_timestamp"]) if events else 0,
                json.dumps(events, ensure_ascii=False),
                json.dumps(features, ensure_ascii=False), coverage,
            ),
        )
        conn.commit()
    return features, provider, cache_hit, fallback_requests


def get_event_for_audit(
    db_path: str, match_id: Any, force_refresh: bool = False
) -> dict[str, Any] | None:
    """Obtém evento/placar no SofaScore, sem gastar RapidAPI."""
    if not SOFASCORE_AUDIT_ENABLED:
        return None
    payload, _, _ = _fetch_resource(
        db_path, match_id, "event", ttl_seconds=30 * 60,
        force_refresh=force_refresh,
    )
    event = payload.get("event") if isinstance(payload, dict) else None
    return event if isinstance(event, dict) else None


def _form_features(prefix: str, team: dict[str, Any]) -> dict[str, float]:
    form = [str(item).upper() for item in (team.get("form") or []) if str(item).upper() in {"W", "D", "L"}][-5:]
    games = len(form)
    wins, draws, losses = form.count("W"), form.count("D"), form.count("L")
    recent = form[-3:]
    recent_points = sum(3 if item == "W" else 1 if item == "D" else 0 for item in recent)
    return {
        f"sofa_pre_{prefix}_games": float(games),
        f"sofa_pre_{prefix}_ppg": (3 * wins + draws) / games if games else 0.0,
        f"sofa_pre_{prefix}_win_rate": wins / games if games else 0.0,
        f"sofa_pre_{prefix}_draw_rate": draws / games if games else 0.0,
        f"sofa_pre_{prefix}_loss_rate": losses / games if games else 0.0,
        f"sofa_pre_{prefix}_recent_points_3": recent_points / max(1, 3 * len(recent)),
        f"sofa_pre_{prefix}_position": _safe_float(team.get("position")),
        f"sofa_pre_{prefix}_points": _safe_float(team.get("value")),
        f"sofa_pre_{prefix}_avg_rating": _safe_float(team.get("avgRating")),
    }


def _parse_pregame_payloads(
    form_payload: dict[str, Any] | None,
    streak_payload: dict[str, Any] | None,
) -> dict[str, float]:
    home = (form_payload or {}).get("homeTeam") or {}
    away = (form_payload or {}).get("awayTeam") or {}
    features: dict[str, float] = {
        "sofa_pre_available": float(bool(home and away)),
        "sofa_pre_streaks_available": float(bool(streak_payload)),
    }
    features.update(_form_features("home", home))
    features.update(_form_features("away", away))
    for name in ("ppg", "win_rate", "draw_rate", "loss_rate", "recent_points_3", "avg_rating"):
        features[f"sofa_pre_{name}_diff"] = (
            features[f"sofa_pre_home_{name}"] - features[f"sofa_pre_away_{name}"]
        )
    home_pos, away_pos = features["sofa_pre_home_position"], features["sofa_pre_away_position"]
    features["sofa_pre_position_adv_home"] = (
        away_pos - home_pos if home_pos > 0 and away_pos > 0 else 0.0
    )
    home_pts, away_pts = features["sofa_pre_home_points"], features["sofa_pre_away_points"]
    features["sofa_pre_points_diff"] = home_pts - away_pts

    trends = Counter()
    for item in (streak_payload or {}).get("general", []):
        name = str(item.get("name") or "").lower()
        team = str(item.get("team") or "both").lower()
        value = _ratio(item.get("value"))
        if value <= 0:
            continue
        targets = [team] if team in {"home", "away"} else ["home", "away"]
        for target in targets:
            if "more than 2.5" in name:
                trends[f"{target}_high_scoring"] = max(trends[f"{target}_high_scoring"], value)
            elif "less than 2.5" in name:
                trends[f"{target}_low_scoring"] = max(trends[f"{target}_low_scoring"], value)
            elif "first to score" in name:
                trends[f"{target}_first_score"] = max(trends[f"{target}_first_score"], value)
            elif "without clean sheet" in name or "conceded" in name:
                trends[f"{target}_concede"] = max(trends[f"{target}_concede"], value)
    for side in ("home", "away"):
        for trend in ("high_scoring", "low_scoring", "first_score", "concede"):
            features[f"sofa_pre_{side}_{trend}_trend"] = float(trends[f"{side}_{trend}"])
    return features


def _parse_h2h_payload(
    payload: dict[str, Any] | None,
    current_home: str,
    current_away: str,
    start_timestamp: int,
) -> dict[str, float]:
    """Resume confrontos anteriores sempre da perspectiva do jogo atual."""
    candidates = []
    for event in (payload or {}).get("events") or []:
        event_ts = _safe_timestamp(event.get("startTimestamp"))
        if event_ts <= 0 or event_ts >= _safe_timestamp(start_timestamp, int(time.time())):
            continue
        home_name = str((event.get("homeTeam") or {}).get("name") or "")
        away_name = str((event.get("awayTeam") or {}).get("name") or "")
        direct = (_similarity(home_name, current_home) >= 0.72
                  and _similarity(away_name, current_away) >= 0.72)
        reverse = (_similarity(home_name, current_away) >= 0.72
                   and _similarity(away_name, current_home) >= 0.72)
        if not (direct or reverse):
            continue
        if not _event_is_finished(event):
            continue
        hs, aws = _event_score(event, "home"), _event_score(event, "away")
        try:
            hs, aws = float(hs), float(aws)
        except (TypeError, ValueError):
            continue
        current_hg, current_ag = (hs, aws) if direct else (aws, hs)
        candidates.append((event_ts, current_hg, current_ag, float(reverse)))
    candidates.sort(reverse=True)
    candidates = candidates[:5]
    if not candidates:
        return {"sofa_h2h_available": 0.0, "sofa_h2h_games": 0.0}

    home_points = sum(3.0 if hg > ag else (1.0 if hg == ag else 0.0)
                      for _, hg, ag, _ in candidates)
    games = float(len(candidates))
    latest_ts, latest_hg, latest_ag, latest_reverse = candidates[0]
    days = max(0.0, (int(start_timestamp) - latest_ts) / 86400.0)
    recent_reverse = float(bool(latest_reverse and days <= 45.0))
    return {
        "sofa_h2h_available": 1.0,
        "sofa_h2h_games": games,
        "sofa_h2h_home_ppg": home_points / games,
        "sofa_h2h_away_ppg": (3.0 * games - home_points
                               - sum(1.0 for _, hg, ag, _ in candidates if hg == ag)) / games,
        "sofa_h2h_draw_rate": sum(1.0 for _, hg, ag, _ in candidates if hg == ag) / games,
        "sofa_h2h_recent_reverse": recent_reverse,
        "sofa_h2h_latest_days": days,
        "sofa_h2h_latest_home_goals": latest_hg,
        "sofa_h2h_latest_away_goals": latest_ag,
        "sofa_h2h_home_aggregate_deficit": (
            max(0.0, latest_ag - latest_hg) * recent_reverse
        ),
        "sofa_h2h_away_aggregate_deficit": (
            max(0.0, latest_hg - latest_ag) * recent_reverse
        ),
    }


def _save_match_link(
    db_path: str,
    match_id: str,
    event: dict[str, Any] | None,
    expected_home: str,
    expected_away: str,
    start_timestamp: int,
) -> tuple[bool, float]:
    event = event or {}
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    home_name = str(home.get("name") or expected_home or "")
    away_name = str(away.get("name") or expected_away or "")
    event_start = _safe_timestamp(event.get("startTimestamp"), start_timestamp)
    identity_ok, similarity = _fixture_identity_matches(
        expected_home, expected_away, start_timestamp,
        home_name, away_name, event_start,
    )
    validated = bool(event and identity_ok)
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sofascore_match_links
               (match_id, provider_event_id, home_team_id, away_team_id,
                home_name, away_name, start_timestamp, validated, similarity, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                match_id, str(event.get("id") or match_id),
                str(home.get("id") or ""), str(away.get("id") or ""),
                home_name, away_name,
                event_start,
                int(validated), float(similarity), int(time.time()),
            ),
        )
        conn.commit()
    return validated, similarity


def _profile_observed_metrics(values, raw_presence):
    """Decode measured columns; legacy all-zero seeds remain unknown."""
    try:
        recorded = json.loads(raw_presence) if raw_presence is not None else None
    except (TypeError, ValueError):
        recorded = None
    if isinstance(recorded, list):
        return set(recorded)
    observed = set()
    for metric in ("xg", "shots", "shots_on_target", "big_chances", "box_shots"):
        pair = (f"{metric}_for", f"{metric}_against")
        if any(_safe_float(values.get(key)) > 0 for key in pair):
            observed.update(key for key in pair if values.get(key) is not None)
    for metric in ("possession", "errors_to_goal"):
        if _safe_float(values.get(metric)) > 0:
            observed.add(metric)
    if observed and values.get("dominance") is not None:
        observed.add("dominance")
    return observed


def _rolling_team_features(
    db_path: str, prefix: str, team_name: str, cutoff_timestamp: int, limit: int = 8
) -> dict[str, float]:
    key = normalize_team_name(team_name)
    rows = []
    if key:
        with closing(_connect(db_path)) as conn:
            rows = conn.execute(
                """SELECT goals_for, goals_against, xg_for, xg_against,
                          shots_for, shots_against,
                          shots_on_target_for, shots_on_target_against,
                          big_chances_for, big_chances_against, dominance,
                          box_shots_for, box_shots_against, possession,
                          errors_to_goal, result_points, is_draw, is_home, team_name,
                          opponent_key, start_timestamp, metric_presence_json
                   FROM sofascore_team_match_profiles
                   WHERE team_key=? AND start_timestamp<?
                   ORDER BY start_timestamp DESC LIMIT ?""",
                (key, int(cutoff_timestamp or time.time()), max(10 * 8, 40)),
            ).fetchall()
    names = (
        "goals_for", "goals_against", "xg_for", "xg_against",
        "shots_for", "shots_against",
        "shots_on_target_for", "shots_on_target_against",
        "big_chances_for", "big_chances_against", "dominance",
        "box_shots_for", "box_shots_against", "possession",
        "errors_to_goal", "result_points", "is_draw", "is_home",
    )
    # O normalizador flexível remove FC/SC para vincular grafias. Para o
    # histórico móvel, porém, isso misturava o FC Lugano suíço com o Lugano
    # argentino. Se há a grafia canônica do link SofaScore, ela prevalece.
    strict = _team_identity_name(team_name)
    exact_rows = [row for row in rows if _team_identity_name(row[-4]) == strict]
    if exact_rows:
        rows = exact_rows
    else:
        rows = [row for row in rows if _similarity(row[-4], team_name) >= 0.78]
    rows = rows[:10]
    data = [dict(zip(names, row[:len(names)])) for row in rows]
    # Strength of each opponent is calculated strictly before the historical
    # match being summarized. This prevents a later season/table state from
    # leaking into an earlier performance adjustment.
    with closing(_connect(db_path)) as conn:
        for raw, values in zip(rows, data):
            opponent_key, played_at = str(raw[-3] or ""), int(raw[-2] or 0)
            opponent_rows = conn.execute(
                """SELECT result_points,goals_for,goals_against
                   FROM sofascore_team_match_profiles
                   WHERE team_key=? AND start_timestamp<?
                   ORDER BY start_timestamp DESC LIMIT 10""",
                (opponent_key, played_at),
            ).fetchall() if opponent_key and played_at else []
            values["opponent_games"] = float(len(opponent_rows))
            values["opponent_ppg"] = (
                sum(_safe_float(item[0]) for item in opponent_rows) / len(opponent_rows)
                if opponent_rows else 1.35
            )
            values["opponent_gf"] = (
                sum(_safe_float(item[1]) for item in opponent_rows) / len(opponent_rows)
                if opponent_rows else 1.30
            )
            values["opponent_ga"] = (
                sum(_safe_float(item[2]) for item in opponent_rows) / len(opponent_rows)
                if opponent_rows else 1.30
            )
    presence = []
    for row, values in zip(rows, data):
        observed = _profile_observed_metrics(values, row[-1])
        observed.update(key for key in ("goals_for", "goals_against", "result_points", "is_draw", "is_home")
                        if values.get(key) is not None)
        presence.append(observed)
    # Existing v5 fields preserve their former eight-game representation.
    legacy_data, legacy_presence = data[:int(limit)], presence[:int(limit)]
    games = len(legacy_data)
    result = {f"sofa_roll_{prefix}_games": float(games)}
    result["sofa_metric_coverage_version"] = 1.0
    for name in names[:-3]:
        values = [_safe_float(row.get(name)) for row, observed in zip(legacy_data, legacy_presence)
                  if name in observed and row.get(name) is not None]
        result[f"sofa_roll_{prefix}_{name}_avg"] = sum(values) / len(values) if values else 0.0
        result[f"sofa_roll_{prefix}_{name}_games"] = float(len(values))
        # The current v5 overlay was tuned on diluted averages. Keep its
        # historical representation explicit until a coverage-aware challenger
        # wins future validation; corrected means remain available to learners.
        legacy_values = [_safe_float(row.get(name), .5 if name == "dominance" else 0)
                         for row in legacy_data]
        result[f"sofa_roll_{prefix}_legacy_{name}_avg"] = (
            sum(legacy_values) / games if games else 0.0
        )
    result[f"sofa_roll_{prefix}_xg_games"] = float(sum(
        {"xg_for", "xg_against"}.issubset(observed)
        for observed in legacy_presence
    ))
    result[f"sofa_roll_{prefix}_ppg"] = (
        sum(_safe_float(row.get("result_points")) for row in legacy_data) / games if games else 0.0
    )
    result[f"sofa_roll_{prefix}_draw_rate"] = (
        sum(int(row.get("is_draw") or 0) for row in legacy_data) / games if games else 0.0
    )
    weights = (1.0, .90, .82, .74, .68, .55, .48, .42, .37, .33)
    metric_names = names[:-3]

    def window_features(window: int) -> None:
        subset, observed_subset = data[:window], presence[:window]
        suffix = str(window)
        result[f"sofa_roll_{prefix}_window_{suffix}_games"] = float(len(subset))
        for name in metric_names:
            measured = [(_safe_float(row.get(name)), weights[index])
                        for index, (row, observed) in enumerate(zip(subset, observed_subset))
                        if name in observed and row.get(name) is not None]
            count = len(measured)
            result[f"sofa_roll_{prefix}_{name}_games_{suffix}"] = float(count)
            result[f"sofa_roll_{prefix}_{name}_avg_{suffix}"] = (
                sum(value for value, _ in measured) / count if count else 0.0
            )
            result[f"sofa_roll_{prefix}_{name}_weighted_avg_{suffix}"] = (
                sum(value * weight for value, weight in measured)
                / sum(weight for _, weight in measured) if measured else 0.0
            )
        opponent_weights = weights[:len(subset)]
        weight_sum = sum(opponent_weights) or 1.0
        for name, prior in (("ppg", 1.35), ("gf", 1.30), ("ga", 1.30)):
            result[f"sofa_roll_{prefix}_opponent_{name}_avg_{suffix}"] = (
                sum(_safe_float(row.get("opponent_" + name), prior) * weight
                    for row, weight in zip(subset, opponent_weights)) / weight_sum
                if subset else prior
            )
        result[f"sofa_roll_{prefix}_opponent_games_min_{suffix}"] = float(
            min((_safe_float(row.get("opponent_games")) for row in subset), default=0.0)
        )
        outcome_pairs = [
            (row, weights[index]) for index, row in enumerate(subset)
            if row.get("goals_for") is not None and row.get("goals_against") is not None
        ]
        if outcome_pairs:
            total_weight = sum(weight for _, weight in outcome_pairs)
            goal_diffs = [(_safe_float(row["goals_for"])-_safe_float(row["goals_against"]), weight)
                          for row, weight in outcome_pairs]
            totals = [(_safe_float(row["goals_for"])+_safe_float(row["goals_against"]), weight)
                      for row, weight in outcome_pairs]
            diff_mean = sum(value*weight for value, weight in goal_diffs) / total_weight
            total_mean = sum(value*weight for value, weight in totals) / total_weight
            result[f"sofa_roll_{prefix}_ppg_weighted_{suffix}"] = sum(
                _safe_float(row.get("result_points"))*weight for row, weight in outcome_pairs
            ) / total_weight
            result[f"sofa_roll_{prefix}_draw_rate_weighted_{suffix}"] = sum(
                float(bool(row.get("is_draw")))*weight for row, weight in outcome_pairs
            ) / total_weight
            result[f"sofa_roll_{prefix}_goal_diff_std_{suffix}"] = math.sqrt(sum(
                weight*(value-diff_mean)**2 for value, weight in goal_diffs
            ) / total_weight)
            result[f"sofa_roll_{prefix}_total_goals_std_{suffix}"] = math.sqrt(sum(
                weight*(value-total_mean)**2 for value, weight in totals
            ) / total_weight)
            result[f"sofa_roll_{prefix}_low_total_rate_{suffix}"] = sum(
                weight*float(value <= 2.0) for value, weight in totals
            ) / total_weight
            result[f"sofa_roll_{prefix}_close_game_rate_{suffix}"] = sum(
                weight*float(abs(value) <= 1.0) for value, weight in goal_diffs
            ) / total_weight
            result[f"sofa_roll_{prefix}_clean_sheet_rate_{suffix}"] = sum(
                weight*float(_safe_float(row["goals_against"]) == 0.0)
                for row, weight in outcome_pairs
            ) / total_weight
            result[f"sofa_roll_{prefix}_failed_to_score_rate_{suffix}"] = sum(
                weight*float(_safe_float(row["goals_for"]) == 0.0)
                for row, weight in outcome_pairs
            ) / total_weight
        else:
            for metric in ("ppg_weighted", "draw_rate_weighted", "goal_diff_std",
                           "total_goals_std", "low_total_rate", "close_game_rate",
                           "clean_sheet_rate", "failed_to_score_rate"):
                result[f"sofa_roll_{prefix}_{metric}_{suffix}"] = 0.0
        expected_venue = 1 if prefix == "home" else 0
        venue = [(row, weights[index]) for index, row in enumerate(subset)
                 if int(_safe_float(row.get("is_home"), -1)) == expected_venue]
        result[f"sofa_roll_{prefix}_venue_games_{suffix}"] = float(len(venue))
        if venue:
            venue_weight = sum(weight for _, weight in venue)
            for metric in ("goals_for", "goals_against", "result_points", "is_draw"):
                result[f"sofa_roll_{prefix}_venue_{metric}_{suffix}"] = sum(
                    _safe_float(row.get(metric))*weight for row, weight in venue
                ) / venue_weight
        else:
            for metric in ("goals_for", "goals_against", "result_points", "is_draw"):
                result[f"sofa_roll_{prefix}_venue_{metric}_{suffix}"] = 0.0
        # Normalize attacking output by the defensive strength faced and
        # defensive output by the attacking strength faced. Shrink short
        # opponent histories back to the neutral 1.30-goal baseline.
        adjusted_pairs = (
            ("goals_for", "opponent_ga"),
            ("xg_for", "opponent_ga"), ("shots_for", "opponent_ga"),
            ("shots_on_target_for", "opponent_ga"),
            ("big_chances_for", "opponent_ga"), ("box_shots_for", "opponent_ga"),
            ("goals_against", "opponent_gf"),
            ("xg_against", "opponent_gf"), ("shots_against", "opponent_gf"),
            ("shots_on_target_against", "opponent_gf"),
            ("big_chances_against", "opponent_gf"),
            ("box_shots_against", "opponent_gf"),
        )
        for metric, opponent_metric in adjusted_pairs:
            adjusted = []
            for index, (row, observed) in enumerate(zip(subset, observed_subset)):
                if metric not in observed or row.get(metric) is None:
                    continue
                reliability = min(1.0, _safe_float(row.get("opponent_games")) / 5.0)
                strength = reliability * _safe_float(row.get(opponent_metric), 1.30) + (1-reliability) * 1.30
                adjusted.append((_safe_float(row.get(metric)) * 1.30 / _clamp(strength, .55, 2.60),
                                 weights[index]))
            result[f"sofa_roll_{prefix}_{metric}_opponent_adjusted_{suffix}"] = (
                sum(value * weight for value, weight in adjusted)
                / sum(weight for _, weight in adjusted) if adjusted else 0.0
            )
            result[f"sofa_roll_{prefix}_{metric}_opponent_adjusted_games_{suffix}"] = float(len(adjusted))

        def wavg(metric, default=0.0):
            pairs = [(_safe_float(row.get(metric), default), weights[index])
                     for index, (row, observed) in enumerate(zip(subset, observed_subset))
                     if metric in observed]
            return (sum(v*w for v,w in pairs)/sum(w for _,w in pairs)) if pairs else None

        shots, target, box = wavg("shots_for"), wavg("shots_on_target_for"), wavg("box_shots_for")
        big, xg = wavg("big_chances_for"), wavg("xg_for")
        shots_against, target_against = wavg("shots_against"), wavg("shots_on_target_against")
        possession, dominance = wavg("possession"), wavg("dominance")
        style_games = sum(
            sum(metric in observed for metric in (
                "shots_for", "shots_on_target_for", "box_shots_for",
                "big_chances_for", "xg_for", "possession", "dominance",
            )) >= 4
            for observed in observed_subset
        )
        style_available = style_games >= 3
        result[f"sofa_roll_{prefix}_style_games_{suffix}"] = float(style_games)
        result[f"sofa_roll_{prefix}_style_reliability_{suffix}"] = min(1.0, style_games/5.0)
        result[f"sofa_roll_{prefix}_style_available_{suffix}"] = float(style_available)
        if style_available:
            s, st, bx, bc = shots or 0.0, target or 0.0, box or 0.0, big or 0.0
            poss = (possession or 50.0) / 100.0
            dom = dominance if dominance is not None else .5
            result[f"sofa_roll_{prefix}_attack_intensity_{suffix}"] = s + 1.5*st + 1.2*bx + 2.0*bc
            result[f"sofa_roll_{prefix}_chance_quality_{suffix}"] = (xg or 0.0) / max(1.0, s)
            result[f"sofa_roll_{prefix}_box_shot_share_{suffix}"] = bx / max(1.0, s)
            result[f"sofa_roll_{prefix}_territorial_control_{suffix}"] = .5*poss + .5*dom
            shot_gap = s - (shots_against or s)
            target_gap = st - (target_against or st)
            # This is explicitly a pressure proxy from territorial and shot
            # production; it is not labelled PPDA or true pressing intensity.
            result[f"sofa_roll_{prefix}_pressure_proxy_{suffix}"] = _clamp(
                .5 + .18*math.tanh(shot_gap/6) + .12*math.tanh(target_gap/3)
                + .12*(dom-.5) + .08*((possession or 50.0)-50)/20, 0.0, 1.0
            )
            result[f"sofa_roll_{prefix}_directness_{suffix}"] = _clamp(
                (bx/max(1.0,s)) * (1.25-.5*poss), 0.0, 1.0
            )
        else:
            for name in ("attack_intensity", "chance_quality", "box_shot_share",
                         "territorial_control", "pressure_proxy", "directness"):
                result[f"sofa_roll_{prefix}_{name}_{suffix}"] = 0.0

    window_features(5)
    window_features(10)
    for name in metric_names:
        if (result.get(f"sofa_roll_{prefix}_{name}_games_10", 0) >= 6
                and result.get(f"sofa_roll_{prefix}_{name}_games_5", 0) > 0):
            result[f"sofa_roll_{prefix}_{name}_trend_5_vs_10"] = (
                result[f"sofa_roll_{prefix}_{name}_weighted_avg_5"]
                - result[f"sofa_roll_{prefix}_{name}_avg_10"]
            )
        else:
            result[f"sofa_roll_{prefix}_{name}_trend_5_vs_10"] = 0.0
    return result


def capture_pregame_contexts(
    db_path: str,
    games: Iterable[dict[str, Any]],
    progress: Callable[[str], None] | None = None,
    recent_form_fallback: (
        Callable[[dict[str, Any], str, str], dict[str, Any] | None] | None
    ) = None,
    pregame_payload_fallback: (
        Callable[[dict[str, Any], bool, bool], dict[str, Any] | None] | None
    ) = None,
    season_context_fallback: (
        Callable[[dict[str, Any]], dict[str, Any] | None] | None
    ) = None,
) -> dict[str, Any]:
    """Coleta contexto e forma viva anterior ao jogo.

    Quando ``recent_form_fallback`` é informado, a ordem é SofaScore ->
    AllSports (callback) -> forma resumida Soccer já armazenada. A última opção
    é lida posteriormente por ``get_soccer_context_features`` e não exige HTTP.
    """
    init_sofascore_db(db_path)
    games = list(games)
    summary = Counter(total=len(games))
    if not SOFASCORE_ENABLED or not SOFASCORE_RADAR_ENABLED:
        summary["disabled"] = len(games)
        return dict(summary)
    now = int(time.time())
    for index, game in enumerate(games, 1):
        match_id = str(game.get("ID") or game.get("match_id") or "")
        start_ts = _safe_timestamp(
            game.get("Timestamp") or game.get("start_timestamp")
        )
        home_name = str(game.get("Time Casa") or game.get("home_team") or "")
        away_name = str(game.get("Time Fora") or game.get("away_team") or "")
        if not match_id:
            summary["invalid"] += 1
            continue
        with closing(_connect(db_path)) as conn:
            existing = conn.execute(
                """SELECT captured_at,features_json,start_timestamp,home_name,away_name
                   FROM sofascore_pregame_context WHERE match_id=?""",
                (match_id,),
            ).fetchone()
        existing_features: dict[str, Any] = {}
        if existing:
            try:
                parsed_existing = json.loads(existing[1] or "{}")
                if isinstance(parsed_existing, dict):
                    existing_features = parsed_existing
            except (TypeError, ValueError, json.JSONDecodeError):
                existing_features = {}
        needs_season_upgrade = bool(
            season_context_fallback is not None
            and (
                "allsports_goal_home_available" not in existing_features
                or (
                    not float(existing_features.get("allsports_goal_home_available", 0) or 0)
                    and not float(existing_features.get("allsports_goal_away_available", 0) or 0)
                    and "allsports_match_context_available" not in existing_features
                )
            )
        )
        existing_matches_game = bool(
            existing and _fixture_identity_matches(
                home_name, away_name, start_ts,
                str(existing[3] or ""), str(existing[4] or ""), existing[2],
            )[0]
        )
        if (existing and existing_matches_game
                and now - int(existing[0] or 0) < LIVE_RECENT_TTL_SECONDS
                and not needs_season_upgrade):
            summary["context_cache_hits"] += 1
            continue
        if existing and existing_matches_game:
            summary["context_refreshes"] += 1
            summary["season_context_upgrades"] += int(needs_season_upgrade)
        elif existing:
            # Mesmo ID ligado a outro confronto/provedor: jamais reutiliza o
            # snapshot antigo. A nova coleta o substituirá somente se validar.
            summary["context_identity_mismatch"] += 1
        # Nunca cria um suposto snapshot pré-jogo depois que a partida começou.
        if start_ts and start_ts <= now:
            summary["already_started"] += 1
            continue
        ttl = max(1800, (start_ts - now) if start_ts else 6 * 3600)
        event_payload, event_hit, _ = _fetch_resource(db_path, match_id, "event", ttl)
        form_payload, form_hit, _ = _fetch_resource(db_path, match_id, "pregame-form", ttl)
        streak_payload, streak_hit, _ = _fetch_resource(db_path, match_id, "team-streaks", ttl)
        h2h_payload, h2h_hit, _ = _fetch_resource(db_path, match_id, "h2h", ttl)
        summary["cache_hits"] += (
            int(event_hit) + int(form_hit) + int(streak_hit) + int(h2h_hit)
        )
        summary["http_requests"] += (
            int(not event_hit) + int(not form_hit) + int(not streak_hit) + int(not h2h_hit)
        )
        needs_form = not bool(
            (form_payload or {}).get("homeTeam")
            and (form_payload or {}).get("awayTeam")
        )
        needs_streaks = not bool(streak_payload)
        event = event_payload.get("event") if isinstance(event_payload, dict) else None
        validated, similarity = _save_match_link(
            db_path, match_id, event if isinstance(event, dict) else None,
            home_name, away_name, start_ts,
        )
        features = _parse_pregame_payloads(form_payload, streak_payload)
        if validated:
            from competition_context import event_competition_flags
            features.update(event_competition_flags(event, game.get("Liga", ""), now, start_ts))
        features.update(_parse_h2h_payload(
            h2h_payload, home_name, away_name, start_ts,
        ))
        features["sofa_pre_link_similarity"] = float(similarity)
        event_home = str(((event or {}).get("homeTeam") or {}).get("name") or home_name)
        event_away = str(((event or {}).get("awayTeam") or {}).get("name") or away_name)
        canonical_home = event_home if validated else home_name
        canonical_away = event_away if validated else away_name
        if recent_form_fallback is not None:
            event_home_id = str(((event or {}).get("homeTeam") or {}).get("id") or "")
            event_away_id = str(((event or {}).get("awayTeam") or {}).get("id") or "")
            for side, team_id, team_name in (
                ("home", event_home_id, canonical_home),
                ("away", event_away_id, canonical_away),
            ):
                live_features, provider, live_hit, fallback_calls = _capture_live_recent_side(
                    db_path, match_id, game, side, team_id, team_name, start_ts,
                    recent_form_fallback,
                )
                features.update(live_features)
                summary["live_cache_hits"] += int(live_hit)
                summary["live_sofascore_http"] += int(not live_hit and bool(team_id))
                summary["live_fallback_calls"] += int(fallback_calls)
                summary[f"live_provider_{provider}"] += int(
                    live_features.get(f"live_recent_{side}_games", 0) >= 3
                )
        # A cota paga prioriza os últimos jogos. Forma/streaks alternativos só
        # são buscados depois que o fallback de forma viva já teve sua chance.
        if pregame_payload_fallback is not None:
            fallback = pregame_payload_fallback(game, needs_form, needs_streaks) or {}
            if needs_form and isinstance(fallback.get("form"), dict):
                form_payload = fallback["form"]
                summary["allsports_form_fallback"] += 1
            if needs_streaks and isinstance(fallback.get("streaks"), dict):
                streak_payload = fallback["streaks"]
                summary["allsports_streaks_fallback"] += 1
            features.update(_parse_pregame_payloads(form_payload, streak_payload))
            if isinstance(fallback.get("features"), dict):
                features.update(fallback["features"])
            summary["allsports_pregame_http"] += int(fallback.get("http_requests", 0) or 0)
            summary["allsports_pregame_cache_hits"] += int(fallback.get("cache_hits", 0) or 0)
        live_home = features.get("live_recent_home_games", 0.0)
        live_away = features.get("live_recent_away_games", 0.0)
        features["live_recent_available"] = float(live_home >= 3 and live_away >= 3)
        features["live_recent_full_coverage"] = float(
            live_home >= LIVE_RECENT_MATCHES and live_away >= LIVE_RECENT_MATCHES
        )
        features.update(_rolling_team_features(
            db_path, "home", canonical_home, start_ts
        ))
        features.update(_rolling_team_features(
            db_path, "away", canonical_away, start_ts
        ))
        if season_context_fallback is not None:
            season_context = season_context_fallback(game) or {}
            if isinstance(season_context.get("features"), dict):
                features.update(season_context["features"])
            summary["allsports_season_http"] += int(
                season_context.get("http_requests", 0) or 0
            )
            summary["allsports_season_cache_hits"] += int(
                season_context.get("cache_hits", 0) or 0
            )
        if event is not None and not validated:
            # Um ID ligado ao confronto errado é pior que dado ausente.
            features = {key: 0.0 for key in features}
            features["sofa_pre_available"] = 0.0
            summary["link_mismatch"] += 1
        roll_home = features.get("sofa_roll_home_games", 0.0)
        roll_away = features.get("sofa_roll_away_games", 0.0)
        features["sofa_roll_available"] = float(roll_home > 0 and roll_away > 0)
        coverage = _clamp(
            0.35 * features.get("sofa_pre_available", 0.0)
            + 0.10 * features.get("sofa_pre_streaks_available", 0.0)
            + 0.25 * features.get("sofa_roll_available", 0.0)
            + 0.30 * features.get("live_recent_available", 0.0)
        )
        with closing(_connect(db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sofascore_pregame_context
                   (match_id, captured_at, start_timestamp, home_name, away_name,
                    features_json, coverage, source_version)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    match_id, int(time.time()), start_ts, home_name, away_name,
                    json.dumps(features, ensure_ascii=False), coverage, ANALYSIS_VERSION,
                ),
            )
            conn.commit()
        summary["captured"] += 1
        summary["available"] += int(features.get("sofa_pre_available", 0.0) > 0)
        summary["live_available"] += int(features.get("live_recent_available", 0.0) > 0)
        if progress and (index % 10 == 0 or index == len(games)):
            progress(f"SofaScore pré-jogo {index}/{len(games)}")
    return dict(summary)


def backfill_historical_pregame(
    db_path: str,
    limit: int = 0,
    progress: Callable[[dict[str, Any]], None] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Recupera a forma pré-jogo arquivada dos jogos de treinamento.

    A rota ``event/{id}/pregame-form`` é um snapshot associado ao evento e não
    uma consulta da forma atual do time. O processo é cronológico, retomável e
    não consulta placar/estatísticas para construir as features da própria
    partida. Respostas sem cobertura também são marcadas como concluídas.
    """
    init_sofascore_db(db_path)
    sql_limit = " LIMIT ?" if int(limit or 0) > 0 else ""
    params: tuple[Any, ...] = (int(limit),) if sql_limit else ()
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            """WITH chronological AS (
                   SELECT t.match_id, t.home_team, t.away_team, t.data_jogo,
                          NTILE(3) OVER (
                              ORDER BY t.data_jogo, t.match_id
                          ) AS temporal_period
                   FROM training_data t
               ), pending AS (
                   SELECT c.match_id, c.home_team, c.away_team, c.data_jogo,
                          b.status AS previous_status, c.temporal_period,
                          ROW_NUMBER() OVER (
                              PARTITION BY c.temporal_period
                              ORDER BY c.data_jogo, c.match_id
                          ) AS period_sequence
                   FROM chronological c
                   LEFT JOIN sofascore_historical_backfill b
                     ON b.match_id=c.match_id
                   WHERE b.match_id IS NULL OR b.status='ERROR'
               )
               SELECT match_id, home_team, away_team, data_jogo, previous_status
               FROM pending
               ORDER BY period_sequence, temporal_period""" + sql_limit,
            params,
        ).fetchall()
        existing = conn.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(has_pregame_form),0),
                      COALESCE(SUM(status='NO_COVERAGE'),0),
                      COALESCE(SUM(status='ERROR'),0)
               FROM sofascore_historical_backfill"""
        ).fetchone()
    summary = Counter(
        queued=len(rows), processed_before=int(existing[0] or 0),
        available_before=int(existing[1] or 0),
        no_coverage_before=int(existing[2] or 0),
        errors_before=int(existing[3] or 0),
    )
    started = time.monotonic()
    consecutive_blocked = 0
    for index, (match_id, home_name, away_name, data_jogo, previous_status) in enumerate(rows, 1):
        if stop_event is not None and stop_event.is_set():
            summary["stopped"] = 1
            break
        match_id = str(match_id)
        payload, cache_hit, http_status = _fetch_resource(
            db_path, match_id, "pregame-form", ttl_seconds=365 * 86400,
            force_refresh=(previous_status == "ERROR"),
        )
        summary["cache_hits"] += int(cache_hit)
        summary["http_requests"] += int(not cache_hit)
        if int(http_status or 0) in {403, 429}:
            consecutive_blocked += 1
        else:
            consecutive_blocked = 0
        home = (payload or {}).get("homeTeam") or {}
        away = (payload or {}).get("awayTeam") or {}
        has_form = bool(home and away)
        if has_form:
            features = _parse_pregame_payloads(payload, None)
            features["sofa_pre_link_similarity"] = 1.0
            # IDs AllSports/Sofa foram validados em amostra estratificada; o
            # vínculo completo foi 59/59 entre eventos encontrados.
            coverage = 0.70
            try:
                cutoff = int(datetime.fromisoformat(str(data_jogo)).timestamp())
            except (TypeError, ValueError, OverflowError):
                cutoff = 0
            with closing(_connect(db_path)) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sofascore_pregame_context
                       (match_id, captured_at, start_timestamp, home_name,
                        away_name, features_json, coverage, source_version)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        match_id, int(time.time()), cutoff, str(home_name or ""),
                        str(away_name or ""), json.dumps(features, ensure_ascii=False),
                        coverage, "historical-pregame-archive-v1",
                    ),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO sofascore_historical_backfill
                       (match_id, processed_at, status, http_status,
                        has_pregame_form, source_version, last_error)
                       VALUES (?,?,?,?,1,?,NULL)""",
                    (
                        match_id, int(time.time()), "AVAILABLE", int(http_status),
                        "historical-pregame-archive-v1",
                    ),
                )
                conn.commit()
            summary["available"] += 1
        else:
            status = "NO_COVERAGE" if int(http_status or 0) in {200, 404} else "ERROR"
            with closing(_connect(db_path)) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sofascore_historical_backfill
                       (match_id, processed_at, status, http_status,
                        has_pregame_form, source_version, last_error)
                       VALUES (?,?,?,?,0,?,?)""",
                    (
                        match_id, int(time.time()), status, int(http_status or 0),
                        "historical-pregame-archive-v1",
                        "sem forma arquivada" if status == "NO_COVERAGE" else "falha transitória",
                    ),
                )
                conn.commit()
            summary[status.lower()] += 1
        summary["processed"] += 1
        # Um 403/429 em sequência é bloqueio do provedor, não ausência de dados.
        # Interromper cedo preserva a fila para retomada e evita martelar o site.
        if consecutive_blocked >= 3:
            summary["circuit_open"] = 1
            summary["blocked_http_status"] = int(http_status or 0)
            summary["remaining_batch"] = max(0, len(rows) - index)
            if progress:
                elapsed = max(0.001, time.monotonic() - started)
                progress({
                    **dict(summary), "total_batch": len(rows),
                    "rate_per_second": summary["processed"] / elapsed,
                    "eta_seconds": None, "current_match_id": match_id,
                })
            break
        if progress and (index % 25 == 0 or index == len(rows)):
            elapsed = max(0.001, time.monotonic() - started)
            rate = summary["processed"] / elapsed
            remaining = max(0, len(rows) - summary["processed"])
            progress({
                **dict(summary),
                "total_batch": len(rows),
                "rate_per_second": rate,
                "eta_seconds": remaining / rate if rate else None,
                "current_match_id": match_id,
            })
    elapsed = max(0.001, time.monotonic() - started)
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["rate_per_second"] = summary["processed"] / elapsed
    return dict(summary)


def historical_backfill_status(db_path: str) -> dict[str, Any]:
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0])
        rows = conn.execute(
            """SELECT status, COUNT(*) FROM sofascore_historical_backfill
               GROUP BY status"""
        ).fetchall()
        cached = int(conn.execute(
            "SELECT COUNT(*) FROM sofascore_pregame_context"
        ).fetchone()[0])
        latest = conn.execute(
            "SELECT MAX(processed_at) FROM sofascore_historical_backfill"
        ).fetchone()[0]
    counts = {str(status): int(count) for status, count in rows}
    # ERROR continua pendente e será tentado novamente na próxima execução.
    processed = counts.get("AVAILABLE", 0) + counts.get("NO_COVERAGE", 0)
    return {
        "total": total, "processed": processed,
        "remaining": max(0, total - processed),
        "coverage_available": counts.get("AVAILABLE", 0),
        "no_coverage": counts.get("NO_COVERAGE", 0),
        "errors": counts.get("ERROR", 0),
        "pregame_context_rows": cached,
        "latest_processed_at": int(latest or 0),
        "status_counts": counts,
    }


def seed_historical_score_profiles(db_path: str) -> dict[str, int]:
    """Cria imediatamente a forma móvel por placar, sem nenhuma requisição HTTP.

    As linhas detalhadas já existentes não são sobrescritas. O backfill de xG e
    chances pode enriquecê-las depois, mas o radar deixa de operar sem histórico
    enquanto essa coleta demorada não termina.
    """
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        games = conn.execute(
            """SELECT match_id, home_team, away_team, data_jogo,
                      home_score, away_score
               FROM training_data
               WHERE home_score IS NOT NULL AND away_score IS NOT NULL
               ORDER BY data_jogo, match_id"""
        ).fetchall()
        before = int(conn.execute(
            "SELECT COUNT(*) FROM sofascore_team_match_profiles"
        ).fetchone()[0])
        now = int(time.time())
        rows = []
        for match_id, home_name, away_name, data_jogo, home_score, away_score in games:
            try:
                start_ts = int(datetime.fromisoformat(str(data_jogo)).timestamp())
            except (TypeError, ValueError, OverflowError):
                start_ts = 0
            home_goals = max(0.0, _safe_float(home_score))
            away_goals = max(0.0, _safe_float(away_score))
            for team_name, opponent_name, is_home, gf, ga in (
                (home_name, away_name, 1, home_goals, away_goals),
                (away_name, home_name, 0, away_goals, home_goals),
            ):
                points = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
                rows.append((
                    str(match_id), normalize_team_name(team_name), "",
                    str(team_name or ""), normalize_team_name(opponent_name),
                    start_ts, is_home, gf, ga,
                    None, None, None, None, None, None, None, None,
                    None, None, None, None, None, points, int(gf == ga), now,
                ))
        conn.executemany(
            """INSERT OR IGNORE INTO sofascore_team_match_profiles
               (match_id, team_key, team_id, team_name, opponent_key,
                start_timestamp, is_home, goals_for, goals_against, xg_for,
                xg_against, shots_for, shots_against, shots_on_target_for,
                shots_on_target_against, big_chances_for, big_chances_against,
                box_shots_for, box_shots_against, possession, dominance,
                errors_to_goal, result_points, is_draw, captured_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        after = int(conn.execute(
            "SELECT COUNT(*) FROM sofascore_team_match_profiles"
        ).fetchone()[0])
    return {
        "games": len(games), "profiles_before": before,
        "profiles_after": after, "profiles_added": max(0, after - before),
    }


def backfill_historical_details(
    db_path: str,
    limit: int = 250,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Enriquece perfis passados com estatísticas finais, uma chamada por jogo.

    O placar e a forma são semeados localmente primeiro. xG/chances do jogo N só
    são lidos por ``_rolling_team_features`` quando o corte é posterior a N.
    """
    init_sofascore_db(db_path)
    seed = seed_historical_score_profiles(db_path)
    batch_limit = max(1, int(limit or 250))
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            """WITH chronological AS (
                   SELECT t.match_id, t.home_team, t.away_team, t.data_jogo,
                          t.home_score, t.away_score,
                          NTILE(3) OVER (ORDER BY t.data_jogo,t.match_id) AS period
                   FROM training_data t
                   WHERE t.home_score IS NOT NULL AND t.away_score IS NOT NULL
               ), pending AS (
                   SELECT c.*, d.status AS previous_status,
                          ROW_NUMBER() OVER (
                              PARTITION BY c.period ORDER BY c.data_jogo,c.match_id
                          ) AS seq
                   FROM chronological c
                   LEFT JOIN sofascore_historical_detail_backfill d
                     ON d.match_id=c.match_id
                   WHERE d.match_id IS NULL OR d.status='ERROR'
               )
               SELECT match_id,home_team,away_team,data_jogo,home_score,away_score,
                      previous_status
               FROM pending ORDER BY seq,period LIMIT ?""",
            (batch_limit,),
        ).fetchall()
    summary = Counter(queued=len(rows))
    summary.update(seed)
    consecutive_blocked = 0
    for index, row in enumerate(rows, 1):
        match_id, home_name, away_name, data_jogo, home_score, away_score, previous = row
        payload, cache_hit, http_status = _fetch_resource(
            db_path, str(match_id), "statistics", ttl_seconds=365 * 86400,
            force_refresh=(previous == "ERROR"),
        )
        summary["cache_hits"] += int(cache_hit)
        summary["http_requests"] += int(not cache_hit)
        metrics, coverage = _statistics_metrics(payload)
        if int(http_status or 0) in {403, 429}:
            consecutive_blocked += 1
        else:
            consecutive_blocked = 0
        if coverage > 0:
            try:
                start_ts = int(datetime.fromisoformat(str(data_jogo)).timestamp())
            except (TypeError, ValueError, OverflowError):
                start_ts = 0
            event = {
                "startTimestamp": start_ts,
                "homeTeam": {"name": str(home_name or "")},
                "awayTeam": {"name": str(away_name or "")},
                "homeScore": {"normaltime": _safe_float(home_score)},
                "awayScore": {"normaltime": _safe_float(away_score)},
            }
            home_dom, _, _ = _dominance(metrics)
            _save_team_profiles(db_path, str(match_id), event, metrics, home_dom)
            status, error = "AVAILABLE", None
            summary["available"] += 1
        else:
            status = "NO_COVERAGE" if int(http_status or 0) in {200, 404} else "ERROR"
            error = "sem estatísticas" if status == "NO_COVERAGE" else "falha transitória"
            summary[status.lower()] += 1
        with closing(_connect(db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sofascore_historical_detail_backfill
                   (match_id,processed_at,status,http_status,statistics_coverage,
                    source_version,last_error) VALUES (?,?,?,?,?,?,?)""",
                (str(match_id), int(time.time()), status, int(http_status or 0),
                 float(coverage), "historical-details-v1", error),
            )
            conn.commit()
        summary["processed"] += 1
        if progress and (index % 25 == 0 or index == len(rows)):
            progress({**dict(summary), "total_batch": len(rows),
                      "current_match_id": str(match_id)})
        if consecutive_blocked >= 3:
            summary["circuit_open"] = 1
            summary["blocked_http_status"] = int(http_status or 0)
            break
    return dict(summary)


def historical_detail_status(db_path: str) -> dict[str, Any]:
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0])
        rows = conn.execute(
            """SELECT status,COUNT(*) FROM sofascore_historical_detail_backfill
               GROUP BY status"""
        ).fetchall()
        profiles = int(conn.execute(
            "SELECT COUNT(*) FROM sofascore_team_match_profiles"
        ).fetchone()[0])
    counts = {str(status): int(count) for status, count in rows}
    processed = counts.get("AVAILABLE", 0) + counts.get("NO_COVERAGE", 0)
    return {
        "total": total, "processed": processed,
        "remaining": max(0, total - processed), "profiles": profiles,
        "available": counts.get("AVAILABLE", 0),
        "no_coverage": counts.get("NO_COVERAGE", 0),
        "errors": counts.get("ERROR", 0), "status_counts": counts,
    }


def _local_h2h_features(
    db_path: str, home_name: str, away_name: str, cutoff_timestamp: int,
) -> dict[str, float]:
    """Retrospecto local anterior ao jogo, sem HTTP e sem usar o placar atual."""
    home_key, away_key = normalize_team_name(home_name), normalize_team_name(away_name)
    if not home_key or not away_key:
        return {}
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            """SELECT start_timestamp, is_home, goals_for, goals_against,
                      result_points, is_draw, team_name
               FROM sofascore_team_match_profiles
               WHERE team_key=? AND opponent_key=? AND start_timestamp<?
               ORDER BY start_timestamp DESC LIMIT 12""",
            (home_key, away_key, int(cutoff_timestamp or time.time())),
        ).fetchall()
    strict = _team_identity_name(home_name)
    exact = [row for row in rows if _team_identity_name(row[-1]) == strict]
    if exact:
        rows = exact
    rows = rows[:5]
    if not rows:
        return {}
    games = float(len(rows))
    latest_ts, latest_is_home, latest_gf, latest_ga, _, _, _ = rows[0]
    days = max(0.0, (int(cutoff_timestamp) - int(latest_ts)) / 86400.0)
    recent_reverse = float(not bool(latest_is_home) and days <= 45.0)
    home_points = sum(_safe_float(row[4]) for row in rows)
    draws = sum(int(row[5] or 0) for row in rows)
    return {
        "sofa_h2h_available": 1.0,
        "sofa_h2h_games": games,
        "sofa_h2h_home_ppg": home_points / games,
        "sofa_h2h_away_ppg": (3.0 * games - home_points - draws) / games,
        "sofa_h2h_draw_rate": draws / games,
        "sofa_h2h_recent_reverse": recent_reverse,
        "sofa_h2h_latest_days": days,
        "sofa_h2h_latest_home_goals": _safe_float(latest_gf),
        "sofa_h2h_latest_away_goals": _safe_float(latest_ga),
        "sofa_h2h_home_aggregate_deficit": (
            max(0.0, _safe_float(latest_ga) - _safe_float(latest_gf))
            * recent_reverse
        ),
        "sofa_h2h_away_aggregate_deficit": (
            max(0.0, _safe_float(latest_gf) - _safe_float(latest_ga))
            * recent_reverse
        ),
    }


def get_pregame_features(
    db_path: str,
    match_id: Any,
    home_name: str = "",
    away_name: str = "",
    cutoff_timestamp: int | None = None,
) -> dict[str, float]:
    """Lê snapshot e perfil anterior; não executa HTTP nem lê a autópsia atual."""
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            """SELECT features_json,start_timestamp,home_name,away_name
               FROM sofascore_pregame_context WHERE match_id=?""",
            (str(match_id or ""),),
        ).fetchone()
        link = conn.execute(
            """SELECT home_name,away_name,validated,start_timestamp
               FROM sofascore_match_links WHERE match_id=?""",
            (str(match_id or ""),),
        ).fetchone()
    if row and home_name and away_name:
        row_matches, _ = _fixture_identity_matches(
            home_name, away_name, cutoff_timestamp,
            str(row[2] or ""), str(row[3] or ""), row[1],
        )
        if not row_matches:
            row = None
    try:
        features = json.loads(row[0]) if row else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        features = {}
    if not isinstance(features, dict):
        features = {}
    cutoff = int(cutoff_timestamp or time.time())
    link_matches = bool(
        link and int(link[2] or 0)
        and _fixture_identity_matches(
            home_name, away_name, cutoff,
            str(link[0] or ""), str(link[1] or ""), link[3],
        )[0]
    )
    canonical_home = str(link[0]) if link_matches and link[0] else home_name
    canonical_away = str(link[1]) if link_matches and link[1] else away_name
    # Perfis podem ter crescido entre coleta e previsão. O WHERE temporal impede
    # uso do próprio jogo, e o snapshot final será gravado logo após a previsão.
    features.update(_rolling_team_features(db_path, "home", canonical_home, cutoff))
    features.update(_rolling_team_features(db_path, "away", canonical_away, cutoff))
    local_h2h = _local_h2h_features(
        db_path, canonical_home, canonical_away, cutoff,
    )
    for key, value in local_h2h.items():
        features.setdefault(key, value)
    features["sofa_roll_available"] = float(
        features.get("sofa_roll_home_games", 0) > 0
        and features.get("sofa_roll_away_games", 0) > 0
    )
    return {
        str(key): _safe_float(value)
        for key, value in features.items()
        if isinstance(value, (int, float))
    }


def analyze_pregame_context(
    base_probabilities: Iterable[float], features: dict[str, Any],
    analysis_profile: str = "current",
) -> dict[str, Any]:
    """Combina o ML com forma, ataque-defesa e risco estrutural de empate."""
    normalized_profile = str(analysis_profile).lower()
    legacy_v5 = normalized_profile in {
        "v5", "sofa-context-v5", "active-v5", "v5-quality", "active-v5-quality"
    }
    coverage_aware = normalized_profile in {"v5-quality", "active-v5-quality"}
    if legacy_v5 and not coverage_aware:
        # Do not silently change the champion's probability scale because a
        # storage bug was fixed. Its old input view is frozen; quality variants
        # and snapshot-only challengers consume the corrected metric means.
        features = dict(features)
        for side in ("home", "away"):
            for metric in ("xg_for", "xg_against", "dominance",
                           "shots_on_target_for", "shots_on_target_against",
                           "big_chances_for", "big_chances_against"):
                legacy_key = f"sofa_roll_{side}_legacy_{metric}_avg"
                if legacy_key in features:
                    features[f"sofa_roll_{side}_{metric}_avg"] = features[legacy_key]

    def metric_samples(side: str, metric: str) -> float:
        key = f"sofa_roll_{side}_{metric}_games"
        if key in features:
            return max(0.0, _safe_float(features[key]))
        # Historical snapshots had no count. Positive pair values prove at
        # least one observation, not coverage of all eight historical games.
        if metric == "xg":
            return float(max(_safe_float(features.get(f"sofa_roll_{side}_xg_for_avg")),
                             _safe_float(features.get(f"sofa_roll_{side}_xg_against_avg"))) > 0)
        return 0.0
    base = [_clamp(_safe_float(value)) for value in base_probabilities]
    total = sum(base) or 1.0
    base = [value / total for value in base]
    signals: list[tuple[float, float, str]] = []
    source_signals: dict[str, list[tuple[float, float]]] = {}
    draw_rates: list[float] = []
    reasons: list[str] = []
    expected_totals: list[float] = []
    parity_values: list[float] = []
    source_reliabilities: list[float] = []
    source_min_games: list[float] = []

    def source_reliability(home_key: str, away_key: str) -> float:
        # Chaves ausentes em snapshots v2 equivalem ao formato antigo (5 jogos).
        home_games = max(0.0, _safe_float(features.get(home_key), 5.0))
        away_games = max(0.0, _safe_float(features.get(away_key), 5.0))
        minimum = min(home_games, away_games)
        source_min_games.append(minimum)
        reliability = _clamp(minimum / 5.0)
        source_reliabilities.append(reliability)
        return reliability

    def add_signal(value: float, weight: float, label: str, source: str) -> None:
        if weight <= 0:
            return
        signals.append((value, weight, label))
        source_signals.setdefault(source, []).append((value, weight))

    def add_pair(home_key: str, away_key: str, scale: float, weight: float,
                 label: str, reliability: float = 1.0,
                 source: str = "other") -> None:
        home, away = _safe_float(features.get(home_key)), _safe_float(features.get(away_key))
        if home_key in features and away_key in features and (home != 0 or away != 0):
            add_signal(math.tanh((home - away) / max(0.01, scale)),
                       weight * reliability, label, source)

    if _safe_float(features.get("context_sfi_available")) > 0:
        sfi_rel = source_reliability("form_sfi_home_games", "form_sfi_away_games")
        add_pair("form_sfi_home_ppg", "form_sfi_away_ppg", 1.2, 1.0, "forma Soccer atual", sfi_rel, "sfi")
        add_pair("form_sfi_home_recent_points_3", "form_sfi_away_recent_points_3", 0.45, 0.55, "momento recente", sfi_rel, "sfi")
        home_gf = _safe_float(features.get("form_sfi_home_goals_for_avg"), 1.3)
        home_ga = _safe_float(features.get("form_sfi_home_goals_against_avg"), 1.3)
        away_gf = _safe_float(features.get("form_sfi_away_goals_for_avg"), 1.3)
        away_ga = _safe_float(features.get("form_sfi_away_goals_against_avg"), 1.3)
        expected_home = max(0.0, (home_gf + away_ga) / 2.0)
        expected_away = max(0.0, (away_gf + home_ga) / 2.0)
        # A fresh HTTP response is not proof that its aggregate is coherent.
        # Contradictory totals/minute counters must not create a false duel.
        # Preserve independent W/D/L; do not invent corrected goals or exclude
        # the fixture. Older snapshots lacking this diagnostic retain semantics.
        sfi_goals_inconsistent = any(_safe_float(features.get(
            f"form_sfi_{side}_goals_inconsistent")) > 0 for side in ("home", "away"))
        sfi_goals_missing = any(f"form_sfi_{side}_goals_available" in features
            and _safe_float(features[f"form_sfi_{side}_goals_available"]) <= 0
            for side in ("home", "away"))
        if not sfi_goals_inconsistent and not sfi_goals_missing:
            add_signal(
                math.tanh((expected_home - expected_away) / 1.15), 1.15 * sfi_rel,
                "ataque contra defesa da Soccer", "sfi",
            )
            expected_totals.append(expected_home + expected_away)
            parity_values.append(math.exp(-abs(expected_home - expected_away) / 0.85))
        else:
            reasons.append("Resumo de gols SFI ausente/contraditório; duelo dessa fonte ignorado")
        draw_rates.extend([
            _safe_float(features.get("form_sfi_home_draw_rate")),
            _safe_float(features.get("form_sfi_away_draw_rate")),
        ])
    if _safe_float(features.get("live_recent_available")) > 0:
        live_rel = source_reliability(
            "live_recent_home_games", "live_recent_away_games"
        )
        add_pair(
            "live_recent_home_strength_adjusted_ppg",
            "live_recent_away_strength_adjusted_ppg",
            1.15, 1.25, "últimos cinco ajustados pelos adversários",
            live_rel, "live",
        )
        add_pair(
            "live_recent_home_weighted_ppg",
            "live_recent_away_weighted_ppg",
            1.15, 0.80, "momento com peso de recência",
            live_rel, "live",
        )
        add_pair(
            "live_recent_home_same_venue_ppg",
            "live_recent_away_same_venue_ppg",
            1.20, 0.45, "forma recente no mando equivalente",
            live_rel, "live",
        )
        add_pair(
            "live_recent_home_trend", "live_recent_away_trend",
            0.65, 0.40, "aceleração ou queda nos últimos três",
            live_rel, "live",
        )
        home_gf = _safe_float(features.get("live_recent_home_goals_for_avg"))
        home_ga = _safe_float(features.get("live_recent_home_goals_against_avg"))
        away_gf = _safe_float(features.get("live_recent_away_goals_for_avg"))
        away_ga = _safe_float(features.get("live_recent_away_goals_against_avg"))
        if max(home_gf, home_ga, away_gf, away_ga) > 0:
            expected_home = (home_gf + away_ga) / 2.0
            expected_away = (away_gf + home_ga) / 2.0
            add_signal(
                math.tanh((expected_home - expected_away) / 1.10),
                1.15 * live_rel, "ataque e defesa nos últimos cinco", "live",
            )
            expected_totals.append(expected_home + expected_away)
            parity_values.append(math.exp(-abs(expected_home - expected_away) / 0.82))
        draw_rates.extend([
            _safe_float(features.get("live_recent_home_draw_rate")),
            _safe_float(features.get("live_recent_away_draw_rate")),
        ])
    if _safe_float(features.get("sofa_pre_available")) > 0:
        pre_rel = source_reliability("sofa_pre_home_games", "sofa_pre_away_games")
        add_pair("sofa_pre_home_ppg", "sofa_pre_away_ppg", 1.2, 1.0, "forma SofaScore", pre_rel, "pre")
        add_pair("sofa_pre_home_avg_rating", "sofa_pre_away_avg_rating", 0.35, 0.45, "rating recente", pre_rel, "pre")
        position_adv = _safe_float(features.get("sofa_pre_position_adv_home"))
        if position_adv:
            add_signal(math.tanh(position_adv / 8.0), 0.65 * pre_rel,
                       "posição na tabela", "pre")
        draw_rates.extend([
            _safe_float(features.get("sofa_pre_home_draw_rate")),
            _safe_float(features.get("sofa_pre_away_draw_rate")),
        ])
    if _safe_float(features.get("sofa_roll_available")) > 0:
        roll_rel = source_reliability("sofa_roll_home_games", "sofa_roll_away_games")
        add_pair("sofa_roll_home_ppg", "sofa_roll_away_ppg", 1.2, 0.85, "resultados recentes detalhados", roll_rel, "roll")
        dominance_rel = (
            min(1.0, min(metric_samples("home", "dominance"), metric_samples("away", "dominance")) / 5)
            if coverage_aware else roll_rel
        )
        add_pair("sofa_roll_home_dominance_avg", "sofa_roll_away_dominance_avg", 0.25, 0.85, "domínio recente", dominance_rel, "roll")
        home_gf = _safe_float(features.get("sofa_roll_home_goals_for_avg"))
        home_ga = _safe_float(features.get("sofa_roll_home_goals_against_avg"))
        away_gf = _safe_float(features.get("sofa_roll_away_goals_for_avg"))
        away_ga = _safe_float(features.get("sofa_roll_away_goals_against_avg"))
        if max(home_gf, home_ga, away_gf, away_ga) > 0:
            goals_home = (home_gf + away_ga) / 2.0
            goals_away = (away_gf + home_ga) / 2.0
            add_signal(
                math.tanh((goals_home - goals_away) / 1.15),
                0.90 * roll_rel, "gols recentes contra a defesa", "roll",
            )
            expected_totals.append(goals_home + goals_away)
            parity_values.append(math.exp(-abs(goals_home - goals_away) / 0.85))
        home_xg = _safe_float(features.get("sofa_roll_home_xg_for_avg"))
        home_xga = _safe_float(features.get("sofa_roll_home_xg_against_avg"))
        away_xg = _safe_float(features.get("sofa_roll_away_xg_for_avg"))
        away_xga = _safe_float(features.get("sofa_roll_away_xg_against_avg"))
        xg_rel = (
            min(1.0, min(metric_samples("home", "xg"), metric_samples("away", "xg")) / 5)
            if coverage_aware else roll_rel
        )
        if ((coverage_aware and xg_rel > 0)
                or (not coverage_aware and max(home_xg, home_xga, away_xg, away_xga) > 0)):
            expected_home = (home_xg + away_xga) / 2.0
            expected_away = (away_xg + home_xga) / 2.0
            add_signal(
                math.tanh((expected_home - expected_away) / 1.05), 1.20 * xg_rel,
                "xG contra defesa recente", "roll",
            )
            expected_totals.append(expected_home + expected_away)
            parity_values.append(math.exp(-abs(expected_home - expected_away) / 0.80))
        draw_rates.extend([
            _safe_float(features.get("sofa_roll_home_draw_rate")),
            _safe_float(features.get("sofa_roll_away_draw_rate")),
        ])

    if _safe_float(features.get("sofa_h2h_available")) > 0:
        h2h_games = max(0.0, _safe_float(features.get("sofa_h2h_games")))
        h2h_rel = _clamp(h2h_games / 4.0)
        source_reliabilities.append(h2h_rel)
        source_min_games.append(h2h_games)
        add_pair("sofa_h2h_home_ppg", "sofa_h2h_away_ppg", 1.4, 0.60,
                 "retrospecto direto", h2h_rel, "h2h")
        draw_rates.append(_safe_float(features.get("sofa_h2h_draw_rate")))

    weight_sum = sum(weight for _, weight, _ in signals)
    side_strength = (
        sum(value * weight for value, weight, _ in signals) / weight_sum
        if weight_sum else 0.0
    )
    source_directions = {
        source: sum(value * weight for value, weight in values)
                / max(1e-9, sum(weight for _, weight in values))
        for source, values in source_signals.items()
    }
    positive = sum(max(0.0, value) for value in source_directions.values())
    negative = sum(max(0.0, -value) for value in source_directions.values())
    source_disagreement = (
        2.0 * min(positive, negative) / (positive + negative)
        if positive + negative > 0 else 0.0
    )
    sample_reliability = (
        sum(source_reliabilities) / len(source_reliabilities)
        if source_reliabilities else 0.0
    )
    quality_denominator = 7.5 if _safe_float(features.get("live_recent_available")) > 0 else 5.5
    quality = _clamp(weight_sum / quality_denominator) * (0.35 + 0.65 * sample_reliability)
    valid_draw_rates = [rate for rate in draw_rates if 0 <= rate <= 1]
    form_draw = sum(valid_draw_rates) / len(valid_draw_rates) if valid_draw_rates else 0.27
    league_draw = _safe_float(features.get("liga_prior_empate"), 0.27)
    if not 0.05 <= league_draw <= 0.60:
        league_draw = 0.27
    low_scoring = max(
        _safe_float(features.get("context_baixa_intensidade")),
        _safe_float(features.get("sofa_pre_home_low_scoring_trend")),
        _safe_float(features.get("sofa_pre_away_low_scoring_trend")),
    )
    if expected_totals:
        external_total = sum(expected_totals) / len(expected_totals)
        low_scoring = max(low_scoring, _clamp((3.10 - external_total) / 1.70))
    parity = sum(parity_values) / len(parity_values) if parity_values else 0.0
    parity = max(
        parity,
        _safe_float(features.get("context_paridade_ppg")),
        _safe_float(features.get("context_paridade_mando")),
        _safe_float(features.get("context_equilibrio_ataques")),
        1.0 - abs(side_strength),
    )
    composite_draw = _safe_float(features.get("context_empate_composto"), league_draw)
    if not 0.05 <= composite_draw <= 0.60:
        composite_draw = league_draw
    parity_prior = 0.16 + 0.20 * parity
    low_score_prior = 0.18 + 0.18 * low_scoring
    raw_draw_risk = _clamp(
        0.25 * base[1] + 0.20 * league_draw + 0.20 * form_draw
        + 0.15 * composite_draw + 0.12 * parity_prior + 0.08 * low_score_prior,
        0.08, 0.55,
    )
    draw_corroboration = sum((
        form_draw >= 0.34,
        league_draw >= 0.30,
        parity >= 0.72,
        low_scoring >= 0.55,
        composite_draw >= 0.32,
    ))
    # Um único sinal (especialmente uma forma de 1 jogo) não pode transformar
    # empate em narrativa dominante. Exigimos sinais independentes convergentes.
    draw_evidence = min(1.0, draw_corroboration / 3.0) * (
        0.40 + 0.60 * sample_reliability
    )
    conservative_draw = 0.55 * base[1] + 0.45 * league_draw
    draw_risk = _clamp(
        conservative_draw + draw_evidence * (raw_draw_risk - conservative_draw),
        0.08, 0.55,
    )

    remaining = 1.0 - draw_risk
    home_share = 1.0 / (1.0 + math.exp(-2.2 * side_strength))
    context_proba = [remaining * home_share, draw_risk, remaining * (1.0 - home_share)]
    base_pick = int(max(range(3), key=lambda idx: base[idx]))
    chosen_side = 1.0 if base_pick == 0 else (-1.0 if base_pick == 2 else 0.0)
    conflict = _clamp(max(0.0, -chosen_side * side_strength) * quality)
    has_external_context = bool(signals or valid_draw_rates)
    alpha = (
        SOFASCORE_CONTEXT_BLEND * (0.35 + 0.65 * quality)
        if has_external_context else 0.0
    )
    # Um conflito forte não pode ser apenas anotado e depois ignorado. O piso
    # garante influência material, ainda limitada para o contexto não substituir
    # sozinho o classificador.
    if conflict >= 0.18 and sample_reliability >= 0.60:
        alpha = max(alpha, min(0.35, SOFASCORE_CONTEXT_BLEND))
    if (draw_risk >= 0.34 and form_draw >= 0.35
            and draw_corroboration >= 2 and sample_reliability >= 0.60):
        alpha = max(alpha, min(0.28, SOFASCORE_CONTEXT_BLEND))
    # Se o próprio campeão já colocou empate no topo e outros sinais também o
    # corroboram, uma única fonte direcional de qualidade média não pode virar
    # a classe para casa/fora por alguns décimos de ponto percentual.
    if base_pick == 1 and draw_corroboration >= 2 and quality < 0.65:
        alpha = 0.0
    # Fontes independentes em direções opostas representam incerteza real, não
    # autorização para uma fonte isolada virar o lado escolhido pelo campeão.
    roll_direction = source_directions.get("roll", 0.0)
    h2h_direction = source_directions.get("h2h", 0.0)
    live_direction = source_directions.get("live", 0.0)
    detailed_consensus_side = (
        1 if sum(value > 0.08 for value in (live_direction, roll_direction, h2h_direction)) >= 2 else
        (-1 if sum(value < -0.08 for value in (live_direction, roll_direction, h2h_direction)) >= 2 else 0)
    )
    base_side = 1 if base_pick == 0 else (-1 if base_pick == 2 else 0)
    reliable_detailed_consensus = (
        source_disagreement >= 0.65
        and detailed_consensus_side != 0
        and detailed_consensus_side == base_side
        and (
            (_safe_float(features.get("sofa_h2h_games")) >= 2
             and _safe_float(features.get("sofa_roll_home_games")) >= 5
             and _safe_float(features.get("sofa_roll_away_games")) >= 5)
            or (
                _safe_float(features.get("live_recent_home_games")) >= 5
                and _safe_float(features.get("live_recent_away_games")) >= 5
                and _safe_float(features.get("sofa_roll_home_games")) >= 5
                and _safe_float(features.get("sofa_roll_away_games")) >= 5
            )
        )
    )
    if reliable_detailed_consensus:
        alpha = 0.0
    elif source_disagreement >= 0.35:
        alpha *= max(0.35, 1.0 - 0.75 * source_disagreement)
    alpha = _clamp(alpha, 0.0, 0.45)
    final = [(1.0 - alpha) * base[i] + alpha * context_proba[i] for i in range(3)]
    final_total = sum(final) or 1.0
    final = [value / final_total for value in final]

    # Em uma volta de mata-mata, o time que venceu a ida tem incentivo para
    # proteger a classificação: empate deixa de ser equivalente a uma partida
    # comum. Esse ajuste só existe com confronto reverso recente documentado.
    home_deficit = _safe_float(features.get("sofa_h2h_home_aggregate_deficit"))
    away_deficit = _safe_float(features.get("sofa_h2h_away_aggregate_deficit"))
    recent_reverse = _safe_float(features.get("sofa_h2h_recent_reverse")) > 0
    knockout_context = max(
        _safe_float(features.get("is_knockout")),
        _safe_float(features.get("is_qualifier")),
        _safe_float(features.get("is_volta")),
    ) > 0
    tactical_adjustment = 0.0
    if recent_reverse and knockout_context and home_deficit > 0:
        tactical_adjustment = min(final[2] * 0.35, 0.055 + 0.015 * home_deficit)
        final[2] -= tactical_adjustment
        final[1] += tactical_adjustment * 0.82
        final[0] += tactical_adjustment * 0.18
    elif recent_reverse and knockout_context and away_deficit > 0:
        tactical_adjustment = min(final[0] * 0.35, 0.055 + 0.015 * away_deficit)
        final[0] -= tactical_adjustment
        final[1] += tactical_adjustment * 0.82
        final[2] += tactical_adjustment * 0.18
    low_sample_draw_override = bool(
        abs(base[0] - base[2]) <= 0.01
        and draw_corroboration >= 4
        and draw_risk >= 0.32
        and sample_reliability <= 0.35
    )
    if low_sample_draw_override and final[1] <= max(final[0], final[2]):
        needed = max(final[0], final[2]) - final[1] + 0.002
        transfer_home = needed * final[0] / max(1e-9, final[0] + final[2])
        transfer_away = needed - transfer_home
        final[0] -= transfer_home
        final[2] -= transfer_away
        final[1] += needed
    # Empates quase empatados no topo eram sistematicamente perdidos pelo
    # argmax por décimos de ponto percentual. A proteção só atua quando o risco
    # independente é alto e ao menos dois sinais pré-jogo corroboram; no replay
    # cronológico ela alterou apenas casos na borda, sem promover empates por
    # simples paridade do classificador.
    near_tie_draw_override = bool(not legacy_v5 and
        final[1] >= 0.27
        and max(final[0], final[2]) - final[1] <= 0.01
        and draw_corroboration >= 2
        and draw_risk >= 0.32
    )
    if near_tie_draw_override and final[1] <= max(final[0], final[2]):
        needed = max(final[0], final[2]) - final[1] + 0.002
        transfer_home = needed * final[0] / max(1e-9, final[0] + final[2])
        transfer_away = needed - transfer_home
        final[0] -= transfer_home
        final[2] -= transfer_away
        final[1] += needed

    # Decisão de lado baseada no duelo efetivo. O replay dos radares anteriores
    # mostrou que PPG genérico isolado é quase aleatório, enquanto forma no mando
    # equivalente e ataque contra a defesa adversária generalizam melhor. O
    # classificador continua como estabilizador; o contexto não altera empates
    # que já estejam no topo.
    base_side_logit = math.log(max(1e-9, final[0]) / max(1e-9, final[2]))
    venue_signal = 0.0
    venue_available = (
        _safe_float(features.get("live_recent_home_same_venue_games")) >= 2
        and _safe_float(features.get("live_recent_away_same_venue_games")) >= 2
    )
    if venue_available:
        venue_signal = math.tanh((
            _safe_float(features.get("live_recent_home_same_venue_ppg"))
            - _safe_float(features.get("live_recent_away_same_venue_ppg"))
        ) / 1.20)
    attack_defense_signal = 0.0
    live_goal_values = (
        _safe_float(features.get("live_recent_home_goals_for_avg")),
        _safe_float(features.get("live_recent_home_goals_against_avg")),
        _safe_float(features.get("live_recent_away_goals_for_avg")),
        _safe_float(features.get("live_recent_away_goals_against_avg")),
    )
    attack_defense_available = max(live_goal_values) > 0
    if attack_defense_available:
        home_gf, home_ga, away_gf, away_ga = live_goal_values
        expected_home = (home_gf + away_ga) / 2.0
        expected_away = (away_gf + home_ga) / 2.0
        attack_defense_signal = math.tanh((expected_home - expected_away) / 1.10)
    roll_signal = 0.0
    roll_duel_available = (
        _safe_float(features.get("sofa_roll_home_games")) >= 3
        and _safe_float(features.get("sofa_roll_away_games")) >= 3
    )
    if roll_duel_available:
        roll_signal = math.tanh((
            _safe_float(features.get("sofa_roll_home_ppg"))
            - _safe_float(features.get("sofa_roll_away_ppg"))
        ) / 1.20)
    duel_side_score = (
        0.75 * base_side_logit
        + 1.00 * venue_signal
        + 0.75 * attack_defense_signal
        + 0.25 * roll_signal
    )
    side_before_duel = 0 if final[0] >= final[2] else 2
    draw_before_duel = final[1] >= max(final[0], final[2])
    side_after_duel = 0 if duel_side_score >= 0 else 2
    duel_side_override = bool(not legacy_v5 and
        not draw_before_duel
        and side_after_duel != side_before_duel
        and (venue_available or attack_defense_available or roll_duel_available)
    )
    if duel_side_override:
        final[0], final[2] = final[2], final[0]
    ordered = sorted(final, reverse=True)
    probability_margin = ordered[0] - ordered[1]
    normalized_entropy = -sum(
        probability * math.log(max(probability, 1e-12)) for probability in final
    ) / math.log(3.0)

    if draw_risk >= 0.30:
        reasons.append(f"risco de empate {draw_risk:.0%}")
    if conflict >= 0.18:
        reasons.append("forma/contexto contradiz o lado originalmente favorito pelo modelo")
    if source_disagreement >= 0.35:
        reasons.append("fontes pré-jogo divergem; influência contextual reduzida")
    if tactical_adjustment > 0:
        reasons.append("volta de mata-mata: líder do agregado pode administrar empate")
    if low_sample_draw_override:
        reasons.append("lados empatados e quatro sinais de empate; proteção aplicada à amostra curta")
    if near_tie_draw_override:
        reasons.append("empate a até 1 p.p. do topo com risco e sinais independentes corroborados")
    if duel_side_override:
        reasons.append("duelo de mando e ataque contra defesa corrigiu o lado do classificador")
    if signals:
        direction = "mandante" if side_strength > 0.08 else ("visitante" if side_strength < -0.08 else "equilíbrio")
        reasons.append(f"evidência contextual aponta {direction}")
        if expected_totals:
            reasons.append(
                f"ataque contra defesa projeta {sum(expected_totals) / len(expected_totals):.2f} gols"
            )
    else:
        reasons.append("contexto externo insuficiente; probabilidades preservadas")
    if source_min_games and min(source_min_games) < 3:
        reasons.append("amostra recente curta; influência contextual reduzida")
    final_pick = int(max(range(3), key=lambda idx: final[idx]))
    final_side = 1.0 if final_pick == 0 else (-1.0 if final_pick == 2 else 0.0)
    selection_conflict = _clamp(
        max(0.0, -final_side * side_strength) * quality
    )
    return {
        "base_probabilities": base,
        "probabilities": final,
        "information_quality": quality,
        "draw_risk": draw_risk,
        "context_conflict": conflict,
        "base_context_conflict": conflict,
        "selection_conflict": selection_conflict,
        "side_strength": side_strength,
        "source_disagreement": source_disagreement,
        "source_directions": source_directions,
        "blend": alpha,
        "sample_reliability": sample_reliability,
        "recent_games_min": min(source_min_games) if source_min_games else 0.0,
        "detailed_both_available": float(
            min(metric_samples("home", "xg"), metric_samples("away", "xg")) > 0
            if coverage_aware else (
                _safe_float(features.get("sofa_roll_home_games")) > 0
                and _safe_float(features.get("sofa_roll_away_games")) > 0
            )
        ),
        "live_recent_both_available": float(
            _safe_float(features.get("live_recent_home_games")) >= 3
            and _safe_float(features.get("live_recent_away_games")) >= 3
        ),
        "draw_corroboration": int(draw_corroboration),
        "probability_margin": probability_margin,
        "normalized_entropy": normalized_entropy,
        "second_leg_tactical_adjustment": tactical_adjustment,
        "low_sample_draw_override": float(low_sample_draw_override),
        "near_tie_draw_override": float(near_tie_draw_override),
        "duel_side_override": float(duel_side_override),
        "duel_side_score": float(duel_side_score),
        "duel_venue_signal": float(venue_signal),
        "duel_attack_defense_signal": float(attack_defense_signal),
        "duel_roll_signal": float(roll_signal),
        "competition_volatility": _clamp(
            _safe_float(features.get("competition_volatility"), 0.08)
        ),
        "reasons": reasons,
        "analysis_version": (
            "sofa-context-v5-quality" if coverage_aware else
            ACTIVE_ANALYSIS_VERSION if normalized_profile == "active-v5"
            else ("sofa-context-v5-replay" if legacy_v5 else ANALYSIS_VERSION)
        ),
    }


def save_prediction_snapshot(
    db_path: str,
    match_id: Any,
    start_timestamp: int | None,
    features: dict[str, Any],
    analysis: dict[str, Any],
    predicted_outcome: str,
) -> None:
    """Congela exatamente o que existia no momento da previsão."""
    captured_at = int(time.time())
    validated_start = _safe_timestamp(start_timestamp)
    # Sem um cutoff verificável, o registro não pode ser chamado de pré-jogo.
    # A segunda barreira também protege chamadas futuras fora do radar normal.
    if not validated_start or captured_at >= validated_start:
        return
    init_sofascore_db(db_path)
    numeric_features = {
        str(key): _safe_float(value)
        for key, value in features.items()
        if isinstance(value, (int, float))
    }
    numeric_features["_feature_version"] = 6
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO ml_prediction_snapshots
               (match_id, captured_at, start_timestamp, feature_version,
                features_json, base_probabilities_json, final_probabilities_json,
                predicted_outcome, information_quality, draw_risk,
                context_conflict, analysis_json, source_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(match_id), captured_at, validated_start, 6,
                json.dumps(numeric_features, ensure_ascii=False),
                json.dumps(analysis.get("base_probabilities") or []),
                json.dumps(analysis.get("probabilities") or []),
                str(predicted_outcome),
                _safe_float(analysis.get("information_quality")),
                _safe_float(analysis.get("draw_risk")),
                _safe_float(analysis.get("context_conflict")),
                json.dumps(analysis, ensure_ascii=False),
                str(analysis.get("analysis_version") or ANALYSIS_VERSION),
            ),
        )
        conn.commit()


def load_prediction_snapshot_features(db_path: str, match_id: Any) -> dict[str, Any]:
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT features_json FROM ml_prediction_snapshots WHERE match_id=?",
            (str(match_id),),
        ).fetchone()
    try:
        result = json.loads(row[0]) if row else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        result = {}
    return result if isinstance(result, dict) else {}


def load_prediction_snapshot_metadata(db_path: str, match_id: Any) -> dict[str, Any]:
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            """SELECT information_quality, draw_risk, context_conflict,
                      source_version, analysis_json
               FROM ml_prediction_snapshots WHERE match_id=?""",
            (str(match_id),),
        ).fetchone()
    if not row:
        return {}
    try:
        analysis = json.loads(row[4] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        analysis = {}
    analysis = analysis if isinstance(analysis, dict) else {}
    probabilities = analysis.get("probabilities") or []
    if len(probabilities) == 3:
        final_pick = int(max(range(3), key=lambda idx: _safe_float(probabilities[idx])))
        final_side = 1.0 if final_pick == 0 else (-1.0 if final_pick == 2 else 0.0)
        legacy_selection_conflict = _clamp(
            max(0.0, -final_side * _safe_float(analysis.get("side_strength")))
            * _safe_float(analysis.get("information_quality", row[0]))
        )
    else:
        legacy_selection_conflict = 0.0
    return {
        "information_quality": _safe_float(row[0]),
        "draw_risk": _safe_float(row[1]),
        "context_conflict": _safe_float(row[2]),
        "base_context_conflict": _safe_float(
            analysis.get("base_context_conflict", row[2])
        ),
        "selection_conflict": _safe_float(
            analysis.get("selection_conflict", legacy_selection_conflict)
        ),
        "source_disagreement": _safe_float(analysis.get("source_disagreement")),
        "source_version": str(row[3] or ANALYSIS_VERSION),
        "sample_reliability": _safe_float(analysis.get("sample_reliability")),
        "recent_games_min": _safe_float(analysis.get("recent_games_min")),
        "detailed_both_available": _safe_float(analysis.get("detailed_both_available")),
        "live_recent_both_available": _safe_float(
            analysis.get("live_recent_both_available")
        ),
        "draw_corroboration": int(_safe_float(analysis.get("draw_corroboration"))),
        "probability_margin": _safe_float(analysis.get("probability_margin")),
        "normalized_entropy": _safe_float(analysis.get("normalized_entropy")),
        "competition_volatility": _safe_float(analysis.get("competition_volatility")),
        "analysis": analysis,
    }


def _statistics_metrics(payload: dict[str, Any] | None) -> tuple[dict[str, float], float]:
    if not isinstance(payload, dict):
        return {}, 0.0
    periods = (payload or {}).get("statistics") or []
    if not isinstance(periods, list):
        return {}, 0.0
    all_period = next((item for item in periods if isinstance(item, dict) and item.get("period") == "ALL"), None)
    if not all_period:
        return {}, 0.0
    values: dict[str, tuple[float, float]] = {}
    available: dict[str, tuple[bool, bool]] = {}
    for group in all_period.get("groups") or []:
        if not isinstance(group, dict):
            continue
        for item in group.get("statisticsItems") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("name") or "")
            if key and key not in values:
                raw_pair = (item.get("homeValue", item.get("home")),
                            item.get("awayValue", item.get("away")))
                def reported(raw):
                    if raw is None or isinstance(raw, bool):
                        return False
                    try:
                        return math.isfinite(float(str(raw).strip().rstrip("%").replace(",", ".")))
                    except (TypeError, ValueError):
                        return False
                available[key] = tuple(reported(raw) for raw in raw_pair)
                values[key] = (
                    _safe_float(item.get("homeValue", item.get("home"))),
                    _safe_float(item.get("awayValue", item.get("away"))),
                )
    aliases = {
        "possession": "ballPossession",
        "xg": "expectedGoals",
        "xgot": "expectedGoalsOnTarget",
        "big_chances": "bigChanceCreated",
        "shots": "totalShotsOnGoal",
        "shots_on_target": "shotsOnGoal",
        "box_shots": "shotsInsideBox",
        "box_touches": "touchesInOppBox",
        "final_third_entries": "finalThirdEntries",
        "errors_to_goal": "errorsLeadToGoal",
    }
    metrics: dict[str, float] = {}
    present = 0
    for output, key in aliases.items():
        home, away = values.get(key, (0.0, 0.0))
        metrics[f"home_{output}"] = home
        metrics[f"away_{output}"] = away
        home_present, away_present = available.get(key, (False, False))
        metrics[f"home_{output}_available"] = float(home_present)
        metrics[f"away_{output}_available"] = float(away_present)
        present += (int(home_present) + int(away_present)) / 2
    return metrics, present / len(aliases)


def _shotmap_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    shots = (payload or {}).get("shotmap") or []
    result: Counter[str] = Counter()
    for shot in shots:
        side = "home" if bool(shot.get("isHome")) else "away"
        result[f"{side}_shotmap_shots"] += 1
        result[f"{side}_shotmap_{str(shot.get('shotType') or 'unknown').lower()}"] += 1
        if str(shot.get("situation") or "").lower() in {"corner", "free-kick", "set-piece", "penalty"}:
            result[f"{side}_set_piece_shots"] += 1
    return {key: float(value) for key, value in result.items()}


def _incident_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    """Extrai eventos que alteram radicalmente um jogo, sobretudo expulsões."""
    incidents = sorted(
        (payload or {}).get("incidents") or [],
        key=lambda item: (_safe_float(item.get("time")),
                          _safe_float(item.get("addedTime"))),
    )
    result: Counter[str] = Counter()
    home_goals = away_goals = 0.0
    for incident in incidents:
        side = "home" if bool(incident.get("isHome")) else "away"
        kind = str(incident.get("incidentType") or "").lower()
        incident_class = str(incident.get("incidentClass") or "").lower()
        minute = _safe_float(incident.get("time")) + _safe_float(
            incident.get("addedTime")
        ) / 100.0
        if kind == "card" and incident_class in {"red", "yellowred"}:
            result[f"{side}_red_cards"] += 1
            first_key = f"{side}_first_red_minute"
            if not result.get(first_key):
                result[first_key] = minute
            if home_goals == away_goals:
                result[f"{side}_red_while_level"] = 1
        if kind == "goal":
            if side == "home":
                home_goals += 1
            else:
                away_goals += 1
            result[f"{side}_last_goal_minute"] = minute
            if bool(incident.get("from") == "penalty") or "penalty" in str(
                incident.get("incidentClass") or ""
            ).lower():
                result[f"{side}_penalty_goals"] += 1
    return {key: float(value) for key, value in result.items()}


def _graph_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    points = [_safe_float(item.get("value")) for item in (payload or {}).get("graphPoints") or []]
    if not points:
        return {}
    home_pressure = sum(max(0.0, value) for value in points)
    away_pressure = sum(max(0.0, -value) for value in points)
    total = home_pressure + away_pressure
    return {
        "home_pressure_share": home_pressure / total if total else 0.5,
        "away_pressure_share": away_pressure / total if total else 0.5,
        "pressure_points": float(len(points)),
    }


def _lineup_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload:
        return {}
    result = {"lineups_confirmed": float(bool(payload.get("confirmed")))}
    for side in ("home", "away"):
        players = (payload.get(side) or {}).get("players") or []
        ratings, starter_ratings, errors = [], [], 0.0
        for item in players:
            stats = item.get("statistics") or {}
            rating = _safe_float(stats.get("rating"))
            if rating > 0:
                ratings.append(rating)
                if not item.get("substitute"):
                    starter_ratings.append(rating)
            errors += _safe_float(stats.get("errorLeadToAGoal"))
        result[f"{side}_lineup_avg_rating"] = sum(ratings) / len(ratings) if ratings else 0.0
        result[f"{side}_starter_avg_rating"] = (
            sum(starter_ratings) / len(starter_ratings) if starter_ratings else 0.0
        )
        result[f"{side}_player_errors_to_goal"] = errors
    return result


def _average_position_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    """Resume o mapa de posição médio sem fazer uma chamada por jogador."""
    result: dict[str, float] = {}
    for side in ("home", "away"):
        entries = (payload or {}).get(side) or []
        weighted = []
        for item in entries:
            x, y = _safe_float(item.get("averageX")), _safe_float(item.get("averageY"))
            points = max(1.0, _safe_float(item.get("pointsCount"), 1.0))
            if x or y:
                weighted.append((x, y, points))
        total = sum(item[2] for item in weighted)
        if total:
            mean_x = sum(x * points for x, _, points in weighted) / total
            mean_y = sum(y * points for _, y, points in weighted) / total
            spread_x = math.sqrt(sum(((x - mean_x) ** 2) * points for x, _, points in weighted) / total)
            spread_y = math.sqrt(sum(((y - mean_y) ** 2) * points for _, y, points in weighted) / total)
        else:
            mean_x = mean_y = spread_x = spread_y = 0.0
        result[f"{side}_heatmap_mean_x"] = mean_x
        result[f"{side}_heatmap_mean_y"] = mean_y
        result[f"{side}_heatmap_length"] = spread_x
        result[f"{side}_heatmap_width"] = spread_y
        result[f"{side}_heatmap_players"] = float(len(weighted))
    return result


def _share(home: float, away: float) -> float:
    total = max(0.0, home) + max(0.0, away)
    return max(0.0, home) / total if total > 0 else 0.5


def _dominance(metrics: dict[str, float]) -> tuple[float, float, float]:
    components = [
        ("xg", 0.28), ("xgot", 0.13), ("big_chances", 0.16),
        ("shots_on_target", 0.15), ("box_shots", 0.11),
        ("box_touches", 0.08), ("shots", 0.05), ("possession", 0.02),
    ]
    weighted, used = 0.0, 0.0
    for name, weight in components:
        home = _safe_float(metrics.get(f"home_{name}"))
        away = _safe_float(metrics.get(f"away_{name}"))
        if home > 0 or away > 0:
            weighted += _share(home, away) * weight
            used += weight
    if "home_pressure_share" in metrics:
        weighted += _safe_float(metrics["home_pressure_share"]) * 0.02
        used += 0.02
    home_dominance = weighted / used if used else 0.5
    return home_dominance, 1.0 - home_dominance, _clamp(used / sum(w for _, w in components))


def _outcomes(event: dict[str, Any]) -> tuple[str, str, float, float]:
    from football_results import regulation_score
    score = regulation_score(event, require_finished=False)
    if score is None:
        raise ValueError("Placar de 90 minutos ausente ou ambíguo")
    home_score, away_score = map(float, score)
    actual = "MANDANTE" if home_score > away_score else ("VISITANTE" if away_score > home_score else "EMPATE")
    return actual, f"{int(home_score)}-{int(away_score)}", home_score, away_score


def _prediction_side(predicted: str, event: dict[str, Any]) -> str:
    text = normalize_team_name(predicted)
    if "empate" in str(predicted).lower():
        return "EMPATE"
    home = normalize_team_name((event.get("homeTeam") or {}).get("name"))
    away = normalize_team_name((event.get("awayTeam") or {}).get("name"))
    if text and home and (_similarity(text, home) >= 0.72 or home in text or text in home):
        return "MANDANTE"
    if text and away and (_similarity(text, away) >= 0.72 or away in text or text in away):
        return "VISITANTE"
    if "mandante" in str(predicted).lower():
        return "MANDANTE"
    if "visitante" in str(predicted).lower():
        return "VISITANTE"
    return "DESCONHECIDO"


def classify_postmortem(
    predicted_side: str,
    actual_side: str,
    prediction_status: str,
    metrics: dict[str, float],
    data_coverage: float,
    snapshot_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Distingue erro de processo, empate mal lido e zebra por variância."""
    home_dom, away_dom, dominance_coverage = _dominance(metrics)
    coverage = _clamp(0.65 * data_coverage + 0.35 * dominance_coverage)
    chosen_dom = home_dom if predicted_side == "MANDANTE" else (
        away_dom if predicted_side == "VISITANTE" else 0.5
    )
    opponent_dom = 1.0 - chosen_dom
    reasons: list[str] = []
    is_green = str(prediction_status).startswith("GREEN") or predicted_side == actual_side
    snapshot_analysis = snapshot_analysis or {}
    probabilities = snapshot_analysis.get("probabilities") or []
    draw_probability = (
        _safe_float(probabilities[1]) if len(probabilities) == 3 else 0.0
    )
    draw_margin = (
        max(_safe_float(probabilities[0]), _safe_float(probabilities[2]))
        - draw_probability if len(probabilities) == 3 else 1.0
    )
    side_strength = _safe_float(snapshot_analysis.get("side_strength"))
    if "selection_conflict" in snapshot_analysis:
        conflict = _safe_float(snapshot_analysis.get("selection_conflict"))
    else:
        # Snapshots v1-v3 guardavam conflito contra o palpite-base. Para não
        # chamar uma correção contextual bem-sucedida de "má leitura", o
        # legado é reconstruído contra a seleção efetivamente publicada.
        chosen_side = 1.0 if predicted_side == "MANDANTE" else (
            -1.0 if predicted_side == "VISITANTE" else 0.0
        )
        conflict = _clamp(
            max(0.0, -chosen_side * side_strength)
            * _safe_float(snapshot_analysis.get("information_quality"))
        )
    information_quality = _safe_float(snapshot_analysis.get("information_quality"))
    sample_reliability = _safe_float(snapshot_analysis.get("sample_reliability"))
    draw_corroboration = int(_safe_float(snapshot_analysis.get("draw_corroboration")))
    source_disagreement = _safe_float(snapshot_analysis.get("source_disagreement"))
    prematch_conflict = (
        conflict >= 0.18 and information_quality >= 0.25
        and sample_reliability >= 0.60
    )
    # O antigo ``draw_risk`` isolado não se mostrou calibrado nos radares
    # concluídos. Um empate só recebe peso de erro previsível quando a própria
    # probabilidade estava perto do topo e três sinais independentes
    # corroboravam. Isso evita ensinar o modelo com narrativas pós-resultado.
    draw_was_foreseeable = (
        draw_probability >= 0.27 and draw_margin <= 0.03
        and draw_corroboration >= 3 and sample_reliability >= 0.60
    )
    chosen_prefix = "home" if predicted_side == "MANDANTE" else (
        "away" if predicted_side == "VISITANTE" else ""
    )
    other_prefix = "away" if chosen_prefix == "home" else "home"
    incident_variance = bool(
        chosen_prefix and not is_green
        and _safe_float(metrics.get(f"{chosen_prefix}_red_while_level")) > 0
        and _safe_float(metrics.get(f"{chosen_prefix}_red_cards"))
            > _safe_float(metrics.get(f"{other_prefix}_red_cards"))
    )
    chosen_xg = _safe_float(metrics.get(f"{chosen_prefix}_xg")) if chosen_prefix else 0.0
    other_xg = _safe_float(metrics.get(f"{other_prefix}_xg")) if chosen_prefix else 0.0
    chosen_shots = _safe_float(metrics.get(f"{chosen_prefix}_shots")) if chosen_prefix else 0.0
    other_shots = _safe_float(metrics.get(f"{other_prefix}_shots")) if chosen_prefix else 0.0
    decisive_core_error = bool(
        chosen_prefix and not is_green
        and other_xg >= chosen_xg + 0.75
        and other_shots >= chosen_shots + 5
    )

    if incident_variance:
        verdict, weight, margin = "MATCH_INCIDENT_VARIANCE", 0.65, 0.05
        reasons.append("a seleção sofreu expulsão com o placar empatado; evento não previsto")
    elif decisive_core_error:
        verdict, weight, margin = "MODEL_READING_ERROR", 1.35, 0.45
        reasons.append("xG e volume de finalizações confirmaram superioridade clara do adversário")
    elif coverage < 0.30:
        if not is_green and actual_side == "EMPATE" and draw_was_foreseeable:
            verdict, weight, margin = "DRAW_RISK_MISSED", 1.45, 0.55
            reasons.append("o risco de empate já estava elevado no pré-jogo")
        elif not is_green and prematch_conflict:
            verdict, weight, margin = "MODEL_READING_ERROR", 1.55, 0.65
            reasons.append("o contexto pré-jogo contrariava a seleção")
        elif not is_green and source_disagreement >= 0.45:
            verdict, weight, margin = "PREMATCH_SOURCE_CONFLICT", 0.90, 0.10
            reasons.append("fontes pré-jogo divergiam; erro não deve ser superponderado")
        else:
            verdict, weight, margin = "INSUFFICIENT_DATA", 1.0, 0.0
            reasons.append("SofaScore sem cobertura estatística suficiente")
    elif is_green:
        if chosen_dom >= 0.55:
            verdict, weight, margin = "GREEN_CONFIRMED", 1.0, 0.0
            reasons.append("resultado e desempenho confirmaram a leitura")
        elif chosen_dom <= 0.43:
            verdict, weight, margin = "GREEN_WITH_WARNING", 0.90, 0.05
            reasons.append("o acerto ocorreu apesar de desempenho inferior")
        else:
            verdict, weight, margin = "GREEN_BALANCED", 1.0, 0.0
            reasons.append("partida equilibrada, com resultado favorável")
    elif actual_side == "EMPATE":
        if chosen_dom >= 0.62:
            verdict, weight, margin = "DRAW_VARIANCE", 0.80, 0.10
            reasons.append("a seleção produziu mais, mas não converteu o domínio em vitória")
        elif chosen_dom <= 0.43 and prematch_conflict:
            verdict, weight, margin = "DRAW_PROCESS_ERROR", 1.70, 0.75
            reasons.append("inferioridade final confirmou a contradição existente no pré-jogo")
        elif draw_was_foreseeable:
            verdict, weight, margin = "DRAW_RISK_MISSED", 1.45, 0.55
            reasons.append("múltiplos sinais pré-jogo indicavam empate e foram subestimados")
        elif chosen_dom <= 0.40:
            verdict, weight, margin = "MODEL_READING_ERROR", 1.35, 0.45
            reasons.append("o adversário dominou um empate que contrariou a seleção")
        else:
            verdict, weight, margin = "DRAW_UNCERTAIN", 1.05, 0.15
            reasons.append("o empate não tinha evidência pré-jogo suficiente para virar erro forte")
    else:
        xg_chosen = metrics.get("home_xg", 0.0) if predicted_side == "MANDANTE" else metrics.get("away_xg", 0.0)
        xg_opp = metrics.get("away_xg", 0.0) if predicted_side == "MANDANTE" else metrics.get("home_xg", 0.0)
        if ((chosen_dom >= 0.62 and _safe_float(xg_chosen) >= _safe_float(xg_opp) + 0.30)
                or (chosen_dom >= 0.50
                    and _safe_float(xg_chosen) >= _safe_float(xg_opp) + 0.15)):
            verdict, weight, margin = "ZEBRA_VARIANCE", 0.75, 0.10
            reasons.append("a seleção dominou chances e xG, mas perdeu: variância/zebra")
        elif chosen_dom <= 0.42:
            if prematch_conflict:
                verdict, weight, margin = "MODEL_READING_ERROR", 1.75, 0.85
                reasons.append("o contexto pré-jogo já contrariava a seleção e o jogo confirmou")
            else:
                verdict, weight, margin = "MODEL_READING_ERROR", 1.30, 0.45
                reasons.append("o adversário foi superior, mas sem forte contradição pré-jogo documentada")
        else:
            verdict, weight, margin = "MIXED_ERROR", 1.10, 0.20
            reasons.append("desempenho misto: evidência insuficiente para punir fortemente o modelo")
    reasons.append(f"domínio da seleção {chosen_dom:.0%}; cobertura {coverage:.0%}")
    return {
        "verdict": verdict,
        "learning_weight": weight,
        "error_margin": margin,
        "chosen_dominance": chosen_dom,
        "opponent_dominance": opponent_dom,
        "process_score": chosen_dom,
        "data_coverage": coverage,
        "reasons": reasons,
    }


def reclassify_stored_postmortems(
    db_path: str, match_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Reaplica as regras atuais usando apenas snapshots/cache já gravados.

    Não acessa SofaScore nem RapidAPI. Isso impede que pesos produzidos por uma
    versão antiga da autópsia continuem contaminando o próximo retreino.
    """
    init_sofascore_db(db_path)
    identifiers = [str(value) for value in (match_ids or []) if str(value)]
    where = "WHERE COALESCE(m.source_version,'') != ?"
    params: list[Any] = [ANALYSIS_VERSION]
    if identifiers:
        where += f" AND m.match_id IN ({','.join('?' for _ in identifiers)})"
        params.extend(identifiers)
    with closing(_connect(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS training_weights (
            match_id TEXT PRIMARY KEY, peso REAL DEFAULT 1.0,
            data_ultima_atualizacao DATETIME, error_margin REAL DEFAULT 0)""")
        weight_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(training_weights)")
        }
        if "error_margin" not in weight_columns:
            conn.execute(
                "ALTER TABLE training_weights ADD COLUMN error_margin REAL DEFAULT 0"
            )
        rows = conn.execute(
            f"""SELECT m.match_id, m.prediction_status, m.predicted_outcome,
                       m.actual_outcome, m.metrics_json, m.data_coverage,
                       s.analysis_json
                FROM match_postmortems m
                LEFT JOIN ml_prediction_snapshots s ON s.match_id=m.match_id
                {where}""", params,
        ).fetchall()
        verdicts: Counter[str] = Counter()
        now = int(time.time())
        for match_id, status, predicted, actual, metrics_json, coverage, analysis_json in rows:
            try:
                metrics = json.loads(metrics_json or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metrics = {}
            try:
                analysis = json.loads(analysis_json or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                analysis = {}
            diagnosis = classify_postmortem(
                str(predicted), str(actual), str(status),
                metrics if isinstance(metrics, dict) else {},
                _safe_float(coverage), analysis if isinstance(analysis, dict) else {},
            )
            verdicts[diagnosis["verdict"]] += 1
            conn.execute(
                """UPDATE match_postmortems SET audited_at=?, verdict=?,
                          process_score=?, chosen_dominance=?, opponent_dominance=?,
                          learning_weight=?, error_margin=?, data_coverage=?,
                          reasons_json=?, source_version=? WHERE match_id=?""",
                (now, diagnosis["verdict"], diagnosis["process_score"],
                 diagnosis["chosen_dominance"], diagnosis["opponent_dominance"],
                 diagnosis["learning_weight"], diagnosis["error_margin"],
                 diagnosis["data_coverage"],
                 json.dumps(diagnosis["reasons"], ensure_ascii=False),
                 ANALYSIS_VERSION, str(match_id)),
            )
            conn.execute(
                """INSERT INTO training_weights
                   (match_id, peso, data_ultima_atualizacao, error_margin)
                   VALUES (?,?,datetime('now'),?)
                   ON CONFLICT(match_id) DO UPDATE SET peso=excluded.peso,
                     error_margin=excluded.error_margin,
                     data_ultima_atualizacao=excluded.data_ultima_atualizacao""",
                (str(match_id), diagnosis["learning_weight"], diagnosis["error_margin"]),
            )
            # A tela e os relatórios devem refletir a mesma classificação.
            try:
                conn.execute(
                    """UPDATE previsoes SET postmortem_verdict=?,
                              postmortem_process_score=?, postmortem_coverage=?,
                              analysis_version=? WHERE match_id=?""",
                    (diagnosis["verdict"], diagnosis["process_score"],
                     diagnosis["data_coverage"], ANALYSIS_VERSION, str(match_id)),
                )
            except sqlite3.OperationalError:
                pass
        conn.commit()
    return {"reclassified": len(rows), "verdicts": dict(verdicts),
            "analysis_version": ANALYSIS_VERSION, "http_requests": 0}


def _save_team_profiles(
    db_path: str,
    match_id: str,
    event: dict[str, Any],
    metrics: dict[str, float],
    home_dominance: float,
) -> None:
    home, away = event.get("homeTeam") or {}, event.get("awayTeam") or {}
    actual, _, home_goals, away_goals = _outcomes(event)
    start_ts = int(event.get("startTimestamp") or time.time())
    now = int(time.time())
    rows = []
    for side, team, opponent, is_home in (
        ("home", home, away, 1), ("away", away, home, 0),
    ):
        other = "away" if side == "home" else "home"
        gf, ga = (home_goals, away_goals) if is_home else (away_goals, home_goals)
        points = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
        observed: list[str] = []

        def measured(column: str, provider_side: str, metric: str) -> float | None:
            key = f"{provider_side}_{metric}"
            flag = metrics.get(f"{key}_available")
            if flag is None:
                # Older cached parsers materialized every missing field as 0.
                # Preserve pair zeros only if the other side proves coverage.
                opposite = "away" if provider_side == "home" else "home"
                available = (
                    _safe_float(metrics.get(key)) > 0
                    or _safe_float(metrics.get(f"{opposite}_{metric}")) > 0
                )
            else:
                available = bool(flag)
            if not available or metrics.get(key) is None:
                return None
            observed.append(column)
            return _safe_float(metrics[key])

        detailed = (
            measured("xg_for", side, "xg"), measured("xg_against", other, "xg"),
            measured("shots_for", side, "shots"), measured("shots_against", other, "shots"),
            measured("shots_on_target_for", side, "shots_on_target"),
            measured("shots_on_target_against", other, "shots_on_target"),
            measured("big_chances_for", side, "big_chances"),
            measured("big_chances_against", other, "big_chances"),
            measured("box_shots_for", side, "box_shots"),
            measured("box_shots_against", other, "box_shots"),
            measured("possession", side, "possession"),
        )
        dominance = (home_dominance if is_home else 1.0 - home_dominance) if observed else None
        if dominance is not None:
            observed.append("dominance")
        errors = measured("errors_to_goal", side, "errors_to_goal")
        rows.append((
            match_id, normalize_team_name(team.get("name")), str(team.get("id") or ""),
            str(team.get("name") or ""), normalize_team_name(opponent.get("name")),
            start_ts, is_home, gf, ga,
            *detailed, dominance, errors, points, int(actual == "EMPATE"), now,
            json.dumps(observed),
        ))
    with closing(_connect(db_path)) as conn:
        metric_columns = (
            "xg_for", "xg_against", "shots_for", "shots_against",
            "shots_on_target_for", "shots_on_target_against", "big_chances_for",
            "big_chances_against", "box_shots_for", "box_shots_against",
            "possession", "dominance", "errors_to_goal",
        )
        for index, row in enumerate(rows):
            previous = conn.execute(
                "SELECT " + ",".join(metric_columns) + ",metric_presence_json "
                "FROM sofascore_team_match_profiles WHERE match_id=? AND team_key=?",
                (row[0], row[1]),
            ).fetchone()
            if not previous:
                continue
            previous_values = dict(zip(metric_columns, previous))
            previous_presence = _profile_observed_metrics(previous_values, previous[-1])
            merged = list(row)
            observed = set(json.loads(merged[-1]))
            for offset, name in enumerate(metric_columns, start=9):
                if merged[offset] is None and name in previous_presence:
                    merged[offset] = previous_values[name]
                    observed.add(name)
            merged[-1] = json.dumps(sorted(observed))
            rows[index] = tuple(merged)
        conn.executemany(
            """INSERT OR REPLACE INTO sofascore_team_match_profiles
               (match_id, team_key, team_id, team_name, opponent_key,
                start_timestamp, is_home, goals_for, goals_against, xg_for,
                xg_against, shots_for, shots_against, shots_on_target_for,
                shots_on_target_against, big_chances_for, big_chances_against,
                box_shots_for, box_shots_against, possession, dominance,
                errors_to_goal, result_points, is_draw, captured_at, metric_presence_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()


def audit_match_postmortem(
    db_path: str,
    match_id: Any,
    event: dict[str, Any],
    predicted: str,
    prediction_status: str,
    force_reclassify: bool = False,
    postmatch_payload_fallback: (
        Callable[[str, list[str]], dict[str, Any] | None] | None
    ) = None,
) -> dict[str, Any]:
    """Coleta a autópsia final, classifica o RED e grava o peso recomendado."""
    init_sofascore_db(db_path)
    match_id = str(match_id)
    with closing(_connect(db_path)) as conn:
        existing = conn.execute(
            "SELECT verdict, learning_weight, data_coverage FROM match_postmortems WHERE match_id=?",
            (match_id,),
        ).fetchone()
    if existing and not force_reclassify:
        return {"verdict": existing[0], "learning_weight": existing[1],
                "data_coverage": existing[2], "cache_hit": True}

    statistics, stat_hit, _ = _fetch_resource(
        db_path, match_id, "statistics", ttl_seconds=365 * 86400
    )
    metrics, stat_coverage = _statistics_metrics(statistics)
    fallback_http = fallback_hits = fallback_quota_skips = 0
    if stat_coverage < 0.30 and postmatch_payload_fallback is not None:
        fallback = postmatch_payload_fallback(match_id, ["statistics"]) or {}
        fallback_http += int(fallback.get("http_requests", 0) or 0)
        fallback_hits += int(fallback.get("cache_hits", 0) or 0)
        fallback_quota_skips += int(fallback.get("quota_skips", 0) or 0)
        alternative_metrics, alternative_coverage = _statistics_metrics(
            (fallback.get("resources") or {}).get("statistics")
        )
        if alternative_coverage > stat_coverage:
            metrics.update(alternative_metrics)
            stat_coverage = alternative_coverage
    is_red = str(prediction_status).startswith("RED")
    extra_hits, extra_requests = 0, 0
    if is_red:
        shotmap, hit, _ = _fetch_resource(db_path, match_id, "shotmap", 365 * 86400)
        extra_hits += int(hit); extra_requests += int(not hit)
        graph, hit, _ = _fetch_resource(db_path, match_id, "graph", 365 * 86400)
        extra_hits += int(hit); extra_requests += int(not hit)
        lineups, hit, _ = _fetch_resource(db_path, match_id, "lineups", 365 * 86400)
        extra_hits += int(hit); extra_requests += int(not hit)
        average_positions, hit, _ = _fetch_resource(
            db_path, match_id, "average-positions", 365 * 86400
        )
        extra_hits += int(hit); extra_requests += int(not hit)
        incidents, hit, _ = _fetch_resource(
            db_path, match_id, "incidents", 365 * 86400
        )
        extra_hits += int(hit); extra_requests += int(not hit)
        metrics.update(_shotmap_metrics(shotmap))
        metrics.update(_graph_metrics(graph))
        metrics.update(_lineup_metrics(lineups))
        metrics.update(_average_position_metrics(average_positions))
        metrics.update(_incident_metrics(incidents))
        if stat_coverage < 0.30 and postmatch_payload_fallback is not None:
            fallback = postmatch_payload_fallback(
                match_id, ["shotmap", "graph", "incidents"]
            ) or {}
            fallback_http += int(fallback.get("http_requests", 0) or 0)
            fallback_hits += int(fallback.get("cache_hits", 0) or 0)
            fallback_quota_skips += int(fallback.get("quota_skips", 0) or 0)
            resources = fallback.get("resources") or {}
            metrics.update(_shotmap_metrics(resources.get("shotmap")))
            metrics.update(_graph_metrics(resources.get("graph")))
            metrics.update(_incident_metrics(resources.get("incidents")))

    actual_side, scoreline, _, _ = _outcomes(event)
    with closing(_connect(db_path)) as conn:
        snapshot_row = conn.execute(
            "SELECT predicted_outcome, analysis_json FROM ml_prediction_snapshots WHERE match_id=?",
            (match_id,),
        ).fetchone()
    predicted_side = _prediction_side(predicted, event)
    snapshot_side = str(snapshot_row[0] or "").upper() if snapshot_row else ""
    # O snapshot registra o lado escolhido antes do jogo e não depende das
    # abreviações divergentes usadas pelos diferentes provedores.
    if snapshot_side in {"MANDANTE", "EMPATE", "VISITANTE"}:
        predicted_side = snapshot_side
    try:
        snapshot_analysis = json.loads(snapshot_row[1]) if snapshot_row and snapshot_row[1] else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        snapshot_analysis = {}
    diagnosis = classify_postmortem(
        predicted_side, actual_side, prediction_status, metrics,
        stat_coverage, snapshot_analysis,
    )
    home_dom, _, _ = _dominance(metrics)
    if stat_coverage > 0:
        _save_team_profiles(db_path, match_id, event, metrics, home_dom)
    now = int(time.time())
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO match_postmortems
               (match_id, audited_at, prediction_status, predicted_outcome,
                actual_outcome, scoreline, verdict, process_score,
                chosen_dominance, opponent_dominance, learning_weight,
                error_margin, data_coverage, metrics_json, reasons_json,
                source_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                match_id, now, prediction_status, predicted_side, actual_side,
                scoreline, diagnosis["verdict"], diagnosis["process_score"],
                diagnosis["chosen_dominance"], diagnosis["opponent_dominance"],
                diagnosis["learning_weight"], diagnosis["error_margin"],
                diagnosis["data_coverage"], json.dumps(metrics, ensure_ascii=False),
                json.dumps(diagnosis["reasons"], ensure_ascii=False), ANALYSIS_VERSION,
            ),
        )
        # Um peso é aplicado uma única vez. A margem fica explicativa; o treino
        # não deve multiplicá-la novamente e supervalorizar todos os REDs.
        conn.execute(
            """INSERT INTO training_weights
               (match_id, peso, data_ultima_atualizacao, error_margin)
               VALUES (?,?,datetime('now'),?)
               ON CONFLICT(match_id) DO UPDATE SET peso=excluded.peso,
                   error_margin=excluded.error_margin,
                   data_ultima_atualizacao=excluded.data_ultima_atualizacao""",
            (match_id, diagnosis["learning_weight"], diagnosis["error_margin"]),
        )
        try:
            conn.execute(
                """UPDATE previsoes SET postmortem_verdict=?,
                          postmortem_process_score=?, postmortem_coverage=?
                   WHERE match_id=?""",
                (
                    diagnosis["verdict"], diagnosis["process_score"],
                    diagnosis["data_coverage"], match_id,
                ),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
    diagnosis.update({
        "actual_outcome": actual_side,
        "predicted_outcome": predicted_side,
        "scoreline": scoreline,
        "http_requests": int(not stat_hit) + extra_requests + fallback_http,
        "cache_hits": int(stat_hit) + extra_hits + fallback_hits,
        "allsports_fallback_http": fallback_http,
        "allsports_fallback_cache_hits": fallback_hits,
        "allsports_fallback_quota_skips": fallback_quota_skips,
        "cache_hit": False,
    })
    return diagnosis


def get_postmortem_summary(db_path: str, match_ids: Iterable[Any]) -> dict[str, int]:
    ids = [str(item) for item in match_ids if str(item)]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT verdict, COUNT(*) FROM match_postmortems WHERE match_id IN ({placeholders}) GROUP BY verdict",
            ids,
        ).fetchall()
    return {str(verdict): int(total) for verdict, total in rows}


def monitor_pregame_context_sources(db_path: str, limit: int = 1000) -> dict[str, dict[str, Any]]:
    """Mede cobertura e acerto das fontes usando somente snapshots pré-jogo.

    A métrica não decide promoção de modelo; ela mostra se uma nova família de
    variáveis já acumulou resultados suficientes para entrar numa validação.
    """
    init_sofascore_db(db_path)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            """SELECT s.predicted_outcome, s.features_json, m.actual_outcome
               FROM ml_prediction_snapshots s
               LEFT JOIN match_postmortems m ON m.match_id=s.match_id
               ORDER BY s.captured_at DESC LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()

    definitions = {
        "sofascore_pre": lambda f: any(
            key.startswith("sofa_pre_") and abs(float(value or 0)) > 1e-9
            for key, value in f.items()
        ),
        "sofascore_recent_form": lambda f: any(
            key.startswith("sofa_roll_") and abs(float(value or 0)) > 1e-9
            for key, value in f.items()
        ),
        "live_recent_form": lambda f: any(
            key.startswith("live_recent_") and abs(float(value or 0)) > 1e-9
            for key, value in f.items()
        ),
        "allsports_goal_distribution": lambda f: (
            float(f.get("allsports_goal_home_available", 0) or 0) > 0
            and float(f.get("allsports_goal_away_available", 0) or 0) > 0
        ),
    }
    counters = {
        source: {"snapshots": 0, "resolved": 0, "correct": 0}
        for source in definitions
    }
    for predicted, raw_features, actual in rows:
        try:
            features = json.loads(raw_features or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            features = {}
        for source, covered in definitions.items():
            try:
                is_covered = bool(covered(features))
            except (TypeError, ValueError):
                is_covered = False
            if not is_covered:
                continue
            counters[source]["snapshots"] += 1
            if str(actual or "").upper() in {"MANDANTE", "EMPATE", "VISITANTE"}:
                counters[source]["resolved"] += 1
                counters[source]["correct"] += int(
                    str(predicted or "").upper() == str(actual).upper()
                )

    now = int(time.time())
    output: dict[str, dict[str, Any]] = {}
    with closing(_connect(db_path)) as conn:
        for source, values in counters.items():
            resolved = values["resolved"]
            accuracy = values["correct"] / resolved if resolved else None
            details = {
                "window": min(len(rows), max(1, int(limit))),
                "ready_for_comparison": resolved >= 100,
            }
            conn.execute(
                """INSERT INTO ml_context_source_monitor
                   (evaluated_at, source, snapshots, resolved, correct,
                    accuracy, details_json) VALUES (?,?,?,?,?,?,?)""",
                (now, source, values["snapshots"], resolved, values["correct"],
                 accuracy, json.dumps(details, ensure_ascii=False)),
            )
            output[source] = {**values, "accuracy": accuracy, **details}
        conn.commit()
    return output
