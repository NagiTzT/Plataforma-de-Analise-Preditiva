"""Rotas e coleta compatíveis com a AllSportsApi2 v2.0.

A API descontinuou o feed plano de partidas por data. Para futebol, a coleta
atual é feita em duas etapas: torneios agendados no dia e eventos de cada
torneio. Este módulo mantém essa lógica em um único lugar para que ``app.py`` e
``robo_auto.py`` não voltem a divergir quando a API mudar.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable


ALLSPORTS_API_VERSION = "v2.0"
DEFAULT_ODDS_PROVIDER_ID = max(1, int(os.getenv("RAPIDAPI_ODDS_PROVIDER_ID", "1")))
SCHEDULE_CACHE_TTL_SECONDS = max(
    0, int(os.getenv("RAPIDAPI_SCHEDULE_CACHE_TTL_SECONDS", "1800"))
)
SCHEDULE_MAX_PAGES = max(1, int(os.getenv("RAPIDAPI_SCHEDULE_MAX_PAGES", "50")))
SCHEDULE_MAX_WORKERS = min(4, max(1, int(os.getenv("RAPIDAPI_SCHEDULE_MAX_WORKERS", "4"))))
# Zero significa "todos os torneios elegíveis encontrados nas páginas do dia".
# Um valor positivo continua disponível como freio operacional opcional.
SCHEDULE_MAX_TOURNAMENTS = max(
    0, int(os.getenv("RAPIDAPI_MAX_TOURNAMENTS_PER_DAY", "0"))
)
SCHEDULE_TARGET_EVENTS = max(
    0, int(os.getenv("RAPIDAPI_TARGET_EVENTS_PER_DAY", "0"))
)
FALLBACK_MAX_TOURNAMENTS = max(
    0, int(os.getenv("RAPIDAPI_FALLBACK_MAX_TOURNAMENTS", "0"))
)
PERSISTENT_CACHE_TTL_SECONDS = max(
    0, int(os.getenv("RAPIDAPI_PERSISTENT_CACHE_TTL_SECONDS", "129600"))
)
PERSISTENT_HISTORICAL_CACHE_TTL_SECONDS = max(
    PERSISTENT_CACHE_TTL_SECONDS,
    int(os.getenv("RAPIDAPI_HISTORICAL_CACHE_TTL_SECONDS", "2592000")),
)
SCHEDULE_CACHE_DB_PATH = os.getenv(
    "RAPIDAPI_SCHEDULE_CACHE_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "allsports_schedule_cache.db"),
)

_LOW_VALUE_TOURNAMENT_TERMS = {
    "u12", "u13", "u14", "u15", "u16", "u17", "u18", "u19", "u20",
    "u21", "u22", "u23", "u24", "sub-", "sub12", "sub13", "sub14",
    "sub15", "sub16", "sub17", "sub18", "sub19", "sub20", "sub21",
    "sub22", "sub23", "sub24",
    "amateur", "amador", "amadores", "youth", "junior", "juniors",
    "aspirantes", "reserve", "reserves", "woman", "women", "feminino",
    "femenino", "femmes", "frauen", "ladies", "girls",
}

_schedule_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_schedule_cache_lock = threading.RLock()
_schedule_date_locks: dict[str, threading.Lock] = {}
_persistent_cache_lock = threading.RLock()
BRT_TIMEZONE = timezone(timedelta(hours=-3))
GOAL_DISTRIBUTION_MAX_NEW_PER_DAY = max(
    0, int(os.getenv("RAPIDAPI_GOAL_DISTRIBUTION_MAX_NEW_PER_DAY", "300"))
)
_goal_distribution_budget_lock = threading.Lock()
_goal_distribution_budget_day = ""
_goal_distribution_budget_used = 0
POSTMATCH_FALLBACK_MAX_NEW_PER_DAY = max(
    0, int(os.getenv("RAPIDAPI_POSTMATCH_FALLBACK_MAX_NEW_PER_DAY", "190"))
)
_postmatch_budget_lock = threading.Lock()
_postmatch_budget_day = ""
_postmatch_budget_used = 0


def radar_window_brt(now: datetime | None = None, cutoff_hour: int = 22) -> tuple[datetime, datetime]:
    """Retorna a próxima janela operacional de 24h, ancorada em 22h BRT."""
    current = now or datetime.now(BRT_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=BRT_TIMEZONE)
    else:
        current = current.astimezone(BRT_TIMEZONE)
    start = current.replace(hour=cutoff_hour, minute=0, second=0, microsecond=0)
    if current >= start:
        start += timedelta(days=1)
    return start, start + timedelta(hours=24)


def schedule_dates_for_brt_window(start: datetime, end: datetime) -> list[date]:
    """Datas civis BRT que intersectam a janela semiaberta ``[start, end)``."""
    start_brt = start.replace(tzinfo=BRT_TIMEZONE) if start.tzinfo is None else start.astimezone(BRT_TIMEZONE)
    end_brt = end.replace(tzinfo=BRT_TIMEZONE) if end.tzinfo is None else end.astimezone(BRT_TIMEZONE)
    if end_brt <= start_brt:
        raise ValueError("A janela do radar precisa terminar depois de começar")
    first = start_brt.date()
    last = (end_brt - timedelta(microseconds=1)).date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _persistent_cache_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(SCHEDULE_CACHE_DB_PATH, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("""CREATE TABLE IF NOT EXISTS schedule_cache (
        host TEXT NOT NULL,
        schedule_date TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        fetched_at REAL NOT NULL,
        PRIMARY KEY (host, schedule_date)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS competition_fallback_cache (
        host TEXT NOT NULL,
        schedule_date TEXT NOT NULL,
        target_key TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        fetched_at REAL NOT NULL,
        PRIMARY KEY (host, schedule_date, target_key)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS pregame_resource_cache (
        host TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        fetched_at REAL NOT NULL,
        PRIMARY KEY (host, cache_key)
    )""")
    connection.commit()
    return connection


def _cache_ttl_for(day: date) -> int:
    if day < datetime.now(BRT_TIMEZONE).date():
        return PERSISTENT_HISTORICAL_CACHE_TTL_SECONDS
    return PERSISTENT_CACHE_TTL_SECONDS


def _cache_covers_requested_scope(result: dict[str, Any]) -> bool:
    """Evita reutilizar um cache antigo truncado por um limite menor."""
    meta = result.get("_meta", {}) if isinstance(result, dict) else {}
    try:
        available = int(meta.get("tournaments_available", 0) or 0)
        consulted = int(meta.get("tournaments_consulted", 0) or 0)
    except (TypeError, ValueError):
        return False
    if available <= 0:
        return bool(meta.get("complete", False))
    required = available if SCHEDULE_MAX_TOURNAMENTS == 0 else min(available, SCHEDULE_MAX_TOURNAMENTS)
    return consulted >= required


def _cache_hit_result(result: dict[str, Any], cache_type: str) -> dict[str, Any]:
    cached_result = {"events": result.get("events", [])}
    meta = dict(result.get("_meta", {}))
    original_requests = int(meta.get("original_estimated_http_requests", meta.get("estimated_http_requests", 0)) or 0)
    meta.update({
        "cache_hit": cache_type,
        "original_estimated_http_requests": original_requests,
        "estimated_http_requests": 0,
    })
    cached_result["_meta"] = meta
    return cached_result


def _load_persistent_schedule(host: str, day: date) -> dict[str, Any] | None:
    if PERSISTENT_CACHE_TTL_SECONDS <= 0:
        return None
    with _persistent_cache_lock:
        connection = _persistent_cache_connection()
        try:
            row = connection.execute(
                "SELECT payload_json, fetched_at FROM schedule_cache WHERE host=? AND schedule_date=?",
                (host, day.isoformat()),
            ).fetchone()
        finally:
            connection.close()
    if not row or time.time() - float(row[1]) > _cache_ttl_for(day):
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not _cache_covers_requested_scope(payload):
        return None
    return _cache_hit_result(payload, "persistent")


def _save_persistent_schedule(host: str, day: date, result: dict[str, Any]) -> None:
    if PERSISTENT_CACHE_TTL_SECONDS <= 0:
        return
    payload_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    with _persistent_cache_lock:
        connection = _persistent_cache_connection()
        try:
            connection.execute(
                """INSERT INTO schedule_cache (host, schedule_date, payload_json, fetched_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(host, schedule_date) DO UPDATE SET
                       payload_json=excluded.payload_json,
                       fetched_at=excluded.fetched_at""",
                (host, day.isoformat(), payload_json, time.time()),
            )
            connection.commit()
        finally:
            connection.close()


def _base_url(host: str) -> str:
    clean_host = str(host or "").strip().strip("/")
    if not clean_host:
        raise ValueError("RAPIDAPI_HOST não pode ser vazio")
    return f"https://{clean_host}/api"


def _part(value: Any, name: str) -> str:
    text = str(value or "").strip().strip("/")
    if not text or any(char in text for char in ("/", "?", "#")):
        raise ValueError(f"{name} inválido: {value!r}")
    return text


def parse_api_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Data inválida para a AllSportsApi2: {value!r}")


def matches_date_url(host: str, value: Any) -> str:
    """Feed legado por data; mantido apenas como referência/fallback."""
    day = parse_api_date(value)
    return f"{_base_url(host)}/matches/{day.day}/{day.month}/{day.year}"


def matches_odds_date_url(host: str, value: Any) -> str:
    day = parse_api_date(value)
    return f"{_base_url(host)}/matches/odds/{day.day}/{day.month}/{day.year}"


def scheduled_tournaments_url(host: str, value: Any, page: int = 1) -> str:
    day = parse_api_date(value)
    return f"{_base_url(host)}/scheduled-tournaments/{day.day}/{day.month}/{day.year}/page/{max(1, int(page))}"


def tournament_scheduled_events_url(host: str, tournament_id: Any, value: Any) -> str:
    day = parse_api_date(value)
    return (
        f"{_base_url(host)}/tournament/{_part(tournament_id, 'tournament_id')}"
        f"/scheduled-events/{day.isoformat()}"
    )


def match_detail_url(host: str, match_id: Any) -> str:
    # A rota segue ativa para futebol, embora o OpenAPI v2.0 só exponha seus
    # sub-recursos compactados em /api/match/{id}/{segmento}.
    return f"{_base_url(host)}/match/{_part(match_id, 'match_id')}"


def match_resource_url(host: str, match_id: Any, resource: str) -> str:
    allowed = {
        "average-positions", "best-players", "commentary", "duel", "form",
        "graph", "highlights", "incidents", "lineups", "managers", "meta",
        "official-tweets", "shotmap", "statistics", "streaks", "votes",
        "weather", "win-probability",
    }
    clean_resource = _part(resource, "resource")
    if clean_resource not in allowed:
        raise ValueError(f"Recurso de partida não documentado na v2.0: {clean_resource}")
    return f"{match_detail_url(host, match_id)}/{clean_resource}"


def goal_distribution_url(host: str, team_id: Any, tournament_id: Any, season_id: Any) -> str:
    return (
        f"{_base_url(host)}/team/{_part(team_id, 'team_id')}"
        f"/tournament/{_part(tournament_id, 'tournament_id')}"
        f"/season/{_part(season_id, 'season_id')}/goal-distributions"
    )


def _reserve_goal_distribution_request() -> bool:
    global _goal_distribution_budget_day, _goal_distribution_budget_used
    today = datetime.now(BRT_TIMEZONE).date().isoformat()
    with _goal_distribution_budget_lock:
        if _goal_distribution_budget_day != today:
            _goal_distribution_budget_day = today
            _goal_distribution_budget_used = 0
        if (GOAL_DISTRIBUTION_MAX_NEW_PER_DAY > 0
                and _goal_distribution_budget_used >= GOAL_DISTRIBUTION_MAX_NEW_PER_DAY):
            return False
        _goal_distribution_budget_used += 1
        return True


def _reserve_postmatch_request() -> bool:
    global _postmatch_budget_day, _postmatch_budget_used
    today = datetime.now(BRT_TIMEZONE).date().isoformat()
    with _postmatch_budget_lock:
        if _postmatch_budget_day != today:
            _postmatch_budget_day = today
            _postmatch_budget_used = 0
        if (POSTMATCH_FALLBACK_MAX_NEW_PER_DAY > 0
                and _postmatch_budget_used >= POSTMATCH_FALLBACK_MAX_NEW_PER_DAY):
            return False
        _postmatch_budget_used += 1
        return True


def _cached_resource(host: str, cache_key: str, ttl_seconds: int, url: str,
                     safe_get: Callable[[str], Any],
                     request_guard: Callable[[], bool] | None = None
                     ) -> tuple[dict[str, Any] | None, bool, int]:
    now = time.time()
    with _persistent_cache_lock:
        with closing(_persistent_cache_connection()) as conn:
            row = conn.execute(
                "SELECT payload_json,fetched_at FROM pregame_resource_cache WHERE host=? AND cache_key=?",
                (host, cache_key),
            ).fetchone()
    if row and now - float(row[1] or 0) < max(0, int(ttl_seconds)):
        try:
            payload = json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        return payload if isinstance(payload, dict) else None, True, 0
    if request_guard is not None and not request_guard():
        return None, False, 0
    payload = safe_get(url)
    if isinstance(payload, dict):
        with _persistent_cache_lock:
            with closing(_persistent_cache_connection()) as conn:
                conn.execute(
                    """INSERT INTO pregame_resource_cache(host,cache_key,payload_json,fetched_at)
                       VALUES(?,?,?,?) ON CONFLICT(host,cache_key) DO UPDATE SET
                       payload_json=excluded.payload_json,fetched_at=excluded.fetched_at""",
                    (host, cache_key, json.dumps(payload, ensure_ascii=False), now),
                )
                conn.commit()
    return payload if isinstance(payload, dict) else None, False, 1


def _goal_distribution_features(payload: dict[str, Any] | None, prefix: str,
                                venue: str) -> dict[str, float]:
    rows = (payload or {}).get("goalDistributions") or []
    selected = next((row for row in rows if str(row.get("type")) == venue), None)
    selected = selected or next((row for row in rows if str(row.get("type")) == "overall"), None)
    if not isinstance(selected, dict):
        return {f"allsports_goal_{prefix}_available": 0.0}
    games = max(0.0, float(selected.get("matches") or 0))
    scored = max(0.0, float(selected.get("scoredGoals") or 0))
    conceded = max(0.0, float(selected.get("concededGoals") or 0))
    periods = selected.get("periods") or []
    early_scored = sum(float(row.get("scoredGoals") or 0) for row in periods
                       if float(row.get("periodEnd") or 0) <= 45)
    early_conceded = sum(float(row.get("concededGoals") or 0) for row in periods
                         if float(row.get("periodEnd") or 0) <= 45)
    return {
        f"allsports_goal_{prefix}_available": float(games > 0),
        f"allsports_goal_{prefix}_games": games,
        f"allsports_goal_{prefix}_gf_avg": scored / games if games else 0.0,
        f"allsports_goal_{prefix}_ga_avg": conceded / games if games else 0.0,
        f"allsports_goal_{prefix}_goal_balance": (scored - conceded) / games if games else 0.0,
        f"allsports_goal_{prefix}_first_half_scored_share": early_scored / max(1.0, scored),
        f"allsports_goal_{prefix}_first_half_conceded_share": early_conceded / max(1.0, conceded),
    }


def fetch_allsports_pregame_context(host: str, game: dict[str, Any],
                                    safe_get: Callable[[str], Any],
                                    need_form: bool = True,
                                    need_streaks: bool = True,
                                    include_goal_distributions: bool = True) -> dict[str, Any]:
    """Busca somente recursos válidos antes do início; nunca consulta dados live/pós-jogo."""
    match_id = str(game.get("ID") or game.get("match_id") or "").strip()
    result: dict[str, Any] = {"http_requests": 0, "cache_hits": 0, "features": {}}
    if not match_id:
        return result
    try:
        start_timestamp = int(game.get("Timestamp") or game.get("start_timestamp") or 0)
    except (TypeError, ValueError):
        start_timestamp = 0
    if start_timestamp and start_timestamp <= int(time.time()):
        result["skipped_after_start"] = 1
        return result
    if need_form:
        result["form"] = safe_get(match_resource_url(host, match_id, "form"))
        result["http_requests"] += 1
    if need_streaks:
        result["streaks"] = safe_get(match_resource_url(host, match_id, "streaks"))
        result["http_requests"] += 1
    if not include_goal_distributions:
        return result
    tournament_id = str(game.get("Unique_Tournament_ID") or "").strip()
    season_id = str(game.get("Season_ID") or "").strip()
    resolved_team_ids = {
        "Home_ID": str(game.get("Home_ID") or "").strip(),
        "Away_ID": str(game.get("Away_ID") or "").strip(),
    }
    # Lotes diarios de odds frequentemente trazem o ID da partida, mas nao os
    # IDs unicos do torneio/temporada. Completa-os uma vez pelo detalhe do jogo
    # e reutiliza o snapshot por 24 h. O recurso e estritamente pre-jogo.
    if not (tournament_id and season_id and all(resolved_team_ids.values())):
        detail, hit, requests = _cached_resource(
            host, f"match-detail:{match_id}", 86400,
            match_detail_url(host, match_id), safe_get,
        )
        result["cache_hits"] += int(hit)
        result["http_requests"] += requests
        event = (detail or {}).get("event") or {}
        tournament = event.get("tournament") or {}
        unique_tournament = tournament.get("uniqueTournament") or {}
        tournament_id = tournament_id or str(unique_tournament.get("id") or "").strip()
        season_id = season_id or str(((event.get("season") or {}).get("id")) or "").strip()
        resolved_team_ids["Home_ID"] = (
            resolved_team_ids["Home_ID"]
            or str(((event.get("homeTeam") or {}).get("id")) or "").strip()
        )
        resolved_team_ids["Away_ID"] = (
            resolved_team_ids["Away_ID"]
            or str(((event.get("awayTeam") or {}).get("id")) or "").strip()
        )
        result["features"].update({
            "allsports_match_context_available": float(bool(event)),
            "allsports_competition_type": float(event.get("competitionType") or 0),
            "allsports_round": float(((event.get("roundInfo") or {}).get("round")) or 0),
            "allsports_home_popularity": float(((event.get("homeTeam") or {}).get("userCount")) or 0),
            "allsports_away_popularity": float(((event.get("awayTeam") or {}).get("userCount")) or 0),
        })
        # The detail URL was resolved by the exact match ID, and the phase is
        # already in this response. Never spend another request for this field.
        if str(event.get("id") or "") == match_id:
            from competition_context import event_competition_flags
            result["features"].update(event_competition_flags(
                event, game.get("Liga", ""), int(time.time()), start_timestamp))
        # O enriquecimento fica disponivel para os demais callbacks da mesma
        # coleta, sem trocar IDs entre provedores.
        if tournament_id:
            game["Unique_Tournament_ID"] = tournament_id
        if season_id:
            game["Season_ID"] = season_id
        for key, value in resolved_team_ids.items():
            if value:
                game[key] = value
    for side, venue, id_key in (("home", "home", "Home_ID"), ("away", "away", "Away_ID")):
        team_id = resolved_team_ids[id_key]
        if not (team_id and tournament_id and season_id):
            result["features"].update({f"allsports_goal_{side}_available": 0.0})
            continue
        key = f"goal:{team_id}:{tournament_id}:{season_id}"
        payload, hit, requests = _cached_resource(
            host, key, 7 * 86400,
            goal_distribution_url(host, team_id, tournament_id, season_id), safe_get,
            request_guard=_reserve_goal_distribution_request,
        )
        result["cache_hits"] += int(hit)
        result["http_requests"] += requests
        result["features"].update(_goal_distribution_features(payload, side, venue))
    features = result["features"]
    home_gf = float(features.get("allsports_goal_home_gf_avg", 0.0))
    home_ga = float(features.get("allsports_goal_home_ga_avg", 0.0))
    away_gf = float(features.get("allsports_goal_away_gf_avg", 0.0))
    away_ga = float(features.get("allsports_goal_away_ga_avg", 0.0))
    features.update({
        "allsports_goal_attack_matchup_diff": (home_gf + away_ga) - (away_gf + home_ga),
        "allsports_goal_expected_total_proxy": (home_gf + home_ga + away_gf + away_ga) / 2.0,
    })
    return result


def fetch_allsports_postmatch_resources(
    host: str,
    match_id: Any,
    resources: Iterable[str],
    safe_get: Callable[[str], Any],
) -> dict[str, Any]:
    """Fallback cacheado para autopsia quando o SofaScore direto nao cobre."""
    identifier = str(match_id or "").strip()
    result: dict[str, Any] = {
        "resources": {}, "http_requests": 0, "cache_hits": 0, "quota_skips": 0,
    }
    if not identifier:
        return result
    allowed = {"statistics", "shotmap", "graph", "incidents"}
    for resource in resources:
        if resource not in allowed:
            continue
        payload, hit, requests = _cached_resource(
            host, f"postmatch:{identifier}:{resource}", 365 * 86400,
            match_resource_url(host, identifier, resource), safe_get,
            request_guard=_reserve_postmatch_request,
        )
        result["cache_hits"] += int(hit)
        result["http_requests"] += requests
        result["quota_skips"] += int(payload is None and not hit and requests == 0)
        if isinstance(payload, dict):
            result["resources"][resource] = payload
    return result


def match_odds_url(host: str, match_id: Any, provider_id: int | None = None) -> str:
    provider = provider_id or DEFAULT_ODDS_PROVIDER_ID
    return f"{match_detail_url(host, match_id)}/odds/{max(1, int(provider))}/all"


def match_winning_odds_url(host: str, match_id: Any, provider_id: int | None = None) -> str:
    provider = provider_id or DEFAULT_ODDS_PROVIDER_ID
    return f"{match_detail_url(host, match_id)}/provider/{max(1, int(provider))}/winning-odds"


def team_matches_url(host: str, team_id: Any, direction: str, page: int = 0) -> str:
    clean_direction = _part(direction, "direction")
    if clean_direction not in {"previous", "next"}:
        raise ValueError("direction deve ser 'previous' ou 'next'")
    return (
        f"{_base_url(host)}/team/{_part(team_id, 'team_id')}/matches/"
        f"{clean_direction}/{max(0, int(page))}"
    )


def fetch_recent_team_events_for_game(
    host: str,
    game: dict[str, Any],
    side: str,
    safe_get: Callable[[str], Any],
    provider_team_id: Any = "",
) -> dict[str, Any] | None:
    """Carrega a primeira página de jogos anteriores de um time.

    ``provider_team_id`` normalmente vem do vínculo SofaScore. Se os IDs dos
    provedores não coincidirem, a função resolve o ID AllSports pelo detalhe da
    partida atual e tenta novamente. O chamador é responsável pelo cache e por
    limitar a validade temporal do retrato.
    """
    clean_side = str(side or "").lower()
    if clean_side not in {"home", "away"}:
        raise ValueError("side deve ser 'home' ou 'away'")
    id_key = "Home_ID" if clean_side == "home" else "Away_ID"
    candidates: list[str] = []
    local_id = str(game.get(id_key) or "").strip()
    if local_id:
        candidates.append(local_id)
    linked_id = str(provider_team_id or "").strip()
    if linked_id and linked_id not in candidates:
        candidates.append(linked_id)

    detail_loaded = False

    def resolve_from_current_match() -> str:
        nonlocal detail_loaded
        detail_loaded = True
        match_id = str(game.get("ID") or game.get("match_id") or "").strip()
        if not match_id:
            return ""
        payload = safe_get(match_detail_url(host, match_id))
        event = payload.get("event") if isinstance(payload, dict) else None
        if not isinstance(event, dict):
            return ""
        home = event.get("homeTeam") or {}
        away = event.get("awayTeam") or {}
        if isinstance(home, dict) and home.get("id"):
            game["Home_ID"] = str(home["id"])
        if isinstance(away, dict) and away.get("id"):
            game["Away_ID"] = str(away["id"])
        team = home if clean_side == "home" else away
        team_id = str(team.get("id") or "").strip() if isinstance(team, dict) else ""
        if team_id:
            game[id_key] = team_id
        return team_id

    if not candidates:
        resolved = resolve_from_current_match()
        if resolved:
            candidates.append(resolved)

    attempted: set[str] = set()
    for team_id in candidates:
        if not team_id or team_id in attempted:
            continue
        attempted.add(team_id)
        payload = safe_get(team_matches_url(host, team_id, "previous", 0))
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            return {"provider": "allsports", "team_id": team_id, "payload": payload}

    # Um ID Sofa e um ID AllSports podem divergir. Só pagamos o detalhe da
    # partida depois que a tentativa direta falha.
    if not detail_loaded:
        resolved = resolve_from_current_match()
        if resolved and resolved not in attempted:
            payload = safe_get(team_matches_url(host, resolved, "previous", 0))
            if isinstance(payload, dict) and isinstance(payload.get("events"), list):
                return {"provider": "allsports", "team_id": resolved, "payload": payload}
    return None


def tournament_seasons_url(host: str, tournament_id: Any) -> str:
    return f"{_base_url(host)}/tournament/{_part(tournament_id, 'tournament_id')}/seasons"


def tournament_standings_url(host: str, tournament_id: Any, season_id: Any) -> str:
    return (
        f"{_base_url(host)}/tournament/{_part(tournament_id, 'tournament_id')}"
        f"/season/{_part(season_id, 'season_id')}/standings/total"
    )


def normalize_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        events = payload.get("events", [])
    elif isinstance(payload, list):
        events = payload
    else:
        events = []
    return [event for event in events if isinstance(event, dict)]


def _competition_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def fetch_competition_events_for_date(
    host: str,
    value: Any,
    competition_names: list[str] | set[str] | tuple[str, ...],
    safe_get: Callable[[str], Any],
) -> dict[str, Any]:
    """Fallback seletivo para campeonatos sem ``fid`` na agenda principal.

    As páginas de torneios são lidas uma vez; somente torneios cujo país ou
    nome combina com os campeonatos solicitados têm seus eventos consultados.
    Isso evita voltar ao custo de uma requisição para todos os torneios do dia.
    """
    day = parse_api_date(value)
    targets = {_competition_text(name) for name in competition_names if str(name).strip()}
    target_key = "|".join(sorted(targets))
    if not targets:
        return {"events": [], "_meta": {"estimated_http_requests": 0}}
    with _persistent_cache_lock:
        connection = _persistent_cache_connection()
        try:
            cached = connection.execute(
                """SELECT payload_json,fetched_at FROM competition_fallback_cache
                   WHERE host=? AND schedule_date=? AND target_key=?""",
                (host, day.isoformat(), target_key),
            ).fetchone()
        finally:
            connection.close()
    if cached and time.time() - float(cached[1]) <= _cache_ttl_for(day):
        try:
            result = json.loads(cached[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            result = None
        if isinstance(result, dict):
            meta = dict(result.get("_meta") or {})
            meta["original_estimated_http_requests"] = int(
                meta.get("original_estimated_http_requests", meta.get("estimated_http_requests", 0)) or 0
            )
            meta["estimated_http_requests"] = 0
            meta["cache_hit"] = "persistent"
            result["_meta"] = meta
            return result
    target_countries = {target.split()[0] for target in targets if target.split()}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    listing_requests = 0
    for page in range(1, SCHEDULE_MAX_PAGES + 1):
        payload = safe_get(scheduled_tournaments_url(host, day, page))
        listing_requests += 1
        page_candidates = _scheduled_tournament_candidates(payload)
        if not page_candidates:
            break
        for candidate in page_candidates:
            candidates_by_id[candidate["id"]] = candidate

    generic = {"cup", "fa", "league", "division", "championship", "football"}

    def semantic_match(candidate: dict[str, Any], target: str) -> bool:
        name = _competition_text(candidate.get("name"))
        name_tokens = set(name.split()) - generic
        target_tokens = set(target.split())
        country = target.split()[0] if target.split() else ""
        descriptive = target_tokens - generic - {country}
        if name and (name in target or target in name):
            return True
        if descriptive and name_tokens and len(descriptive & name_tokens) >= min(2, len(descriptive)):
            return True
        return False

    selected_by_id: dict[str, dict[str, Any]] = {}
    unresolved = set(targets)
    for target in targets:
        matches = [
            candidate for candidate in candidates_by_id.values()
            if semantic_match(candidate, target)
        ]
        if matches:
            unresolved.discard(target)
            for candidate in matches:
                selected_by_id[candidate["id"]] = candidate
    # Traduções sem tokens em comum são resolvidas pelo país; depois a ligação
    # por horário + dois nomes impede associar o jogo ao torneio errado.
    for target in unresolved:
        country = target.split()[0] if target.split() else ""
        if not country:
            continue
        for candidate in candidates_by_id.values():
            category_tokens = set(_competition_text(candidate.get("category")).split())
            if country in category_tokens:
                selected_by_id[candidate["id"]] = candidate
    selected = sorted(
        selected_by_id.values(),
        key=lambda item: (-item.get("priority", 0), item.get("name", "")),
    )
    if FALLBACK_MAX_TOURNAMENTS:
        selected = selected[:FALLBACK_MAX_TOURNAMENTS]
    events: dict[str, dict[str, Any]] = {}
    tournament_requests = 0
    failures = 0
    for candidate in selected:
        payload = safe_get(tournament_scheduled_events_url(host, candidate["id"], day))
        tournament_requests += 1
        if payload is None:
            failures += 1
            continue
        for event in normalize_events(payload):
            event_id = str(event.get("id") or "")
            if event_id:
                events[event_id] = event
    result = {
        "events": list(events.values()),
        "_meta": {
            "date": day.isoformat(),
            "requested_competitions": sorted(targets),
            "unresolved_by_name": sorted(unresolved),
            "target_countries": sorted(target_countries),
            "tournaments_available": len(candidates_by_id),
            "tournaments_selected": len(selected),
            "tournament_limit": FALLBACK_MAX_TOURNAMENTS,
            "listing_requests": listing_requests,
            "tournament_requests": tournament_requests,
            "estimated_http_requests": listing_requests + tournament_requests,
            "tournament_failures": failures,
        },
    }
    with _persistent_cache_lock:
        connection = _persistent_cache_connection()
        try:
            connection.execute(
                """INSERT INTO competition_fallback_cache
                   (host,schedule_date,target_key,payload_json,fetched_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(host,schedule_date,target_key) DO UPDATE SET
                       payload_json=excluded.payload_json,
                       fetched_at=excluded.fetched_at""",
                (host, day.isoformat(), target_key,
                 json.dumps(result, ensure_ascii=False, separators=(",", ":")), time.time()),
            )
            connection.commit()
        finally:
            connection.close()
    return result


def _scheduled_tournament_candidates(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    for scheduled in payload.get("scheduled", []):
        if not isinstance(scheduled, dict):
            continue
        wrapper = scheduled.get("tournament", {})
        if not isinstance(wrapper, dict):
            continue
        # O ID aceito por /api/tournament/{id}/scheduled-events é o torneio
        # interno. O ID do wrapper identifica apenas o agrupamento agendado.
        tournament = wrapper.get("tournament")
        if not isinstance(tournament, dict):
            tournament = wrapper
        tournament_id = str(tournament.get("id") or "").strip()
        if not tournament_id:
            continue
        name = str(wrapper.get("name") or tournament.get("name") or "")
        category = str(
            wrapper.get("category", {}).get("name")
            if isinstance(wrapper.get("category"), dict)
            else ""
        )
        if not category and isinstance(tournament.get("category"), dict):
            category = str(tournament["category"].get("name") or "")
        normalized_name = f"{name} {category}".lower()
        if any(term in normalized_name for term in _LOW_VALUE_TOURNAMENT_TERMS):
            continue
        counts = scheduled.get("timezoneEventCount", {})
        event_count = 1
        if isinstance(counts, dict) and counts:
            numeric_counts = []
            for count in counts.values():
                try:
                    numeric_counts.append(max(0, int(count)))
                except (TypeError, ValueError):
                    continue
            if numeric_counts:
                event_count = max(1, max(numeric_counts))
        try:
            priority = int(wrapper.get("priority") or tournament.get("priority") or 0)
        except (TypeError, ValueError):
            priority = 0
        result.append({
            "id": tournament_id,
            "name": name,
            "priority": priority,
            "event_count": event_count,
        })
    return result


def _date_lock(cache_key: str) -> threading.Lock:
    with _schedule_cache_lock:
        return _schedule_date_locks.setdefault(cache_key, threading.Lock())


def fetch_football_events_for_date(
    safe_get: Callable[..., Any],
    host: str,
    value: Any,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Busca os jogos de futebol pelo fluxo atual documentado da API v2.0.

    O retorno mantém ``{"events": [...]}``, formato já consumido pelo projeto,
    e inclui ``_meta`` apenas para diagnóstico/contagem de requisições.
    """
    day = parse_api_date(value)
    cache_key = f"{host}:{day.isoformat()}"
    now = time.monotonic()
    if not force_refresh and SCHEDULE_CACHE_TTL_SECONDS > 0:
        with _schedule_cache_lock:
            cached = _schedule_cache.get(cache_key)
            if (cached and now - cached[0] <= SCHEDULE_CACHE_TTL_SECONDS
                    and _cache_covers_requested_scope(cached[1])):
                return _cache_hit_result(cached[1], "memory")

    if not force_refresh:
        persistent = _load_persistent_schedule(host, day)
        if persistent is not None:
            with _schedule_cache_lock:
                _schedule_cache[cache_key] = (time.monotonic(), persistent)
            return persistent

    with _date_lock(cache_key):
        now = time.monotonic()
        if not force_refresh and SCHEDULE_CACHE_TTL_SECONDS > 0:
            with _schedule_cache_lock:
                cached = _schedule_cache.get(cache_key)
                if (cached and now - cached[0] <= SCHEDULE_CACHE_TTL_SECONDS
                        and _cache_covers_requested_scope(cached[1])):
                    return _cache_hit_result(cached[1], "memory")
        if not force_refresh:
            persistent = _load_persistent_schedule(host, day)
            if persistent is not None:
                with _schedule_cache_lock:
                    _schedule_cache[cache_key] = (time.monotonic(), persistent)
                return persistent

        candidates_by_id: dict[str, dict[str, Any]] = {}
        pages_consulted = 0
        page_failures = 0
        pagination_truncated = False
        for page in range(1, SCHEDULE_MAX_PAGES + 1):
            payload = safe_get(scheduled_tournaments_url(host, day, page))
            pages_consulted += 1
            if not isinstance(payload, dict):
                page_failures += 1
                break
            for candidate in _scheduled_tournament_candidates(payload):
                tournament_id = candidate["id"]
                current = candidates_by_id.get(tournament_id)
                if current is None:
                    candidates_by_id[tournament_id] = candidate
                else:
                    current["priority"] = max(current["priority"], candidate["priority"])
                    current["event_count"] = max(current["event_count"], candidate["event_count"])
            has_next_page = bool(payload.get("hasNextPage"))
            if not has_next_page:
                break
            if page == SCHEDULE_MAX_PAGES:
                pagination_truncated = True

        candidates = sorted(
            candidates_by_id.values(),
            key=lambda item: (-item["priority"], -item["event_count"], item["name"]),
        )
        selected_candidates: list[dict[str, Any]] = []
        estimated_events = 0
        for candidate in candidates:
            if SCHEDULE_MAX_TOURNAMENTS and len(selected_candidates) >= SCHEDULE_MAX_TOURNAMENTS:
                break
            selected_candidates.append(candidate)
            estimated_events += candidate["event_count"]
            if SCHEDULE_TARGET_EVENTS and estimated_events >= SCHEDULE_TARGET_EVENTS:
                break
        tournament_ids = [candidate["id"] for candidate in selected_candidates]

        events: list[dict[str, Any]] = []
        seen_events: set[str] = set()

        def fetch_tournament(tournament_id: str) -> tuple[bool, list[dict[str, Any]]]:
            payload = safe_get(tournament_scheduled_events_url(host, tournament_id, day))
            return payload is not None, normalize_events(payload)

        tournament_failures = 0
        if tournament_ids:
            with ThreadPoolExecutor(max_workers=SCHEDULE_MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_tournament, tid): tid for tid in tournament_ids}
                for future in as_completed(futures):
                    try:
                        success, tournament_events = future.result()
                    except Exception:
                        success = False
                        tournament_events = []
                    if not success:
                        tournament_failures += 1
                    for event in tournament_events:
                        event_id = str(event.get("id") or "").strip()
                        dedupe_key = event_id or repr(event)
                        if dedupe_key not in seen_events:
                            seen_events.add(dedupe_key)
                            events.append(event)

        events.sort(key=lambda event: (int(event.get("startTimestamp") or 0), str(event.get("id") or "")))
        result = {
            "events": events,
            "_meta": {
                "source": "scheduled-tournaments-v2",
                "api_version": ALLSPORTS_API_VERSION,
                "date": day.isoformat(),
                "pages_consulted": pages_consulted,
                "tournaments_available": len(candidates),
                "tournaments_consulted": len(tournament_ids),
                "estimated_events_selected": estimated_events,
                "tournament_limit": SCHEDULE_MAX_TOURNAMENTS,
                "page_failures": page_failures,
                "pagination_truncated": pagination_truncated,
                "tournament_failures": tournament_failures,
                "tournaments_succeeded": len(tournament_ids) - tournament_failures,
                "estimated_http_requests": pages_consulted + len(tournament_ids),
            },
        }
        unconsulted_tournaments = max(0, len(candidates) - len(tournament_ids))
        missing_count_exact = page_failures == 0 and not pagination_truncated
        minimum_listing_retries = page_failures + int(pagination_truncated)
        result["_meta"].update({
            "unconsulted_tournaments": unconsulted_tournaments,
            "requests_missing_to_complete": (
                tournament_failures + unconsulted_tournaments + minimum_listing_retries
            ),
            "requests_missing_count_exact": missing_count_exact,
        })
        completion_rate = (
            (len(tournament_ids) - tournament_failures) / len(tournament_ids)
            if tournament_ids else 0.0
        )
        collection_complete = (
            bool(candidates)
            and page_failures == 0
            and not pagination_truncated
            and tournament_failures == 0
        )
        result["_meta"]["completion_rate"] = completion_rate
        result["_meta"]["complete"] = collection_complete
        if SCHEDULE_CACHE_TTL_SECONDS > 0:
            with _schedule_cache_lock:
                _schedule_cache[cache_key] = (time.monotonic(), result)
        # Evita perpetuar uma agenda parcial quando acabarem as chaves. Respostas
        # 204 são {}, portanto contam como consultas concluídas sem eventos.
        if collection_complete:
            _save_persistent_schedule(host, day, result)
        return result
