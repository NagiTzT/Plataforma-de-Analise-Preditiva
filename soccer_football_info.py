"""Integração contextual com a API Soccer Football Info.

Esta fonte é usada somente para contexto pré-jogo (forma e classificação). As
odds e o filtro de valor continuam pertencendo à AllSportsAPI2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

import requests
from api_key_config import configured_keys


SOCCER_API_HOST = os.getenv(
    "SOCCER_API_HOST", "soccer-football-info.p.rapidapi.com"
).strip()
SOCCER_API_DAILY_LIMIT = max(
    1, min(190, int(os.getenv("SOCCER_API_DAILY_LIMIT", "190")))
)
SOCCER_API_MIN_INTERVAL_SECONDS = max(
    0.26, float(os.getenv("SOCCER_API_MIN_INTERVAL_SECONDS", "0.29"))
)
SOCCER_FORM_TTL_SECONDS = max(
    3600, int(os.getenv("SOCCER_FORM_TTL_SECONDS", str(12 * 3600)))
)
SOCCER_PAGE_CACHE_TTL_SECONDS = max(
    900, int(os.getenv("SOCCER_PAGE_CACHE_TTL_SECONDS", str(6 * 3600)))
)
SOCCER_MAX_PAGES_PER_DAY = max(
    1, int(os.getenv("SOCCER_MAX_PAGES_PER_DAY", "160"))
)

def _configured_keys() -> list[str]:
    return configured_keys("SOCCER_API_KEYS", "soccer")


SOCCER_API_KEYS = _configured_keys()

_schema_lock = threading.RLock()
_initialized_databases: set[str] = set()
_quota_lock = threading.RLock()
_rate_lock = threading.Lock()
_last_request_at = 0.0
_rotation_cursor = 0
_league_mapping_cache: dict[tuple[str, str], tuple[str, str]] = {}
_league_mapping_lock = threading.RLock()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def _db(db_path: str):
    """Transação curta que sempre libera o arquivo SQLite no Windows."""
    conn = _connect(db_path)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def init_soccer_context_db(db_path: str) -> None:
    """Cria as tabelas da integração sem alterar as tabelas existentes."""
    absolute = os.path.abspath(db_path)
    with _schema_lock:
        if absolute in _initialized_databases:
            return
        with _db(absolute) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS soccer_api_key_usage (
                    key_id TEXT NOT NULL,
                    usage_date TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    server_remaining INTEGER,
                    reset_at TEXT,
                    last_http_status INTEGER,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (key_id, usage_date)
                );

                CREATE TABLE IF NOT EXISTS soccer_page_cache (
                    utc_date TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY (utc_date, page)
                );

                CREATE TABLE IF NOT EXISTS soccer_team_form_cache (
                    provider_team_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    perf_json TEXT,
                    position REAL,
                    raw_json TEXT NOT NULL,
                    captured_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_soccer_team_name
                    ON soccer_team_form_cache(normalized_name);

                CREATE TABLE IF NOT EXISTS soccer_team_aliases (
                    source_normalized TEXT NOT NULL,
                    league_normalized TEXT NOT NULL DEFAULT '',
                    source_name TEXT NOT NULL,
                    provider_team_id TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    last_seen_at INTEGER NOT NULL,
                    PRIMARY KEY (source_normalized, league_normalized)
                );

                CREATE TABLE IF NOT EXISTS soccer_match_links (
                    allsports_match_id TEXT PRIMARY KEY,
                    provider_match_id TEXT,
                    provider_home_id TEXT,
                    provider_away_id TEXT,
                    source_home_name TEXT,
                    source_away_name TEXT,
                    provider_home_name TEXT,
                    provider_away_name TEXT,
                    start_timestamp INTEGER,
                    confidence REAL NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                """
            )
        _initialized_databases.add(absolute)


def _key_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _usage_day() -> str:
    # A RapidAPI normalmente contabiliza por UTC; manter o controle conservador
    # nessa mesma data evita liberar uma chave cedo demais no horário BRT.
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reserve_key(db_path: str) -> tuple[str | None, str | None]:
    global _rotation_cursor
    init_soccer_context_db(db_path)
    now = int(time.time())
    usage_day = _usage_day()
    with _quota_lock, _db(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for offset in range(len(SOCCER_API_KEYS)):
            idx = (_rotation_cursor + offset) % len(SOCCER_API_KEYS)
            key = SOCCER_API_KEYS[idx]
            key_id = _key_id(key)
            row = conn.execute(
                """SELECT request_count, status, server_remaining
                   FROM soccer_api_key_usage
                   WHERE key_id=? AND usage_date=?""",
                (key_id, usage_day),
            ).fetchone()
            count = int(row[0] or 0) if row else 0
            status = str(row[1] or "active") if row else "active"
            server_remaining = row[2] if row else None
            available_by_server = (
                server_remaining is None or int(server_remaining) > 10
            )
            if status == "active" and count < SOCCER_API_DAILY_LIMIT and available_by_server:
                conn.execute(
                    """INSERT INTO soccer_api_key_usage
                       (key_id, usage_date, request_count, status, updated_at)
                       VALUES (?, ?, 1, 'active', ?)
                       ON CONFLICT(key_id, usage_date) DO UPDATE SET
                           request_count=request_count+1,
                           updated_at=excluded.updated_at""",
                    (key_id, usage_day, now),
                )
                conn.commit()
                _rotation_cursor = idx
                return key, key_id
        conn.commit()
    return None, None


def _header_int(headers: Any, *names: str) -> int | None:
    for name in names:
        raw = headers.get(name) if headers is not None else None
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _record_response(
    db_path: str, key_id: str, status_code: int, headers: Any
) -> None:
    global _rotation_cursor
    remaining = _header_int(
        headers,
        "x-ratelimit-requests-remaining",
        "X-RateLimit-Requests-Remaining",
        "x-ratelimit-request-remaining",
    )
    reset_at = None
    if headers is not None:
        reset_at = (
            headers.get("x-ratelimit-requests-reset")
            or headers.get("X-RateLimit-Requests-Reset")
        )
    status = "active"
    if status_code == 429 or (remaining is not None and remaining <= 10):
        status = "exhausted"
    elif status_code in {401, 403}:
        status = "invalid"
    with _quota_lock, _db(db_path) as conn:
        conn.execute(
            """UPDATE soccer_api_key_usage
               SET status=?, server_remaining=?, reset_at=?,
                   last_http_status=?, updated_at=?,
                   request_count=CASE WHEN ?='exhausted'
                                      THEN MAX(request_count, ?)
                                      ELSE request_count END
               WHERE key_id=? AND usage_date=?""",
            (
                status,
                remaining,
                str(reset_at) if reset_at is not None else None,
                int(status_code),
                int(time.time()),
                status,
                SOCCER_API_DAILY_LIMIT,
                key_id,
                _usage_day(),
            ),
        )
        conn.commit()
    if status != "active" and SOCCER_API_KEYS:
        for idx, key in enumerate(SOCCER_API_KEYS):
            if _key_id(key) == key_id:
                _rotation_cursor = (idx + 1) % len(SOCCER_API_KEYS)
                break


def soccer_quota_status(db_path: str) -> dict[str, Any]:
    init_soccer_context_db(db_path)
    rows = {}
    with _db(db_path) as conn:
        for row in conn.execute(
            """SELECT key_id, request_count, status, server_remaining, reset_at
               FROM soccer_api_key_usage WHERE usage_date=?""",
            (_usage_day(),),
        ):
            rows[row[0]] = row[1:]
    keys = []
    total_available = 0
    for index, key in enumerate(SOCCER_API_KEYS, 1):
        key_id = _key_id(key)
        count, status, server_remaining, reset_at = rows.get(
            key_id, (0, "active", None, None)
        )
        local_available = max(0, SOCCER_API_DAILY_LIMIT - int(count or 0))
        if server_remaining is not None:
            local_available = min(local_available, max(0, int(server_remaining) - 10))
        if status != "active":
            local_available = 0
        total_available += local_available
        keys.append(
            {
                "index": index,
                "key_id": key_id,
                "used": int(count or 0),
                "available": local_available,
                "status": status,
                "server_remaining": server_remaining,
                "reset_at": reset_at,
            }
        )
    return {
        "date_utc": _usage_day(),
        "daily_limit_per_key": SOCCER_API_DAILY_LIMIT,
        "total_available": total_available,
        "keys": keys,
    }


def _rate_limit() -> None:
    global _last_request_at
    with _rate_lock:
        elapsed = time.monotonic() - _last_request_at
        remaining = SOCCER_API_MIN_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)
        _last_request_at = time.monotonic()


def _items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("result", "matches", "data", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for nested in ("matches", "events", "items", "data"):
                candidate = value.get(nested)
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, dict)]
    return []


def _pagination(payload: dict[str, Any], item_count: int) -> tuple[int, int]:
    containers = [payload]
    for key in ("pagination", "paging", "pager", "meta"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.insert(0, value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            containers.insert(0, value[0])

    def number(names: Iterable[str]) -> int | None:
        for container in containers:
            for name in names:
                try:
                    value = int(container.get(name))
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    return value
        return None

    per_page = number(("per_page", "perPage", "page_size", "limit")) or max(item_count, 25)
    total_pages = number(("total_pages", "pages", "page_count", "last_page"))
    total_items = number(("total", "total_items", "items"))
    if total_pages is None and total_items is not None and per_page > 0:
        total_pages = int(math.ceil(total_items / per_page))
    return max(1, int(total_pages or 1)), max(1, int(per_page))


class SoccerFootballInfoClient:
    """Cliente com cache SQLite e rotação conservadora das três chaves."""

    def __init__(
        self,
        db_path: str,
        session: Any | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.db_path = db_path
        self.session = session or requests.Session()
        self.progress = progress
        self.http_requests = 0
        self.cache_hits = 0
        init_soccer_context_db(db_path)

    def _emit(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def _cached_page(self, day_text: str, page: int) -> dict[str, Any] | None:
        with _db(self.db_path) as conn:
            row = conn.execute(
                """SELECT payload_json FROM soccer_page_cache
                   WHERE utc_date=? AND page=? AND expires_at>?""",
                (day_text, int(page), int(time.time())),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        self.cache_hits += 1
        return payload if isinstance(payload, dict) else None

    def _save_page(self, day_text: str, page: int, payload: dict[str, Any]) -> None:
        now = int(time.time())
        try:
            payload_day = datetime.strptime(day_text, "%Y%m%d").date()
        except ValueError:
            payload_day = datetime.now(timezone.utc).date()
        ttl = SOCCER_PAGE_CACHE_TTL_SECONDS
        if payload_day < datetime.now(timezone.utc).date():
            ttl = max(ttl, 30 * 86400)
        with _db(self.db_path) as conn:
            conn.execute(
                """INSERT INTO soccer_page_cache
                   (utc_date, page, payload_json, fetched_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(utc_date, page) DO UPDATE SET
                       payload_json=excluded.payload_json,
                       fetched_at=excluded.fetched_at,
                       expires_at=excluded.expires_at""",
                (day_text, int(page), json.dumps(payload, ensure_ascii=False), now, now + ttl),
            )

    def fetch_page(
        self, day_text: str, page: int, force_refresh: bool = False
    ) -> dict[str, Any] | None:
        if not force_refresh:
            cached = self._cached_page(day_text, page)
            if cached is not None:
                return cached

        attempts = max(1, len(SOCCER_API_KEYS))
        url = f"https://{SOCCER_API_HOST}/matches/day/full/"
        for _ in range(attempts):
            key, key_id = _reserve_key(self.db_path)
            if not key or not key_id:
                self._emit("cota local das três chaves contextuais encerrada")
                return None
            _rate_limit()
            try:
                response = self.session.get(
                    url,
                    headers={
                        "x-rapidapi-host": SOCCER_API_HOST,
                        "x-rapidapi-key": key,
                    },
                    params={"d": day_text, "p": int(page), "l": "en_US"},
                    timeout=30,
                )
            except requests.RequestException:
                self.http_requests += 1
                continue
            self.http_requests += 1
            _record_response(self.db_path, key_id, response.status_code, response.headers)
            if response.status_code in {401, 403, 429}:
                continue
            if response.status_code != 200:
                return None
            try:
                payload = response.json()
            except ValueError:
                return None
            if not isinstance(payload, dict):
                return None
            self._save_page(day_text, page, payload)
            return payload
        return None

    def fetch_day(
        self, utc_day: date, force_refresh: bool = False
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        day_text = utc_day.strftime("%Y%m%d")
        before_http = self.http_requests
        before_cache = self.cache_hits
        first = self.fetch_page(day_text, 1, force_refresh=force_refresh)
        if first is None:
            return [], {
                "utc_date": day_text,
                "complete": False,
                "pages_expected": 1,
                "pages_loaded": 0,
                "pages_missing": 1,
                "http_requests": self.http_requests - before_http,
                "cache_hits": self.cache_hits - before_cache,
            }

        events = _items(first)
        total_pages, per_page = _pagination(first, len(events))
        total_pages = min(total_pages, SOCCER_MAX_PAGES_PER_DAY)
        pages_loaded = 1
        page = 2
        # Se a API omitir a paginação, uma página cheia indica que devemos
        # continuar até encontrar uma página curta/vazia.
        unknown_pagination = total_pages == 1 and len(events) >= per_page
        page_limit = SOCCER_MAX_PAGES_PER_DAY if unknown_pagination else total_pages
        while page <= page_limit:
            payload = self.fetch_page(day_text, page, force_refresh=force_refresh)
            if payload is None:
                break
            page_items = _items(payload)
            pages_loaded += 1
            events.extend(page_items)
            if unknown_pagination and len(page_items) < per_page:
                total_pages = page
                break
            page += 1
        if unknown_pagination and page > page_limit:
            total_pages = page_limit

        deduped: dict[str, dict[str, Any]] = {}
        for index, event in enumerate(events):
            event_id = str(_event_id(event) or f"{day_text}:{index}")
            deduped[event_id] = event
        missing = max(0, total_pages - pages_loaded)
        return list(deduped.values()), {
            "utc_date": day_text,
            "complete": missing == 0,
            "pages_expected": total_pages,
            "pages_loaded": pages_loaded,
            "pages_missing": missing,
            "http_requests": self.http_requests - before_http,
            "cache_hits": self.cache_hits - before_cache,
        }

    def fetch_window(
        self, start: datetime, end: datetime, force_refresh: bool = False
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        current = start_utc.date()
        all_events: list[dict[str, Any]] = []
        day_meta = []
        while current <= end_utc.date():
            events, meta = self.fetch_day(current, force_refresh=force_refresh)
            all_events.extend(events)
            day_meta.append(meta)
            current += timedelta(days=1)
        filtered = []
        for event in all_events:
            timestamp = event_start_timestamp(event)
            if timestamp and int(start_utc.timestamp()) <= timestamp < int(end_utc.timestamp()):
                filtered.append(event)
        return filtered, {
            "complete": all(item.get("complete") for item in day_meta),
            "days": day_meta,
            "events_downloaded": len(all_events),
            "events_in_window": len(filtered),
            "http_requests": sum(int(item.get("http_requests", 0)) for item in day_meta),
            "cache_hits": sum(int(item.get("cache_hits", 0)) for item in day_meta),
            "pages_missing": sum(int(item.get("pages_missing", 0)) for item in day_meta),
            "quota": soccer_quota_status(self.db_path),
        }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    # Os horários sem offset desse endpoint são UTC, não BRT.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_start_timestamp(event: dict[str, Any]) -> int:
    for key in (
        "date", "datetime", "start_time", "startTime", "startTimestamp",
        "timestamp", "scheduled_at", "scheduledAt",
    ):
        parsed = _parse_datetime(event.get(key))
        if parsed is not None:
            return int(parsed.timestamp())
    return 0


def _event_id(event: dict[str, Any]) -> Any:
    return event.get("id") or event.get("match_id") or event.get("matchId")


def event_bet365_fid(event: dict[str, Any]) -> str:
    """Extrai o fixture id compartilhado pelo link Bet365 e pela AllSports."""
    url = str(event.get("bet365_url") or event.get("bet365Url") or "")
    match = re.search(r"/E(\d+)", url, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _fractional_to_decimal(value: Any) -> float:
    text = str(value or "").strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return round(float(numerator) / denominator_value + 1.0, 4)
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _current_full_time_odds(market: dict[str, Any]) -> tuple[float, float, float]:
    choices = market.get("choices") if isinstance(market, dict) else None
    if not isinstance(choices, list):
        return 0.0, 0.0, 0.0
    values = {}
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        name = str(choice.get("name") or "").strip().upper()
        # fractionalValue é a cotação atual. initialFractionalValue nunca é
        # usado, pois representa a abertura que o usuário decidiu ignorar.
        values[name] = _fractional_to_decimal(choice.get("fractionalValue"))
    return values.get("1", 0.0), values.get("X", 0.0), values.get("2", 0.0)


def index_allsports_odds_by_bet365_fid(
    odds_payloads: Iterable[dict[str, Any] | None],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Indexa os lotes de odds sem depender da agenda cara da AllSports."""
    indexed: dict[str, tuple[str, dict[str, Any]]] = {}
    for payload in odds_payloads:
        if not isinstance(payload, dict):
            continue
        block = payload.get("odds", payload)
        if not isinstance(block, dict):
            continue
        for allsports_id, market in block.items():
            if not isinstance(market, dict):
                continue
            fid = str(market.get("fid") or "").strip()
            if not fid:
                continue
            market_name = normalize_team_name(market.get("marketName") or "")
            if market_name and market_name not in {"full time", "fulltime"}:
                continue
            previous = indexed.get(fid)
            if previous is None or (previous[1].get("suspended") and not market.get("suspended")):
                indexed[fid] = (str(allsports_id), market)
    return indexed


def index_allsports_odds_by_match_id(
    odds_payloads: Iterable[dict[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    """Indexa o mesmo lote pelo ID AllSports para o fallback sem ``fid``."""
    indexed: dict[str, dict[str, Any]] = {}
    for payload in odds_payloads:
        if not isinstance(payload, dict):
            continue
        block = payload.get("odds", payload)
        if not isinstance(block, dict):
            continue
        for match_id, market in block.items():
            if not isinstance(market, dict):
                continue
            market_name = normalize_team_name(market.get("marketName") or "")
            if market_name and market_name not in {"full time", "fulltime"}:
                continue
            previous = indexed.get(str(match_id))
            if previous is None or (previous.get("suspended") and not market.get("suspended")):
                indexed[str(match_id)] = market
    return indexed


_TRANSLATION = str.maketrans({"ı": "i", "ł": "l", "ø": "o", "đ": "d", "ß": "ss"})
_CLUB_STOPWORDS = {
    "fc", "cf", "sc", "ac", "afc", "fk", "sk", "club", "football", "futbol",
    "futebol", "de", "do", "da", "the", "calcio", "krc",
}
_GENERIC_SINGLE_TOKENS = {"united", "city", "town", "athletic", "sporting", "real"}
_TOKEN_ALIASES = {
    "utd": "united", "ath": "athletic", "st": "saint", "dep": "deportivo",
    "lyonnais": "lyon", "w": "women",
}


def team_category_signature(name: Any) -> tuple[bool, str, bool]:
    """Return (women, youth age/generic, reserve) from explicit name markers.

    These markers are identity, not weak spelling suffixes. Ignoring them joins
    first teams to women's, youth and reserve squads and contaminates every
    downstream form/rating feature.
    """
    raw = str(name or "")
    normalized = normalize_team_name(raw)
    tokens = normalized.split()
    women = bool(re.search(
        r"(?:^|\W)(?:w|women|womens|woman|feminino|feminina|femenino|femenina|frauen)(?:\W|$)",
        raw.lower(),
    ))
    age_match = re.search(r"(?:^| )(u(?:1[2-9]|2[0-4]))(?: |$)", normalized)
    youth = age_match.group(1) if age_match else (
        "youth" if any(token in {"youth", "academy", "junior", "juniors", "primavera"}
                       for token in tokens) else ""
    )
    reserve = bool(
        any(token in {"reserve", "reserves", "reserva", "reservas"} for token in tokens)
        or re.search(r"(?:^|\s)(?:ii|2|b)(?:\s|$)", normalized)
        or "b team" in normalized or "team b" in normalized
    )
    return women, youth, reserve


def team_categories_compatible(left: Any, right: Any) -> bool:
    """Require all explicit squad-category markers to agree exactly."""
    return team_category_signature(left) == team_category_signature(right)


def normalize_team_name(name: Any) -> str:
    text = str(name or "").strip().lower().translate(_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\b(u|sub)[- ]?(1[2-9]|2[0-4])\b", r"u\2", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    normalized = [_TOKEN_ALIASES.get(token, token) for token in tokens]
    return " ".join(normalized)


def _meaningful_tokens(name: Any) -> list[str]:
    return [token for token in normalize_team_name(name).split() if token not in _CLUB_STOPWORDS]


def team_name_similarity(left: Any, right: Any) -> float:
    if not team_categories_compatible(left, right):
        return 0.0
    left_norm = normalize_team_name(left)
    right_norm = normalize_team_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm.replace(" ", "") == right_norm.replace(" ", ""):
        return 0.99
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    left_set, right_set = set(left_tokens), set(right_tokens)
    if left_set and right_set and left_set == right_set:
        return 0.98
    containment = 0.0
    matched_right = set()
    fuzzy_matches = 0
    for left_token in left_tokens:
        for right_index, right_token in enumerate(right_tokens):
            if right_index in matched_right:
                continue
            shorter_length = min(len(left_token), len(right_token))
            if (left_token == right_token or
                    (shorter_length >= 4 and
                     (left_token.startswith(right_token) or right_token.startswith(left_token)))):
                matched_right.add(right_index)
                fuzzy_matches += 1
                break
    fuzzy_subset = (
        fuzzy_matches == min(len(left_tokens), len(right_tokens))
        if left_tokens and right_tokens else False
    )
    if left_set and right_set and ((left_set <= right_set or right_set <= left_set) or fuzzy_subset):
        shorter = left_set if len(left_set) <= len(right_set) else right_set
        if not (len(shorter) == 1 and next(iter(shorter)) in _GENERIC_SINGLE_TOKENS):
            containment = 0.93 if len(shorter) == 1 else 0.96
    union_size = len(left_tokens) + len(right_tokens) - fuzzy_matches
    jaccard = fuzzy_matches / union_size if union_size else 0.0
    sequence = SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
    compact = SequenceMatcher(None, left_norm.replace(" ", ""), right_norm.replace(" ", "")).ratio()
    return round(max(containment, 0.62 * sequence + 0.38 * jaccard, compact * 0.90), 4)


def _first_dict(source: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, dict):
            return value
    return {}


def event_home_team(event: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        event,
        ("teamA", "home_team", "homeTeam", "home", "localteam", "localTeam", "team_home"),
    )


def event_away_team(event: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        event,
        ("teamB", "away_team", "awayTeam", "away", "visitorteam", "visitorTeam", "team_away"),
    )


def _team_name(team: dict[str, Any]) -> str:
    return str(team.get("name") or team.get("title") or team.get("short_name") or "").strip()


def _team_id(team: dict[str, Any]) -> str:
    value = team.get("id") or team.get("team_id") or team.get("teamId")
    return str(value or "").strip()


def _league(event: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(event, ("championship", "league", "tournament", "competition"))


def _league_name(event: dict[str, Any]) -> str:
    league = _league(event)
    return str(league.get("name") or league.get("title") or "").strip()


def _country_name(event: dict[str, Any]) -> str:
    league = _league(event)
    country = _first_dict(league, ("country", "category")) or _first_dict(
        event, ("country", "category")
    )
    return str(country.get("name") or country.get("title") or "").strip()


def _season_label(event: dict[str, Any]) -> str:
    league = _league(event)
    season = _first_dict(event, ("season",)) or _first_dict(
        league, ("season", "currentSeason")
    )
    return str(season.get("id") or season.get("name") or season.get("title") or "").strip()


def _allsports_context_ids(*sources: dict[str, Any] | None) -> dict[str, str]:
    """Extrai IDs AllSports quando eles já vieram no lote; não faz nova chamada."""
    result = {"tournament_id": "", "season_id": "", "unique_tournament_id": "",
              "home_id": "", "away_id": ""}
    for source in sources:
        if not isinstance(source, dict):
            continue
        event = source.get("event") if isinstance(source.get("event"), dict) else source
        tournament = _first_dict(event, ("tournament", "competition", "league"))
        season = _first_dict(event, ("season",)) or _first_dict(tournament, ("season", "currentSeason"))
        unique = _first_dict(tournament, ("uniqueTournament", "unique_tournament"))
        home, away = event_home_team(event), event_away_team(event)
        result["tournament_id"] = result["tournament_id"] or str(
            event.get("tournamentId") or tournament.get("id") or "")
        result["season_id"] = result["season_id"] or str(
            event.get("seasonId") or season.get("id") or tournament.get("seasonId") or "")
        result["unique_tournament_id"] = result["unique_tournament_id"] or str(
            event.get("uniqueTournamentId") or unique.get("id") or "")
        result["home_id"] = result["home_id"] or _team_id(home)
        result["away_id"] = result["away_id"] or _team_id(away)
    return result


def _resolve_local_league_mapping(db_path: str, league_name: str) -> tuple[str, str]:
    """Relaciona liga Soccer a IDs já aprendidos, aceitando só vínculo inequívoco."""
    source = normalize_team_name(league_name)
    if not source:
        return "", ""
    cache_key = (os.path.abspath(db_path), source)
    with _league_mapping_lock:
        if cache_key in _league_mapping_cache:
            return _league_mapping_cache[cache_key]
    try:
        with _db(db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mapeamento_ligas'"
            ).fetchone()
            if not exists:
                result = ("", "")
                with _league_mapping_lock:
                    _league_mapping_cache[cache_key] = result
                return result
            rows = conn.execute(
                """SELECT liga_nome, tournament_id, season_id
                   FROM mapeamento_ligas
                   WHERE tournament_id IS NOT NULL AND tournament_id!=''
                     AND season_id IS NOT NULL AND season_id!=''"""
            ).fetchall()
    except sqlite3.Error:
        return "", ""
    ranked = []
    source_tokens = set(source.split())
    for name, tournament_id, season_id in rows:
        target = normalize_team_name(name)
        if not target:
            continue
        score = 1.0 if source == target else SequenceMatcher(None, source, target).ratio()
        target_tokens = set(target.split())
        if source_tokens and target_tokens:
            score = max(score, len(source_tokens & target_tokens) / len(source_tokens | target_tokens))
        ranked.append((score, str(tournament_id), str(season_id)))
    ranked.sort(reverse=True)
    if not ranked:
        result = ("", "")
        with _league_mapping_lock:
            _league_mapping_cache[cache_key] = result
        return result
    best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    # Exato sempre vale; fuzzy somente muito forte e claramente único.
    exact_is_unique = not (
        best[0] >= 0.999 and len(ranked) > 1 and ranked[1][0] >= 0.999
        and ranked[1][1:] != best[1:]
    )
    if (best[0] >= 0.999 and exact_is_unique) or (
        best[0] >= 0.93 and best[0] - second_score >= 0.06
    ):
        result = (best[1], best[2])
    else:
        result = ("", "")
    with _league_mapping_lock:
        _league_mapping_cache[cache_key] = result
    return result


def match_allsports_game(
    game: dict[str, Any], events: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    source_ts = int(game.get("Timestamp") or game.get("startTimestamp") or 0)
    source_home = game.get("Time Casa") or game.get("home_team") or ""
    source_away = game.get("Time Fora") or game.get("away_team") or ""
    source_league = game.get("Liga") or game.get("liga") or ""
    candidates = []
    for event in events:
        target_ts = event_start_timestamp(event)
        time_diff = abs(target_ts - source_ts) if source_ts and target_ts else 10**9
        if time_diff > 45 * 60:
            continue
        target_home = _team_name(event_home_team(event))
        target_away = _team_name(event_away_team(event))
        home_score = team_name_similarity(source_home, target_home)
        away_score = team_name_similarity(source_away, target_away)
        if min(home_score, away_score) < 0.58:
            continue
        league_score = team_name_similarity(source_league, _league_name(event))
        time_score = max(0.0, 1.0 - time_diff / (45 * 60))
        total = 0.44 * home_score + 0.44 * away_score + 0.08 * league_score + 0.04 * time_score
        candidates.append((total, home_score, away_score, time_diff, event))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    margin = best[0] - candidates[1][0] if len(candidates) > 1 else 1.0
    # Horário exato + dois nomes razoáveis é forte; em horário aproximado
    # exigimos nomes mais próximos e margem contra o segundo candidato.
    accepted = (
        best[3] <= 10 * 60 and best[0] >= 0.72 and min(best[1], best[2]) >= 0.62
    ) or (
        best[3] <= 45 * 60 and best[0] >= 0.84
        and min(best[1], best[2]) >= 0.76 and margin >= 0.04
    )
    if not accepted:
        return None
    return {
        "event": best[4],
        "confidence": round(best[0], 4),
        "home_similarity": best[1],
        "away_similarity": best[2],
        "time_difference_seconds": best[3],
        "unique_margin": round(margin, 4),
    }


def _performance_sequence(team: dict[str, Any]) -> list[str]:
    raw = team.get("perf")
    if raw in (None, ""):
        raw = team.get("form")
    if isinstance(raw, dict):
        raw = (
            raw.get("l_5_matches") or raw.get("last_5_matches")
            or raw.get("results") or raw.get("form") or raw.get("value")
        )
    if isinstance(raw, (list, tuple)):
        values = [str(item).upper()[:1] for item in raw]
    else:
        values = re.findall(r"[WDLVED]", str(raw or "").upper())
    # O endpoint é consultado com en_US, portanto D significa draw. Ainda
    # aceitamos V/E quando algum campeonato devolve abreviações localizadas.
    mapping = {"V": "W", "E": "D"}
    return [mapping.get(item, item) for item in values if mapping.get(item, item) in {"W", "D", "L"}][-5:]


def _numeric(team: dict[str, Any], *names: str) -> float:
    wanted = {normalize_team_name(name).replace(" ", "_") for name in names}

    def walk(value: Any) -> float | None:
        if not isinstance(value, dict):
            return None
        for key, item in value.items():
            key_norm = normalize_team_name(key).replace(" ", "_")
            if key_norm in wanted:
                try:
                    return float(item)
                except (TypeError, ValueError):
                    pass
        for item in value.values():
            found = walk(item)
            if found is not None:
                return found
        return None

    return float(walk(team) or 0.0)


def _cache_team(db_path: str, team: dict[str, Any], now: int) -> tuple[str, bool]:
    name = _team_name(team)
    provider_id = _team_id(team) or f"name:{normalize_team_name(name)}"
    if not name or provider_id == "name:":
        return "", False
    with _db(db_path) as conn:
        row = conn.execute(
            """SELECT expires_at, perf_json, raw_json
               FROM soccer_team_form_cache WHERE provider_team_id=?""",
            (provider_id,),
        ).fetchone()
        sequence = _performance_sequence(team)
        perf_json = json.dumps(sequence)
        raw_json = json.dumps(team, ensure_ascii=False, sort_keys=True)
        # A agenda diária pode trazer uma sequência W/D/L nova antes do TTL.
        # Nesse caso o conteúdo vence o relógio: nunca mantemos a forma antiga
        # apenas porque a linha ainda não expirou.
        changed = bool(
            row and (
                str(row[1] or "[]") != perf_json
                or str(row[2] or "{}") != raw_json
            )
        )
        refresh = not row or int(row[0] or 0) <= now or changed
        if refresh:
            conn.execute(
                """INSERT INTO soccer_team_form_cache
                   (provider_team_id, provider_name, normalized_name, perf_json,
                    position, raw_json, captured_at, expires_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider_team_id) DO UPDATE SET
                       provider_name=excluded.provider_name,
                       normalized_name=excluded.normalized_name,
                       perf_json=excluded.perf_json,
                       position=excluded.position,
                       raw_json=excluded.raw_json,
                       captured_at=excluded.captured_at,
                       expires_at=excluded.expires_at,
                       last_seen_at=excluded.last_seen_at""",
                (
                    provider_id,
                    name,
                    normalize_team_name(name),
                    perf_json,
                    _numeric(team, "position", "rank", "standing_position") or None,
                    raw_json,
                    now,
                    now + SOCCER_FORM_TTL_SECONDS,
                    now,
                ),
            )
        else:
            conn.execute(
                "UPDATE soccer_team_form_cache SET last_seen_at=? WHERE provider_team_id=?",
                (now, provider_id),
            )
    return provider_id, refresh


def _save_match_link(
    db_path: str, game: dict[str, Any], matched: dict[str, Any], now: int
) -> int:
    event = matched["event"]
    home = event_home_team(event)
    away = event_away_team(event)
    home_id, refreshed_home = _cache_team(db_path, home, now)
    away_id, refreshed_away = _cache_team(db_path, away, now)
    allsports_id = str(game.get("ID") or game.get("match_id") or "")
    source_home = str(game.get("Time Casa") or game.get("home_team") or "")
    source_away = str(game.get("Time Fora") or game.get("away_team") or "")
    league_norm = normalize_team_name(game.get("Liga") or game.get("liga") or "")
    with _db(db_path) as conn:
        conn.execute(
            """INSERT INTO soccer_match_links
               (allsports_match_id, provider_match_id, provider_home_id,
                provider_away_id, source_home_name, source_away_name,
                provider_home_name, provider_away_name, start_timestamp,
                confidence, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(allsports_match_id) DO UPDATE SET
                   provider_match_id=excluded.provider_match_id,
                   provider_home_id=excluded.provider_home_id,
                   provider_away_id=excluded.provider_away_id,
                   source_home_name=excluded.source_home_name,
                   source_away_name=excluded.source_away_name,
                   provider_home_name=excluded.provider_home_name,
                   provider_away_name=excluded.provider_away_name,
                   start_timestamp=excluded.start_timestamp,
                   confidence=excluded.confidence,
                   updated_at=excluded.updated_at""",
            (
                allsports_id,
                str(_event_id(event) or ""),
                home_id,
                away_id,
                source_home,
                source_away,
                _team_name(home),
                _team_name(away),
                event_start_timestamp(event),
                float(matched["confidence"]),
                now,
            ),
        )
        # Só aprende alias com pareamento forte; um erro aqui contaminaria
        # todos os jogos futuros daquele time/liga.
        for source_name, provider_id, provider_name, score in (
            (source_home, home_id, _team_name(home), matched["home_similarity"]),
            (source_away, away_id, _team_name(away), matched["away_similarity"]),
        ):
            if provider_id and score >= 0.86 and matched["confidence"] >= 0.80:
                conn.execute(
                    """INSERT INTO soccer_team_aliases
                       (source_normalized, league_normalized, source_name,
                        provider_team_id, provider_name, confidence, last_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(source_normalized, league_normalized) DO UPDATE SET
                           provider_team_id=CASE
                               WHEN excluded.provider_team_id=soccer_team_aliases.provider_team_id
                                    OR excluded.confidence>soccer_team_aliases.confidence
                               THEN excluded.provider_team_id
                               ELSE soccer_team_aliases.provider_team_id END,
                           provider_name=CASE
                               WHEN excluded.provider_team_id=soccer_team_aliases.provider_team_id
                                    OR excluded.confidence>soccer_team_aliases.confidence
                               THEN excluded.provider_name
                               ELSE soccer_team_aliases.provider_name END,
                           confidence=MAX(confidence, excluded.confidence),
                           last_seen_at=excluded.last_seen_at""",
                    (
                        normalize_team_name(source_name), league_norm, source_name,
                        provider_id, provider_name, float(score), now,
                    ),
                )
    return int(refreshed_home) + int(refreshed_away)


def collect_soccer_radar_games(
    db_path: str,
    start: datetime,
    end: datetime,
    odds_payloads: Iterable[dict[str, Any] | None],
    min_home_odd: float = 1.99,
    min_away_odd: float = 1.99,
    progress: Callable[[str], None] | None = None,
    client: SoccerFootballInfoClient | None = None,
    allsports_fallback_loader: Callable[[set[str]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Monta o radar com agenda Soccer e odds atuais da AllSports.

    A ligação é determinística: ``bet365_url`` da Soccer contém ``E<fid>`` e
    o lote diário da AllSports expõe o mesmo ``fid``. Isso elimina a consulta
    de centenas de torneios da AllSports.
    """
    init_soccer_context_db(db_path)
    phase4_started_at = time.time()
    client = client or SoccerFootballInfoClient(db_path, progress=progress)
    events, fetch_meta = client.fetch_window(start, end)
    odds_index = index_allsports_odds_by_bet365_fid(odds_payloads)
    from phase2_observations import append as observe, new_run, market_probabilities
    from phase4_research import ensure_challenger_registry
    research_run_id = new_run()
    ensure_challenger_registry(db_path)
    observe(db_path, research_run_id, '', 'collection_started', {
        'window_start': start.isoformat(), 'window_end': end.isoformat(),
        'total_games_found': len(events), 'fetch_meta': fetch_meta,
    })
    odds_by_match_id = index_allsports_odds_by_match_id(odds_payloads)
    missing_competitions = {
        _league_name(event) for event in events
        if not event_bet365_fid(event) and _league_name(event)
    }
    fallback_events: list[dict[str, Any]] = []
    fallback_meta: dict[str, Any] = {}
    if missing_competitions and allsports_fallback_loader is not None:
        try:
            fallback_payload = allsports_fallback_loader(missing_competitions) or {}
            fallback_events = [
                item for item in (fallback_payload.get("events") or [])
                if isinstance(item, dict)
            ]
            fallback_meta = dict(fallback_payload.get("_meta") or {})
        except Exception as exc:  # fallback nunca pode derrubar o radar
            if progress:
                progress(f"fallback AllSports sem fid falhou: {exc}")
    brt = timezone(timedelta(hours=-3))
    now = int(time.time())
    games: list[dict[str, Any]] = []
    research_candidates: list[dict[str, Any]] = []
    research_seen_ids: set[str] = set()
    seen_allsports_ids: set[str] = set()
    counters = {
        "events_with_bet365_fid": 0,
        "linked_to_current_odds": 0,
        "missing_bet365_fid": 0,
        "missing_current_odds": 0,
        "suspended_markets": 0,
        "invalid_odds": 0,
        "outside_odds_filter": 0,
        "forms_refreshed": 0,
        "fallback_events": len(fallback_events),
        "fallback_linked": 0,
        "fallback_without_odds": 0,
        "phase4_eligible": 0,
        "phase4_non_eligible": 0,
        "phase4_b1_home_low": 0,
        "phase4_b2_away_low": 0,
        "phase4_b3_both_low": 0,
        "phase4_capture_failures": 0,
        "phase4_health_warnings": 0,
    }
    unmatched_examples = []
    for event_index, event in enumerate(events):
        home = event_home_team(event)
        away = event_away_team(event)
        home_name = _team_name(home)
        away_name = _team_name(away)
        source_id = str(_event_id(event) or '')
        observation_id = 'soccer:' + (source_id or 'missing:' + str(event_index))
        def collection_decision(reason, linked_id=None):
            observe(db_path, research_run_id, observation_id, 'collection_decision', {
                'reason': reason, 'allsports_match_id': linked_id,
                'features': None, 'quality_score': None, 'gate_decision': 'DISABLED',
                'feature_status': 'not_evaluated_at_collection',
            })
        observe(db_path, research_run_id, observation_id, 'agenda', {
            'provider': 'soccer-football-info', 'match_id': source_id,
            'home_team': home_name, 'away_team': away_name,
            'league': _league_name(event), 'kickoff': event_start_timestamp(event),
        })
        if not home_name or not away_name:
            collection_decision('invalid_team_names')
            continue
        fid = event_bet365_fid(event)
        odds_match = None
        allsports_event = None
        if fid:
            counters["events_with_bet365_fid"] += 1
            odds_match = odds_index.get(fid)
        else:
            counters["missing_bet365_fid"] += 1
            source_game = {
                "Timestamp": event_start_timestamp(event),
                "Time Casa": home_name, "Time Fora": away_name,
                "Liga": _league_name(event),
            }
            linked = match_allsports_game(source_game, fallback_events)
            if linked:
                allsports_event = linked["event"]
                fallback_id = str(allsports_event.get("id") or "")
                market = odds_by_match_id.get(fallback_id)
                if market is not None:
                    odds_match = (fallback_id, market)
                    counters["fallback_linked"] += 1
                else:
                    counters["fallback_without_odds"] += 1
        if not odds_match:
            counters["missing_current_odds"] += 1
            collection_decision('missing_current_odds_or_link')
            if len(unmatched_examples) < 8:
                cause = "sem fid/fallback" if not fid else "sem odds atuais"
                unmatched_examples.append(f"{home_name} x {away_name} ({cause})")
            continue
        allsports_id, market = odds_match
        counters["linked_to_current_odds"] += 1
        observed_odds = _current_full_time_odds(market)
        market_seen_at = time.time()
        observe(db_path, research_run_id, str(allsports_id), 'market', {
            'provider': 'allsports', 'soccer_event_id': source_id,
            'home_odd': observed_odds[0], 'draw_odd': observed_odds[1],
            'away_odd': observed_odds[2], 'vendor_quote_timestamp': None,
            'suspended': bool(market.get('suspended')),
            'kickoff': event_start_timestamp(event),
            'vendor_quote_timestamp_verified': False,
            'prices_seen_at': market_seen_at,
            'as_observed_benchmark': market_probabilities(observed_odds, market_seen_at,
                event_start_timestamp(event), bool(market.get('suspended'))),
            'eligible_prices': min(observed_odds) > 1 and
                observed_odds[0] > min_home_odd and observed_odds[2] > min_away_odd,
        })
        from phase3_research import capture_markets
        capture_markets(db_path, research_run_id, allsports_id, market, 'allsports',
                        event_start_timestamp(event), market_seen_at)
        if market.get("suspended"):
            counters["suspended_markets"] += 1
            collection_decision('suspended_market', allsports_id)
            continue
        odd_home, odd_draw, odd_away = _current_full_time_odds(market)
        if min(odd_home, odd_draw, odd_away) <= 1.0:
            counters["invalid_odds"] += 1
            collection_decision('invalid_odds', allsports_id)
            continue
        # Phase 4 freezes every valid-market candidate before the production
        # odds rule. This is research-only and cannot add a game to the radar.
        from phase4_research import capture_eligibility_candidate
        phase4_capture = capture_eligibility_candidate(
            db_path, research_run_id, allsports_id,
            [odd_home, odd_draw, odd_away], event_start_timestamp(event),
            market_seen_at, provider='allsports', home_team=home_name,
            away_team=away_name, league=_league_name(event), country=_country_name(event),
            season=_season_label(event), source_event_id=source_id,
            suspended=bool(market.get('suspended')),
        )
        eligibility = phase4_capture["group"]
        counters["phase4_capture_failures"] += int(not phase4_capture["saved"])
        counters["phase4_health_warnings"] += int(phase4_capture["health"] != "PASS")
        if eligibility == 'A_ELIGIBLE':
            counters["phase4_eligible"] += 1
        elif eligibility.startswith('B'):
            counters["phase4_non_eligible"] += 1
            if eligibility == 'B1_HOME_LOW':
                counters["phase4_b1_home_low"] += 1
            elif eligibility == 'B2_AWAY_LOW':
                counters["phase4_b2_away_low"] += 1
            elif eligibility == 'B3_BOTH_LOW':
                counters["phase4_b3_both_low"] += 1
        research_timestamp = event_start_timestamp(event)
        if eligibility != 'INVALID' and research_timestamp and allsports_id not in research_seen_ids:
            research_league = _league_name(event) or "Liga não informada"
            research_ids = _allsports_context_ids(allsports_event, market)
            if not research_ids["tournament_id"] or not research_ids["season_id"]:
                mapped_tournament, mapped_season = _resolve_local_league_mapping(
                    db_path, research_league
                )
                research_ids["tournament_id"] = research_ids["tournament_id"] or mapped_tournament
                research_ids["season_id"] = research_ids["season_id"] or mapped_season
            research_candidates.append({
                "ID": allsports_id,
                "Liga": research_league,
                "Time Casa": home_name,
                "Time Fora": away_name,
                "Odd Casa": float(odd_home),
                "Empate": float(odd_draw),
                "Odd Fora": float(odd_away),
                "Timestamp": research_timestamp,
                "Home_ID": research_ids["home_id"],
                "Away_ID": research_ids["away_id"],
                "Tournament_ID": research_ids["tournament_id"],
                "Season_ID": research_ids["season_id"],
                "Unique_Tournament_ID": research_ids["unique_tournament_id"],
                "Research_Run_ID": research_run_id,
                "Eligibility_Group": eligibility,
            })
            research_seen_ids.add(allsports_id)
        if odd_home <= min_home_odd or odd_away <= min_away_odd:
            counters["outside_odds_filter"] += 1
            collection_decision('outside_odds_filter', allsports_id)
            continue
        if allsports_id in seen_allsports_ids:
            collection_decision('duplicate_match', allsports_id)
            continue
        timestamp = event_start_timestamp(event)
        if not timestamp:
            collection_decision('missing_kickoff', allsports_id)
            continue
        event_brt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(brt)
        league_name = _league_name(event) or "Liga não informada"
        context_ids = _allsports_context_ids(allsports_event, market)
        if not context_ids["tournament_id"] or not context_ids["season_id"]:
            mapped_tournament, mapped_season = _resolve_local_league_mapping(
                db_path, league_name
            )
            context_ids["tournament_id"] = context_ids["tournament_id"] or mapped_tournament
            context_ids["season_id"] = context_ids["season_id"] or mapped_season
        game = {
            "ID": allsports_id,
            "Dia": event_brt.strftime("%d/%m"),
            "Hora": event_brt.strftime("%H:%M"),
            "Liga": league_name,
            "Time Casa": home_name,
            "Time Fora": away_name,
            "Odd Casa": float(odd_home),
            "Empate": float(odd_draw),
            "Odd Fora": float(odd_away),
            "Timestamp": timestamp,
            "Confronto": f"{home_name} vs {away_name}",
            # IDs de time/torneio não são intercambiáveis entre provedores.
            # O ID AllSports da partida é preservado para odds e auditoria.
            "Home_ID": context_ids["home_id"],
            "Away_ID": context_ids["away_id"],
            "Tournament_ID": context_ids["tournament_id"],
            "Season_ID": context_ids["season_id"],
            "Unique_Tournament_ID": context_ids["unique_tournament_id"],
            "Soccer_Home_ID": _team_id(home),
            "Soccer_Away_ID": _team_id(away),
            "Soccer_Event_ID": str(_event_id(event) or ""),
            "Bet365_FID": fid,
            "Odds_Source": "AllSports atual",
            "Research_Run_ID": research_run_id,
        }
        matched = {
            "event": event,
            "confidence": 1.0,
            "home_similarity": 1.0,
            "away_similarity": 1.0,
            "time_difference_seconds": 0,
            "unique_margin": 1.0,
        }
        counters["forms_refreshed"] += _save_match_link(db_path, game, matched, now)
        game["Soccer_Context_Matched"] = True
        game["Soccer_Context_Confidence"] = 1.0
        games.append(game)
        collection_decision('collection_eligible', allsports_id)
        seen_allsports_ids.add(allsports_id)

    observe(db_path, research_run_id, '', 'collection_finished', {
        'total_games_found': len(events), 'counters': counters,
        'qualified_games': len(games), 'quality_gate': 'DISABLED',
    })
    from phase4_research import finalize_run_health
    phase4_health = finalize_run_health(
        db_path, research_run_id, counters, total_events=len(events),
        operational_candidates=len(games), started_at=phase4_started_at,
    )
    return games, {
        "research_run_id": research_run_id,
        **fetch_meta,
        **counters,
        "fallback_meta": fallback_meta,
        "odds_entries": len(odds_index),
        "qualified_games": len(games),
        "research_candidates": research_candidates,
        "phase4_health": phase4_health,
        "unmatched_examples": unmatched_examples,
    }


def enrich_allsports_games(
    games: list[dict[str, Any]],
    db_path: str,
    start: datetime,
    end: datetime,
    progress: Callable[[str], None] | None = None,
    client: SoccerFootballInfoClient | None = None,
) -> dict[str, Any]:
    """Vincula agenda AllSports ao contexto semanal sem tocar nas odds."""
    init_soccer_context_db(db_path)
    client = client or SoccerFootballInfoClient(db_path, progress=progress)
    events, fetch_meta = client.fetch_window(start, end)
    matched_count = 0
    refreshed_forms = 0
    now = int(time.time())
    unmatched_names = []
    for game in games:
        matched = match_allsports_game(game, events)
        if not matched:
            unmatched_names.append(
                f"{game.get('Time Casa', '')} x {game.get('Time Fora', '')}"
            )
            continue
        matched_count += 1
        refreshed_forms += _save_match_link(db_path, game, matched, now)
        event = matched["event"]
        game["Soccer_Context_Matched"] = True
        game["Soccer_Context_Confidence"] = matched["confidence"]
        game["Soccer_Home_ID"] = _team_id(event_home_team(event))
        game["Soccer_Away_ID"] = _team_id(event_away_team(event))
    return {
        **fetch_meta,
        "games": len(games),
        "matched": matched_count,
        "unmatched": len(games) - matched_count,
        "forms_refreshed": refreshed_forms,
        "unmatched_examples": unmatched_names[:8],
    }


def _load_team_context(conn: sqlite3.Connection, provider_id: str) -> dict[str, Any]:
    if not provider_id:
        return {}
    row = conn.execute(
        """SELECT perf_json, position, raw_json, captured_at, expires_at
           FROM soccer_team_form_cache WHERE provider_team_id=?""",
        (provider_id,),
    ).fetchone()
    if not row:
        return {}
    try:
        perf = json.loads(row[0] or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        perf = []
    try:
        raw = json.loads(row[2] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        raw = {}
    return {
        "perf": perf if isinstance(perf, list) else [],
        "position": float(row[1] or 0.0),
        "raw": raw if isinstance(raw, dict) else {},
        "captured_at": int(row[3] or 0),
        "stale": int(row[4] or 0) <= int(time.time()),
    }


def _team_features(prefix: str, context: dict[str, Any]) -> dict[str, float]:
    from pregame_form_quality import sfi_extended_performance, sfi_goal_integrity
    perf = [item for item in context.get("perf", []) if item in {"W", "D", "L"}][-5:]
    games = len(perf)
    wins = perf.count("W")
    draws = perf.count("D")
    losses = perf.count("L")
    ppg = (3 * wins + draws) / games if games else 0.0
    raw = context.get("raw", {})
    raw_perf = raw.get("perf") if isinstance(raw, dict) else None
    # SFI l_5_matches is newest -> oldest. Verified in 524 comparisons with
    # non-palindromic dated pregame histories (one reverse match, 179 different
    # windows). Keep unknown legacy formats unchanged; never reverse the cache
    # in place or the next read would reverse it again.
    newest_first = isinstance(raw_perf, dict) and bool(
        raw_perf.get("l_5_matches") or raw_perf.get("last_5_matches"))
    recent = perf[:3] if newest_first else perf[-3:]
    recent_points = sum(3 if item == "W" else 1 if item == "D" else 0 for item in recent)
    return {
        f"form_sfi_{prefix}_available": float(bool(games)),
        f"form_sfi_{prefix}_games": float(games),
        f"form_sfi_{prefix}_ppg": float(ppg),
        f"form_sfi_{prefix}_win_rate": wins / games if games else 0.0,
        f"form_sfi_{prefix}_draw_rate": draws / games if games else 0.0,
        f"form_sfi_{prefix}_loss_rate": losses / games if games else 0.0,
        f"form_sfi_{prefix}_recent_points_3": recent_points / max(1, len(recent) * 3),
        f"form_sfi_{prefix}_position": float(context.get("position", 0.0)),
        f"form_sfi_{prefix}_goals_for_avg": _numeric(
            raw, "avg_goals_scored", "average_goals_scored", "goals_for_avg"
        ),
        f"form_sfi_{prefix}_goals_against_avg": _numeric(
            raw, "avg_goals_conceded", "average_goals_conceded", "goals_against_avg"
        ),
        f"form_sfi_{prefix}_stale": float(bool(context.get("stale"))),
        f"form_sfi_{prefix}_sequence_newest_first": float(newest_first),
        **{f"form_sfi_{prefix}_{key}": value
           for key, value in sfi_goal_integrity(raw).items()},
        **{f"form_sfi_{prefix}_{key}": value
           for key, value in sfi_extended_performance(raw).items()},
    }


def get_soccer_context_features(
    db_path: str,
    allsports_match_id: Any,
    home_name: Any = "",
    away_name: Any = "",
    league: Any = "",
) -> dict[str, float]:
    """Retorna apenas variáveis pré-jogo; nunca expõe odds da segunda API."""
    init_soccer_context_db(db_path)
    match_id = str(allsports_match_id or "")
    home_id = away_id = ""
    link_confidence = 0.0
    with _db(db_path) as conn:
        if match_id:
            row = conn.execute(
                """SELECT provider_home_id, provider_away_id, confidence,
                          source_home_name, source_away_name,
                          provider_home_name, provider_away_name
                   FROM soccer_match_links WHERE allsports_match_id=?""",
                (match_id,),
            ).fetchone()
            if row:
                home_ok = not home_name or max(
                    team_name_similarity(home_name, row[3]),
                    team_name_similarity(home_name, row[5]),
                ) >= 0.68
                away_ok = not away_name or max(
                    team_name_similarity(away_name, row[4]),
                    team_name_similarity(away_name, row[6]),
                ) >= 0.68
                if home_ok and away_ok and str(row[0] or "") != str(row[1] or ""):
                    home_id, away_id, link_confidence = (
                        str(row[0] or ""), str(row[1] or ""), float(row[2] or 0.0)
                    )
        if not home_id and home_name:
            league_norm = normalize_team_name(league)
            row = conn.execute(
                """SELECT provider_team_id, confidence FROM soccer_team_aliases
                   WHERE source_normalized=? AND league_normalized IN (?, '')
                   ORDER BY CASE WHEN league_normalized=? THEN 0 ELSE 1 END,
                            confidence DESC LIMIT 1""",
                (normalize_team_name(home_name), league_norm, league_norm),
            ).fetchone()
            if row:
                home_id = str(row[0]); link_confidence = max(link_confidence, float(row[1] or 0.0))
        if not away_id and away_name:
            league_norm = normalize_team_name(league)
            row = conn.execute(
                """SELECT provider_team_id, confidence FROM soccer_team_aliases
                   WHERE source_normalized=? AND league_normalized IN (?, '')
                   ORDER BY CASE WHEN league_normalized=? THEN 0 ELSE 1 END,
                            confidence DESC LIMIT 1""",
                (normalize_team_name(away_name), league_norm, league_norm),
            ).fetchone()
            if row:
                away_id = str(row[0]); link_confidence = max(link_confidence, float(row[1] or 0.0))
        # A mesma identidade nos dois lados é sempre uma ligação corrompida;
        # usar sua forma duplicada seria um sinal artificial de equilíbrio.
        if not home_id or not away_id or home_id == away_id:
            home_id = away_id = ""
            link_confidence = 0.0
        home = _load_team_context(conn, home_id)
        away = _load_team_context(conn, away_id)

    features = {}
    features.update(_team_features("home", home))
    features.update(_team_features("away", away))
    features["context_sfi_link_confidence"] = float(link_confidence)
    features["context_sfi_available"] = float(bool(home and away))
    features["form_sfi_ppg_diff"] = (
        features["form_sfi_home_ppg"] - features["form_sfi_away_ppg"]
    )
    features["form_sfi_recent_diff"] = (
        features["form_sfi_home_recent_points_3"]
        - features["form_sfi_away_recent_points_3"]
    )
    features["form_sfi_position_diff"] = (
        features["form_sfi_home_position"] - features["form_sfi_away_position"]
    )
    return features
