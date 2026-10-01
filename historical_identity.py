"""Vínculo conservador entre nomes do radar e a identidade do histórico local."""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Any

from soccer_football_info import (
    normalize_team_name, team_categories_compatible, team_name_similarity,
)


_STOPWORDS = {
    "fc", "cf", "sc", "ac", "afc", "fk", "sk", "club", "football",
    "futbol", "futebol", "de", "do", "da", "the", "calcio", "krc",
    "united", "city", "town", "athletic", "sporting", "real",
}
_CACHE_TTL_SECONDS = 3600
_cache_lock = threading.Lock()
_team_caches: dict[str, "_IdentityIndex"] = {}


@dataclass
class _IdentityIndex:
    built_at: float
    by_normalized: dict[str, list[tuple[str, int]]]
    by_token: dict[str, set[str]]
    frequencies: dict[str, int]


def _database_key(conn: Any) -> str:
    try:
        for _, name, path in conn.execute("PRAGMA database_list"):
            if name == "main":
                return str(path or id(conn))
    except Exception:
        pass
    return str(id(conn))


def _meaningful_tokens(value: Any) -> list[str]:
    return [
        token for token in normalize_team_name(value).split()
        if len(token) >= 4 and token not in _STOPWORDS
    ]


def _canonical_name_key(index: "_IdentityIndex", name: str) -> tuple[int, int, int, str]:
    """Prefere o alias frequente e tipograficamente limpo, de forma determinística."""
    collapsed = " ".join(str(name).split())
    whitespace_noise = len(str(name)) - len(collapsed)
    return (
        index.frequencies.get(name, 0),
        -whitespace_noise,
        -len(collapsed),
        collapsed.casefold(),
    )


def _build_team_index(conn: Any) -> _IdentityIndex:
    rows = conn.execute(
        """SELECT team_name, SUM(appearances) FROM (
               SELECT home_team AS team_name, COUNT(*) AS appearances
               FROM training_data WHERE home_team IS NOT NULL AND home_team!=''
               GROUP BY home_team
               UNION ALL
               SELECT away_team AS team_name, COUNT(*) AS appearances
               FROM training_data WHERE away_team IS NOT NULL AND away_team!=''
               GROUP BY away_team
           ) GROUP BY team_name"""
    ).fetchall()
    by_normalized: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    by_token: dict[str, set[str]] = collections.defaultdict(set)
    frequencies: dict[str, int] = {}
    for raw_name, appearances in rows:
        name = str(raw_name or "").strip()
        normalized = normalize_team_name(name)
        if not name or not normalized:
            continue
        frequency = int(appearances or 0)
        frequencies[name] = frequency
        by_normalized[normalized].append((name, frequency))
        for token in _meaningful_tokens(name):
            by_token[token].add(name)
    return _IdentityIndex(time.time(), dict(by_normalized), dict(by_token), frequencies)


def _team_index(conn: Any) -> _IdentityIndex:
    key = _database_key(conn)
    with _cache_lock:
        cached = _team_caches.get(key)
        if cached and time.time() - cached.built_at < _CACHE_TTL_SECONDS:
            return cached
        built = _build_team_index(conn)
        _team_caches[key] = built
        return built


def resolve_training_team_name(conn: Any, source_name: Any) -> tuple[str, float]:
    """Retorna identidade histórica somente quando o pareamento é inequívoco."""
    source = str(source_name or "").strip()
    normalized = normalize_team_name(source)
    if not normalized:
        return source, 0.0
    index = _team_index(conn)
    exact = index.by_normalized.get(normalized) or []
    if exact:
        chosen = max((name for name, _ in exact), key=lambda name: _canonical_name_key(index, name))
        return chosen, 1.0

    candidates: set[str] = set()
    for token in _meaningful_tokens(source):
        candidates.update(index.by_token.get(token, ()))
    if not candidates:
        return source, 0.0
    # Várias grafias que normalizam para a mesma identidade não são rivais.
    # Contá-las separadamente fazia o segundo score empatar com o primeiro e
    # rejeitava justamente aliases inequívocos.
    candidate_groups: dict[str, list[str]] = collections.defaultdict(list)
    for candidate in candidates:
        if team_categories_compatible(source, candidate):
            candidate_groups[normalize_team_name(candidate)].append(candidate)
    representatives = [
        max(group, key=lambda candidate: _canonical_name_key(index, candidate))
        for group in candidate_groups.values()
    ]
    scored = sorted(
        ((team_name_similarity(source, candidate), index.frequencies.get(candidate, 0), candidate)
         for candidate in representatives),
        reverse=True,
    )
    if not scored:
        return source, 0.0
    best_score, _, best_name = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.90 and best_score - second_score >= 0.03:
        return best_name, float(best_score)
    return source, 0.0


def training_team_aliases(conn: Any, source_name: Any) -> tuple[str, ...]:
    """Names that training canonicalizes through the same normalized identity."""
    canonical, _ = resolve_training_team_name(conn, source_name)
    normalized = normalize_team_name(canonical)
    index = _team_index(conn)
    aliases = {
        name for name, _ in (index.by_normalized.get(normalized) or [])
        if team_categories_compatible(canonical, name)
    }
    aliases.add(str(canonical))
    return tuple(sorted(name for name in aliases if name))


def clear_identity_cache() -> None:
    with _cache_lock:
        _team_caches.clear()
