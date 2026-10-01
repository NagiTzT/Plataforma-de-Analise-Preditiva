"""Classificacao deterministica do contexto de uma competicao.

As flags sao derivadas apenas do nome conhecido antes do jogo. Assim podem ser
usadas tanto no radar quanto na reconstrucao cronologica do treino sem leakage.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalizar(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


_CUP_WORDS = {
    "cup", "copa", "taca", "pokal", "coppa", "coupe", "trophy",
    "shield", "beker", "kubok", "kupa", "pokalen",
}
_KNOCKOUT_WORDS = {
    "playoff", "playoffs", "play off", "play offs", "knockout", "knock out",
    "mata mata", "eliminatoria", "eliminatorias", "umspil",
}
_KNOCKOUT_ROUNDS = {
    "final", "finals", "semifinal", "semifinals", "semi final", "semi finals",
    "quarterfinal", "quarterfinals", "quarter final", "quarter finals",
    "quartas", "oitavas", "round of 16", "round of 32", "round of 64",
}
_GROUP_WORDS = {
    "group", "groups", "grupo", "grupos", "league phase", "fase de liga",
    "round robin", "fase de grupos", "group stage",
}
_AMBIGUOUS_FINAL_PHASES = {"final stage", "final phase", "final round", "fase final"}
_FRIENDLY_WORDS = {
    "friendly", "friendlies", "amistoso", "amistosos", "amistosa",
    "amistosas", "club friendly", "international friendly",
}


def _has_phrase(text: str, phrases: set[str]) -> bool:
    """Match whole normalized words, not 'final' inside unrelated words."""
    return any(f" {phrase} " in f" {text} " for phrase in phrases)


def _explicit_knockout(text: str) -> bool:
    # "Final stage" can be a second round-robin phase. Only the final MATCH
    # or a specifically named elimination round establishes this flag.
    for phrase in _AMBIGUOUS_FINAL_PHASES:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _has_phrase(text, _KNOCKOUT_WORDS | _KNOCKOUT_ROUNDS)


def competition_flags(league_name: Any, stage_name: Any = None) -> dict[str, float]:
    """Classify the name and optional explicitly provided pregame stage.

    A generic league name does not reveal its current phase. Callers may pass
    verified tournament/round metadata as ``stage_name``; neither date, teams
    nor the eventual result are used to guess that phase. In particular, the
    KSÍ calls its Lengjudeild knockout promotion competition ``Umspil`` (rules
    23.1.8.2). Promotion/relegation alone need not mean knockout: those stages
    can also be round-robin groups.
    """
    text = _normalizar(league_name)
    stage = _normalizar(stage_name)
    if stage_name is not None:
        text = (text + " " + stage).strip()
    tokens = set(text.split())
    is_cup = any(word in tokens for word in _CUP_WORDS) or "challenge cup" in text
    # An explicit current round supersedes stale group wording in the generic
    # tournament label. A numeric round without a phase adds no evidence.
    phase_text = stage if stage and (
        _explicit_knockout(stage) or _has_phrase(stage, _GROUP_WORDS)
    ) else text
    group_phase = _has_phrase(phase_text, _GROUP_WORDS)
    explicit_knockout = _explicit_knockout(phase_text)
    # A named group phase is not an elimination round even when a provider
    # calls it a "promotion play-off group" or "final group".
    explicit_knockout = explicit_knockout and not group_phase
    is_friendly = any(word in text for word in _FRIENDLY_WORDS)
    is_knockout = explicit_knockout or (is_cup and not group_phase)
    is_two_leg = any(word in text for word in (
        "2nd leg", "second leg", "return leg", "segunda mao", "jogo de volta",
    ))
    is_qualifier = any(word in text for word in (
        "qualifier", "qualification", "qualifying", "preliminary", "qualificacao",
    ))
    is_youth_or_reserve = bool(re.search(
        r"(?:^| )(?:u|sub) ?(?:17|18|19|20|21|23)(?: |$)|"
        r"(?:^| )(?:reserve|reserves|youth|academy|junior|primavera)(?: |$)|"
        r"(?:^| )(?:b team|team b)(?: |$)", text,
    )) or any(word in text for word in (
        "mls next pro", "next pro league", "development league",
    ))
    is_lower_tier = any(word in text for word in (
        "kolmonen", "4 deild", "third division", "division 3",
        "serie d", "regional league", "non league", "amateur",
    ))
    is_women = bool(re.search(
        r"(?:^| )(?:women|woman|womens|feminino|feminina|femenino|femenina|frauen)(?: |$)",
        text,
    )) or bool(re.search(r"\(w\)", str(league_name or "").lower()))

    volatility = 0.08
    volatility += 0.18 * float(is_cup)
    volatility += 0.12 * float(is_knockout)
    volatility += 0.10 * float(is_qualifier)
    volatility += 0.16 * float(is_youth_or_reserve)
    volatility += 0.10 * float(is_lower_tier)
    volatility += 0.05 * float(is_women)
    volatility += 0.22 * float(is_friendly)
    # Família primária mutuamente exclusiva. Ela permite calibrar as
    # probabilidades sem criar um modelo frágil para cada liga pequena.
    family_friendly = is_friendly
    family_qualifier = is_qualifier and not family_friendly
    family_knockout = is_knockout and not family_friendly and not family_qualifier
    family_cup_group = is_cup and group_phase and not any((
        family_friendly, family_qualifier, family_knockout,
    ))
    family_league = not any((family_friendly, family_qualifier,
                             family_knockout, family_cup_group))
    return {
        "is_cup": float(is_cup),
        "is_knockout": float(is_knockout),
        "is_volta": float(is_two_leg),
        "is_qualifier": float(is_qualifier),
        "is_youth_or_reserve": float(is_youth_or_reserve),
        "is_lower_tier": float(is_lower_tier),
        "is_women": float(is_women),
        "is_friendly": float(is_friendly),
        "competition_family_league": float(family_league),
        "competition_family_cup_group": float(family_cup_group),
        "competition_family_knockout": float(family_knockout),
        "competition_family_qualifier": float(family_qualifier),
        "competition_family_friendly": float(family_friendly),
        "competition_volatility": min(1.0, volatility),
    }


def event_competition_flags(event: Any, league_name: Any, captured_at: int,
                            expected_start: int) -> dict[str, float]:
    """Preserve explicit phase from an already fetched, verified pregame event.

    No extra request; no season-end table, score or current postgame cache can
    be used to reconstruct the phase of an old frozen prediction.
    """
    if not isinstance(event, dict) or expected_start <= captured_at:
        return {}
    try:
        start = int(event.get('startTimestamp') or 0)
    except (TypeError, ValueError):
        return {}
    status = event.get('status') or {}
    if not isinstance(status, dict):
        return {}
    if start != expected_start or status.get('type') in ('finished','inprogress'):
        return {}
    tournament = event.get('tournament') or {}
    round_info = event.get('roundInfo') or {}
    if not isinstance(tournament, dict) or not isinstance(round_info, dict):
        return {}
    tournament_name = str(tournament.get('name') or '')
    round_name = str(round_info.get('name') or '')
    # A generic league + numeric round does not establish a phase.
    combined = _normalizar(tournament_name + ' ' + round_name)
    if not (_explicit_knockout(combined) or _has_phrase(combined, _GROUP_WORDS)):
        return {}
    flags = competition_flags(str(league_name or '') + ' ' + tournament_name,
                              stage_name=round_name or tournament_name)
    flags['context_competition_phase_available'] = 1.
    return flags


__all__ = ["competition_flags", "event_competition_flags"]
