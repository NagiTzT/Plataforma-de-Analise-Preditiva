"""Cross-provider fixture identity checks used before accepting match data."""

from __future__ import annotations

import math
from typing import Any

from soccer_football_info import team_name_similarity


def safe_event_timestamp(value: Any) -> int:
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number > 0 else 0
    except (TypeError, ValueError, OverflowError):
        return 0


def event_matches_prediction(
    event: dict[str, Any] | None,
    match_id: Any,
    confrontation: str,
    expected_start: Any,
    name_floor: float = 0.68,
    time_tolerance_seconds: int = 12 * 3600,
) -> bool:
    """Require ID, both ordered teams and kickoff to describe one fixture."""
    if not isinstance(event, dict):
        return False
    event_id = event.get("id")
    if event_id is not None and str(event_id) != str(match_id):
        return False
    pieces = str(confrontation or "").split(" vs ", 1)
    if len(pieces) != 2 or not all(piece.strip() for piece in pieces):
        return False
    expected_home, expected_away = (piece.strip() for piece in pieces)
    actual_home = str((event.get("homeTeam") or {}).get("name") or "")
    actual_away = str((event.get("awayTeam") or {}).get("name") or "")
    if min(
        team_name_similarity(expected_home, actual_home),
        team_name_similarity(expected_away, actual_away),
    ) < float(name_floor):
        return False
    expected_ts = safe_event_timestamp(expected_start)
    actual_ts = safe_event_timestamp(event.get("startTimestamp"))
    if expected_ts and actual_ts and abs(expected_ts - actual_ts) > int(time_tolerance_seconds):
        return False
    return True
