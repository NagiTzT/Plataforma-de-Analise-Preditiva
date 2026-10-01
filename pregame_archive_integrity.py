"""Contrato de identidade e tempo para contexto pré-jogo arquivado."""

from datetime import datetime, timedelta, timezone
from typing import Any

from soccer_football_info import team_name_similarity


BRT = timezone(timedelta(hours=-3))


def archive_matches_training_fixture(
    training_home: Any,
    training_away: Any,
    training_datetime: Any,
    archive_home: Any,
    archive_away: Any,
    archive_start_timestamp: Any,
    max_delta_seconds: int = 12 * 3600,
) -> bool:
    """Aceita o arquivo somente para o mesmo confronto ordenado e kickoff."""
    if (team_name_similarity(training_home, archive_home) < 0.68
            or team_name_similarity(training_away, archive_away) < 0.68):
        return False
    try:
        training_dt = datetime.fromisoformat(str(training_datetime))
        if training_dt.tzinfo is None:
            training_dt = training_dt.replace(tzinfo=BRT)
        else:
            training_dt = training_dt.astimezone(BRT)
        archive_ts = int(float(archive_start_timestamp))
        if archive_ts <= 0:
            return False
        archive_dt = datetime.fromtimestamp(archive_ts, tz=BRT)
    except (TypeError, ValueError, OverflowError):
        return False
    return abs((archive_dt - training_dt).total_seconds()) <= int(max_delta_seconds)
