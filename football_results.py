"""Conservative regulation-time results shared by training, radar and audit."""
import math
import unicodedata

OUTCOMES = ("MANDANTE", "EMPATE", "VISITANTE")


def _goal(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return int(value) if math.isfinite(value) and value >= 0 and value.is_integer() else None
    except (ValueError, TypeError):
        return None


def event_finished(event):
    if not isinstance(event, dict):
        return False
    status = event.get("status") or {}
    if isinstance(status, str):
        return status.lower() in {"finished", "afterextra", "afterpenalties", "ended"}
    return isinstance(status, dict) and str(status.get("type", "")).lower() in {"finished", "afterextra", "afterpenalties", "ended"}


def regulation_score(event, require_finished=True):
    """Return (home, away) or None; never invent zeros/use shootout totals."""
    if not isinstance(event, dict) or (require_finished and not event_finished(event)):
        return None
    scores = [event.get(key) or {} for key in ("homeScore", "awayScore")]
    if not all(isinstance(score, dict) for score in scores):
        return None
    for key in ("normaltime", "normalTime"):
        pair = tuple(_goal(score.get(key)) for score in scores)
        if None not in pair:
            return pair
    # Sofa/AllSports period1/period2 contain goals within each regular period.
    periods = [[_goal(score.get(key)) for key in ("period1", "period2")] for score in scores]
    if all(None not in side for side in periods):
        return tuple(sum(side) for side in periods)
    status = event.get("status") or {}
    text = str(status).lower()
    has_extra = (any(term in text for term in ("extra", "penalt", "shootout"))
                 or (isinstance(status, dict) and status.get("code") in {110, 120})
                 or any(any(key in score and score[key] is not None
                            for key in ("overtime", "extra1", "extra2", "penalties")) for score in scores))
    if has_extra:
        return None
    for key in ("current", "display"):
        pair = tuple(_goal(score.get(key)) for score in scores)
        if None not in pair:
            return pair
    return None


def score_outcome(score):
    if score is None:
        return None
    home, away = score
    return "MANDANTE" if home > away else "VISITANTE" if away > home else "EMPATE"


def incident_goal_score(incident):
    """Verified in-regulation goal score for the existing early-payout check."""
    if not isinstance(incident, dict) or incident.get('incidentType') != 'goal':
        return None
    minute = _goal(incident.get('time'))
    pair = tuple(_goal(incident.get(k)) for k in ('homeScore','awayScore'))
    if minute is None or minute > 90 or None in pair:
        return None
    return pair


def resolve_pick_side(snapshot_side, pick, home_name="", away_name=""):
    side = str(snapshot_side or "").upper()
    if side in OUTCOMES:
        return side
    def canonical(value):
        return " ".join(''.join(ch for ch in unicodedata.normalize("NFKD", str(value or ""))
                               if not unicodedata.combining(ch)).casefold().split())
    name = canonical(pick)
    if name in {"empate", "draw", "x"}:
        return "EMPATE"
    if name in {"mandante", "casa", "home", "1"}:
        return "MANDANTE"
    if name in {"visitante", "fora", "away", "2"}:
        return "VISITANTE"
    matches = [side for side, team in (("MANDANTE",home_name),("VISITANTE",away_name))
               if name and canonical(team) and name == canonical(team)]
    return matches[0] if len(matches) == 1 else None


def settlement_status(score, side):
    if score is None or side not in OUTCOMES:
        return None
    return "GREEN ✅" if score_outcome(score) == side else "RED ❌"
