"""Pure integrity helpers for the radar prediction list."""


def _prediction_identity(item):
    """Return a stable identity without collapsing distinct fixtures by label."""
    match_id = str(item.get("ID") or item.get("ID_Jogo") or "").strip()
    if match_id:
        return ("match_id", match_id)
    return (
        "fixture",
        str(item.get("Confronto") or "").strip().casefold(),
        str(item.get("Liga_Exata") or item.get("Liga") or "").strip().casefold(),
        int(item.get("Timestamp") or 0),
    )


def deduplicate_predictions(items):
    """Deduplicate retries of one event, keeping its strongest prediction.

    The old code keyed only by ``Confronto``. It consequently discarded a
    legitimate second match when the same clubs met twice inside the radar
    window (or when two categories shared a display label).
    """
    unique = {}
    order = []
    for item in items:
        key = _prediction_identity(item)
        if key not in unique:
            order.append(key)
            unique[key] = item
            continue
        try:
            incumbent = float(unique[key].get("Confiança", 0) or 0)
            candidate = float(item.get("Confiança", 0) or 0)
        except (TypeError, ValueError):
            incumbent = candidate = 0.0
        if candidate > incumbent:
            unique[key] = item
    return [unique[key] for key in order]
