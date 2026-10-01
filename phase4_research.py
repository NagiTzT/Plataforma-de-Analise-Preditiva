"""Frozen Phase 4 research instrumentation.

This module never changes an operational pick, odds rule, ticket or publication.
It records the complete valid-market pool before the strict 1.99 filter and
freezes pre-registered shadow definitions in the append-only sidecar.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from phase2_observations import append, market_probabilities


PHASE4_VERSION = "phase4-preregistered-v1"
HOME_ODD_THRESHOLD = 1.99
AWAY_ODD_THRESHOLD = 1.99


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def eligibility_group(home_odd, away_odd):
    """Return the pre-registered strict eligibility group."""
    home = _finite(home_odd)
    away = _finite(away_odd)
    if home is None or away is None or home <= 1 or away <= 1:
        return "INVALID"
    home_low = home <= HOME_ODD_THRESHOLD
    away_low = away <= AWAY_ODD_THRESHOLD
    if not home_low and not away_low:
        return "A_ELIGIBLE"
    if home_low and not away_low:
        return "B1_HOME_LOW"
    if away_low and not home_low:
        return "B2_AWAY_LOW"
    return "B3_BOTH_LOW"


def odds_bucket(home_odd, away_odd):
    minimum = min(float(home_odd), float(away_odd))
    bounds = (
        (1.40, "LT_1_40"), (1.60, "1_40_TO_1_60"),
        (1.80, "1_60_TO_1_80"), (1.99, "1_80_TO_1_99"),
        (2.20, "1_99_TO_2_20"), (2.50, "2_20_TO_2_50"),
    )
    return next((name for upper, name in bounds if minimum < upper), "GE_2_50")


def eligible_distance_bucket(home_odd, away_odd):
    minimum = min(float(home_odd), float(away_odd))
    if minimum <= HOME_ODD_THRESHOLD:
        return None
    if minimum < 2.10:
        return "1_99_TO_2_10"
    if minimum < 2.30:
        return "2_10_TO_2_30"
    if minimum < 2.60:
        return "2_30_TO_2_60"
    return "GE_2_60"


def market_metrics(odds, observed_at, kickoff, suspended=False):
    """Compute only metrics supported by the genuinely observed 1X2 quote."""
    benchmark = market_probabilities(odds, observed_at, kickoff, suspended)
    if benchmark is None:
        return None
    probabilities = [float(x) for x in benchmark["probabilities"]]
    ordered = sorted(probabilities, reverse=True)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "p_market_home": probabilities[0],
        "p_market_draw": probabilities[1],
        "p_market_away": probabilities[2],
        "market_entropy": entropy,
        "market_entropy_normalized": entropy / math.log(3),
        "home_away_market_gap": abs(probabilities[0] - probabilities[2]),
        "top1_market_probability": ordered[0],
        "top2_market_probability": ordered[1],
        "top1_top2_market_gap": ordered[0] - ordered[1],
        "overround": float(benchmark["overround"]),
        "expected_total_goals": None,
        "asian_handicap_balance": None,
        "btts_probability": None,
        "missing_reasons": {
            "expected_total_goals": "auxiliary_over_under_not_observed",
            "asian_handicap_balance": "asian_handicap_not_observed",
            "btts_probability": "btts_not_observed",
        },
    }


def capture_eligibility_candidate(
    db_path, run_id, match_id, odds, kickoff, observed_at, *, provider,
    home_team, away_team, league, country=None, season=None, source_event_id=None,
    suspended=False, metadata_provenance=None,
):
    """Freeze a valid-market candidate before applying the production rule."""
    metrics = market_metrics(odds, observed_at, kickoff, suspended)
    if metrics is None:
        reason = ("invalid_for_pregame_evaluation" if kickoff and observed_at >= float(kickoff)
                  else "invalid_or_incomplete_1x2")
        saved = append(db_path, run_id, str(match_id), "phase4_capture_issue_v1", {
            "phase4_version": PHASE4_VERSION, "provider": provider,
            "observed_at": _finite(observed_at), "event_time": _finite(kickoff),
            "issue": reason, "severity": "ERROR", "affects_production": False,
        })
        return {"group": "INVALID", "saved": saved, "health": "INVALID", "reason": reason}
    group = eligibility_group(odds[0], odds[2])
    payload = {
        "phase4_version": PHASE4_VERSION,
        "provider": provider,
        "observed_at": float(observed_at),
        "event_time": _finite(kickoff),
        "event_key": f"{provider}:{match_id}",
        "available_before_decision": bool(kickoff and observed_at < float(kickoff)),
        "odds_age_seconds": float(kickoff) - float(observed_at),
        "vendor_quote_timestamp": None,
        "data_version": "as-observed-current-quote-v1",
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "country": country,
        "metadata_provenance": metadata_provenance,
        "season": season,
        "source_event_id": source_event_id,
        "home_odd": float(odds[0]),
        "draw_odd": float(odds[1]),
        "away_odd": float(odds[2]),
        "eligibility_group": group,
        "production_eligible": group == "A_ELIGIBLE",
        "production_rule": "home_odd > 1.99 AND away_odd > 1.99",
        "minimum_side_odd": min(float(odds[0]), float(odds[2])),
        "odd_distance_from_1_99": min(float(odds[0]), float(odds[2])) - 1.99,
        "odds_bucket": odds_bucket(odds[0], odds[2]),
        "eligible_distance_bucket": eligible_distance_bucket(odds[0], odds[2]),
        "market": metrics,
        "pool_difficulty_score": None,
        "pool_difficulty_reason": "weights_not_estimated_before_holdout",
        "affects_production": False,
    }
    probability_sum = metrics["p_market_home"] + metrics["p_market_draw"] + metrics["p_market_away"]
    payload["health_checks"] = {
        "kickoff_valid": bool(kickoff), "observed_before_kickoff": observed_at < float(kickoff),
        "odds_gt_one": all(float(x) > 1 for x in odds),
        "probabilities_finite": all(math.isfinite(float(x)) for x in (
            metrics["p_market_home"], metrics["p_market_draw"], metrics["p_market_away"])),
        "probability_sum_valid": abs(probability_sum - 1.0) <= 1e-9,
        "entropy_valid": 0 <= metrics["market_entropy"] <= math.log(3) + 1e-9,
        "overround_plausible": -0.05 <= metrics["overround"] <= 0.50,
        "eligibility_consistent": group == eligibility_group(odds[0], odds[2]),
    }
    payload["health_status"] = ("PASS" if all(payload["health_checks"].values()) else "WARN")
    saved = append(db_path, run_id, str(match_id), "phase4_prefilter_v1", payload)
    shadow_saved = capture_prefilter_shadow(db_path, run_id, match_id, metrics, group, kickoff)
    return {"group": group, "saved": bool(saved and shadow_saved),
            "health": payload["health_status"], "reason": None}


def challenger_definitions(champion_version=None, champion_hash=None, training_cutoff=None):
    primary = ["multiclass_log_loss", "multiclass_brier", "accuracy"]
    secondary = ["draw_ap", "draw_auc", "class_recall", "calibration"]
    common = {
        "created_at": "2026-09-18T00:00:00-03:00",
        "evaluation_protocol": PHASE4_VERSION,
        "pick_policy": "argmax",
        "quality_policy": "research_only_no_gate",
        "primary_metrics": primary,
        "secondary_metrics": secondary,
        "n_target": 1500,
    }
    return [
        {**common, "challenger_id": "CHAMPION", "version": str(champion_version or "runtime"),
         "status": "FROZEN", "algorithm": "active_operational_artifact",
         "training_cutoff": training_cutoff, "artifact_hash": champion_hash,
         "features": ["active_champion_feature_order"], "market_features": [],
         "draw_features": ["active_champion_p_draw"], "data_requirements": ["champion_features"],
         "hypotheses": ["same frozen champion can be compared inside and outside eligibility"]},
        {**common, "challenger_id": "MARKET_1X2", "version": "market-proportional-v1",
         "status": "FROZEN", "algorithm": "proportional_overround_removal",
         "training_cutoff": None, "artifact_hash": hashlib.sha256(b"market-proportional-v1").hexdigest(),
         "features": ["home_odd", "draw_odd", "away_odd"],
         "market_features": ["p_market_home", "p_market_draw", "p_market_away"],
         "draw_features": ["p_market_draw", "market_entropy"],
         "data_requirements": ["valid_timestamped_pregame_1x2"],
         "hypotheses": ["market benchmark predicts both pools", "eligible pool is structurally harder"]},
        {**common, "challenger_id": "MARKET_FULL", "version": "market-full-design-v1",
         "status": "DRAFT", "algorithm": None, "training_cutoff": None, "artifact_hash": None,
         "features": ["1x2", "over_under", "btts", "asian_handicap", "double_chance", "dnb"],
         "market_features": ["auxiliary_markets"], "draw_features": ["under", "handicap_balance"],
         "data_requirements": ["prospective_auxiliary_market_coverage"],
         "hypotheses": ["auxiliary markets add information beyond 1x2"]},
        {**common, "challenger_id": "STRENGTH", "version": "strength-design-v1",
         "status": "DRAFT", "algorithm": None, "training_cutoff": None, "artifact_hash": None,
         "features": ["temporal_strength", "home_away_strength"], "market_features": [],
         "draw_features": ["strength_gap"], "data_requirements": ["prospective_strength_snapshot"],
         "hypotheses": ["temporal strength adds value prospectively"]},
        {**common, "challenger_id": "ATTACK_DEFENSE", "version": "attack-defense-design-v1",
         "status": "DRAFT", "algorithm": None, "training_cutoff": None, "artifact_hash": None,
         "features": ["attack", "defense", "home_attack", "away_attack", "home_defense", "away_defense"],
         "market_features": [], "draw_features": ["low_goal_matchup"],
         "data_requirements": ["sufficient_prospective_attack_defense_coverage"],
         "hypotheses": ["opponent-adjusted matchup adds value"]},
        {**common, "challenger_id": "DRAW", "version": "draw-design-v1", "status": "DRAFT",
         "algorithm": None, "training_cutoff": None, "artifact_hash": None,
         "features": ["market_draw", "entropy", "expected_goals", "strength_gap", "loss_resistance"],
         "market_features": ["1x2", "over_under", "asian_handicap"],
         "draw_features": ["pre_registered_draw_tournament_D0_D11"],
         "data_requirements": ["prospective_draw_family_coverage"],
         "hypotheses": ["new information discriminates draws"]},
        {**common, "challenger_id": "FULL", "version": "full-design-v1", "status": "DRAFT",
         "algorithm": None, "training_cutoff": None, "artifact_hash": None,
         "features": ["only_families_approved_by_prospective_signal_tournament"],
         "market_features": ["approved_only"], "draw_features": ["approved_only"],
         "data_requirements": ["at_least_one_approved_family"],
         "hypotheses": ["approved families combine without losing temporal stability"]},
    ]


def champion_identity(db_path):
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("""SELECT modelo_blob, scaler_params, feature_order,
                data_treinamento, model_version FROM modelos_ml WHERE liga='GLOBAL'""").fetchone()
        if not row:
            return None, None, None
        digest = hashlib.sha256(
            bytes(row[0]) + str(row[1]).encode() + str(row[2]).encode()
        ).hexdigest()
        return row[4], digest, row[3]
    except (sqlite3.Error, TypeError, ValueError):
        return None, None, None


def ensure_challenger_registry(db_path):
    version, digest, cutoff = champion_identity(db_path)
    definitions = challenger_definitions(version, digest, cutoff)
    for item in definitions:
        fingerprint = item.get("artifact_hash") or item["version"]
        append(db_path, "phase4-registry", f"{item['challenger_id']}:{fingerprint}",
               "challenger_registered", item)
    return definitions


def capture_prefilter_shadow(db_path, run_id, match_id, market, group, kickoff):
    candidates = [
        {"challenger_id": "MARKET_1X2", "version": "market-proportional-v1",
         "available": True,
         "probabilities": [market["p_market_home"], market["p_market_draw"], market["p_market_away"]],
         "feature_coverage": {"valid_pregame_1x2": True}},
        {"challenger_id": "CHAMPION", "version": None, "available": False,
         "probabilities": None, "missing_reason": "not_scored_at_prefilter_collection"},
        {"challenger_id": "MARKET_FULL", "version": "market-full-design-v1", "available": False,
         "probabilities": None, "missing_reason": "auxiliary_model_not_registered"},
        {"challenger_id": "STRENGTH", "version": "strength-design-v1", "available": False,
         "probabilities": None, "missing_reason": "prospective_model_not_registered"},
        {"challenger_id": "ATTACK_DEFENSE", "version": "attack-defense-design-v1", "available": False,
         "probabilities": None, "missing_reason": "prospective_model_not_registered"},
        {"challenger_id": "DRAW", "version": "draw-design-v1", "available": False,
         "probabilities": None, "missing_reason": "prospective_model_not_registered"},
        {"challenger_id": "FULL", "version": "full-design-v1", "available": False,
         "probabilities": None, "missing_reason": "no_approved_signal_family"},
    ]
    return append(db_path, run_id, str(match_id), "shadow_phase4_prefilter_v1", {
        "phase4_version": PHASE4_VERSION, "frozen_at": time.time(), "kickoff": kickoff,
        "eligibility_group": group, "candidates": candidates,
        "affects_production": False, "outcomes_opened": False,
    })


def capture_prediction_shadow(db_path, run_id, match_id, champion_probabilities, odds,
                              kickoff, model_version, feature_coverage=None, *,
                              stage="shadow_phase4_prediction_v1",
                              eligibility=None):
    now = time.time()
    market = market_probabilities(odds, now, kickoff)
    candidates = [
        {"challenger_id": "CHAMPION", "version": str(model_version), "available": True,
         "probabilities": list(champion_probabilities),
         "feature_coverage": feature_coverage, "p_pick_correct_estimated": None,
         "draw_risk": float(champion_probabilities[1])},
        {"challenger_id": "MARKET_1X2", "version": "market-proportional-v1",
         "available": bool(market), "probabilities": market["probabilities"] if market else None,
         "feature_coverage": {"valid_pregame_1x2": bool(market)},
         "missing_reason": None if market else "invalid_or_not_pregame"},
    ]
    for cid, version, reason in (
        ("MARKET_FULL", "market-full-design-v1", "auxiliary_model_not_registered"),
        ("STRENGTH", "strength-design-v1", "prospective_model_not_registered"),
        ("ATTACK_DEFENSE", "attack-defense-design-v1", "prospective_model_not_registered"),
        ("DRAW", "draw-design-v1", "prospective_model_not_registered"),
        ("FULL", "full-design-v1", "no_approved_signal_family"),
    ):
        candidates.append({"challenger_id": cid, "version": version, "available": False,
                           "probabilities": None, "missing_reason": reason})
    return append(db_path, run_id, str(match_id), stage, {
        "phase4_version": PHASE4_VERSION, "frozen_at": now, "kickoff": kickoff,
        "eligibility_group": eligibility,
        "candidates": candidates, "affects_production": False,
        "holdout_protocol": "trigger-based; minimum 20 radars and sample targets; no tuning",
    })


def finalize_run_health(db_path, run_id, counters, *, total_events, operational_candidates,
                        started_at, provider="allsports"):
    """Freeze an outcome-blind infrastructure health event for one radar."""
    valid = int(counters.get("phase4_eligible", 0)) + int(counters.get("phase4_non_eligible", 0))
    failures = int(counters.get("phase4_capture_failures", 0))
    warnings = int(counters.get("phase4_health_warnings", 0))
    denominator = max(1, valid + failures)
    alerts = []
    if total_events and valid == 0:
        alerts.append("NO_VALID_PREFILTER_MARKETS")
    if failures / denominator > .05:
        alerts.append("CAPTURE_FAILURE_RATE_GT_5_PERCENT")
    if warnings / denominator > .05:
        alerts.append("HEALTH_WARNING_RATE_GT_5_PERCENT")
    if operational_candidates > int(counters.get("phase4_eligible", 0)):
        alerts.append("OPERATIONAL_POOL_EXCEEDS_ELIGIBLE_POOL")
    sidecar = Path(db_path).resolve().with_name("phase2_observations.db")
    triggers = 0
    hash_valid = None
    duplicates = None
    try:
        with closing(sqlite3.connect(sidecar, timeout=2.0)) as conn:
            triggers = int(conn.execute("""SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'
                AND name IN ('observations_no_update','observations_no_delete')""").fetchone()[0])
            frozen = conn.execute("SELECT sha256,payload FROM observations WHERE run_id=?",
                                  (str(run_id),)).fetchall()
            hash_valid = all(hashlib.sha256(raw.encode()).hexdigest() == digest
                             for digest, raw in frozen)
            duplicates = int(conn.execute("""SELECT COUNT(*) FROM (
                SELECT run_id,match_id,stage,COUNT(*) n FROM observations
                WHERE run_id=? GROUP BY run_id,match_id,stage HAVING n>1)""",
                (str(run_id),)).fetchone()[0])
    except sqlite3.Error:
        alerts.append("SIDECAR_HEALTH_QUERY_FAILED")
    if triggers != 2:
        alerts.append("IMMUTABILITY_TRIGGERS_MISSING")
    if hash_valid is False:
        alerts.append("HASH_VALIDATION_FAILED")
    if duplicates:
        alerts.append("ACCIDENTAL_DUPLICATES")
    payload = {
        "phase4_version": PHASE4_VERSION, "pipeline_version": "phase4-capture-v1",
        "provider": provider, "started_at": float(started_at), "finished_at": time.time(),
        "latency_seconds": max(0.0, time.time() - float(started_at)),
        "total_events": int(total_events), "valid_markets": valid,
        "eligible": int(counters.get("phase4_eligible", 0)),
        "non_eligible": int(counters.get("phase4_non_eligible", 0)),
        "operational_candidates": int(operational_candidates),
        "capture_failures": failures, "health_warnings": warnings,
        "capture_failure_rate": failures / denominator,
        "immutable_triggers": triggers, "hash_valid": hash_valid,
        "duplicate_keys": duplicates, "alerts": alerts,
        "status": "PASS" if not alerts else "ALERT",
        "performance_metrics_opened": False,
    }
    append(db_path, run_id, "", "phase4_run_health_v1", payload)
    return payload


def finalize_shadow_health(db_path, run_id, *, candidates, scored, failed):
    """Record blind completeness of champion shadow scoring after collection."""
    candidates, scored, failed = int(candidates), int(scored), int(failed)
    alerts = []
    if candidates and scored == 0:
        alerts.append("NO_CHAMPION_SHADOW_ROWS")
    if scored + failed < candidates:
        alerts.append("SHADOW_ACCOUNTING_INCOMPLETE")
    if candidates and failed / candidates > .05:
        alerts.append("SHADOW_FAILURE_RATE_GT_5_PERCENT")
    payload = {
        "phase4_version": PHASE4_VERSION, "pipeline_version": "phase4-capture-v1",
        "candidates": candidates, "scored": scored, "failed": failed,
        "completion_rate": (scored / candidates if candidates else 1.0),
        "alerts": alerts, "status": "PASS" if not alerts else "ALERT",
        "performance_metrics_opened": False,
    }
    append(db_path, run_id, "", "phase4_shadow_health_v1", payload)
    return payload


def preregistration_document():
    return {
        "phase4_version": PHASE4_VERSION,
        "production_rule_frozen": "home_odd > 1.99 AND away_odd > 1.99",
        "production_changes": False,
        "evaluation_blinding": "do not inspect outcomes before trigger",
        "primary_hypotheses": [
            "eligible market entropy is higher than non-eligible",
            "eligible draw rate is higher than non-eligible",
            "champion accuracy is lower in eligible than non-eligible",
            "market accuracy is lower in eligible than non-eligible",
            "champion and market log loss are worse in eligible",
        ],
        "primary_metrics": ["multiclass_log_loss", "multiclass_brier", "accuracy"],
        "secondary_metrics": ["draw_ap", "draw_auc", "calibration", "class_recall"],
        "evaluation_trigger": {
            "minimum_radars": 20,
            "minimum_resolved_eligible": 1500,
            "minimum_resolved_non_eligible": 1500,
            "minimum_valid_market_each_group": 1500,
            "minimum_family_observations": 500,
            "minimum_leagues_with_20_each_group": 5,
            "representativeness_required": True,
        },
        "multiple_testing": "effect consistency + plausibility + temporal repetition; no single p-value promotion",
        "quality_reopen_condition": "prospective improvement in log loss/brier, discrimination, or material draw AP",
    }


def write_preregistration(path):
    payload = preregistration_document()
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(raw, encoding="utf-8")
    return hashlib.sha256(raw.encode()).hexdigest()
