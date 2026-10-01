"""Phase 4 capture/readiness dashboard. It deliberately never reads outcomes."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from phase4_research import challenger_definitions, champion_identity, preregistration_document
from phase4_metadata import country_from_league


def proportion_n(p, delta, alpha_z=1.9599639845, power_z=0.8416212336):
    """Approximate required N per group for a balanced two-proportion study."""
    return math.ceil(2 * p * (1 - p) * (alpha_z + power_z) ** 2 / delta ** 2)


def continuous_n(effect_size, alpha_z=1.9599639845, power_z=0.8416212336):
    return math.ceil(2 * (alpha_z + power_z) ** 2 / effect_size ** 2)


def load_sidecar(path):
    path = Path(path)
    if not path.exists():
        return [], {"status": "NOT_STARTED", "exists": False, "integrity_ok": None,
                    "immutable_triggers": 0, "records": 0, "runs": 0}
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "observations" not in tables:
            return [], {"status": "INVALID", "exists": True, "integrity_ok": False,
                        "immutable_triggers": 0, "records": 0, "runs": 0}
        rows = [dict(r) for r in conn.execute("SELECT * FROM observations ORDER BY observed_at")]
        triggers = conn.execute("""SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'
            AND name IN ('observations_no_update','observations_no_delete')""").fetchone()[0]
    valid = all(hashlib.sha256(r["payload"].encode()).hexdigest() == r["sha256"] for r in rows)
    for r in rows:
        try:
            r["data"] = json.loads(r["payload"])
            if r["stage"] == "phase4_outcome_v1":
                # Readiness needs association counts, never the label/class.
                r["data"] = {"eligibility_group": r["data"].get("eligibility_group")}
        except json.JSONDecodeError:
            r["data"] = None
            valid = False
    return rows, {"status": "HEALTHY" if valid and triggers == 2 else "INVALID",
                  "exists": True, "integrity_ok": valid, "immutable_triggers": triggers,
                  "records": len(rows), "runs": len({r["run_id"] for r in rows
                                                        if r["stage"] == "collection_started"})}


def summarize(db_path, sidecar_path):
    rows, health = load_sidecar(sidecar_path)
    pipeline_error_rows = [r for r in rows
                           if r["stage"] == "phase4_pipeline_error_v1" and r["data"]]
    invalidated_runs = {r["run_id"] for r in pipeline_error_rows}
    started_runs = {r["run_id"] for r in rows if r["stage"] == "collection_started"}
    finished_runs = {r["run_id"] for r in rows if r["stage"] == "collection_finished"}
    run_health_pass = {r["run_id"] for r in rows
                       if r["stage"] == "phase4_run_health_v1" and r["data"]
                       and r["data"].get("status") == "PASS"}
    shadow_health_pass = {r["run_id"] for r in rows
                          if r["stage"] == "phase4_shadow_health_v1" and r["data"]
                          and r["data"].get("status") == "PASS"}
    valid_runs = (started_runs & finished_runs & run_health_pass & shadow_health_pass) - invalidated_runs
    incomplete_runs = started_runs - valid_runs - invalidated_runs
    analysis_rows = [r for r in rows
                     if r["run_id"] == "phase4-registry" or r["run_id"] in valid_runs]
    prefilter_rows = [r for r in analysis_rows if r["stage"] == "phase4_prefilter_v1" and r["data"]]
    parent_by_key = {(r['run_id'],r['match_id']):r for r in prefilter_rows}
    metadata_repairs = 0
    for repair in analysis_rows:
        if repair['stage'] != 'phase4_metadata_repair_v1' or not repair['data']:
            continue
        parent = parent_by_key.get((repair['run_id'],repair['match_id']))
        data = repair['data']
        if (parent and not parent['data'].get('country')
                and data.get('parent_sha256') == parent['sha256']
                and data.get('static_metadata_only') is True
                and data.get('country') == country_from_league(parent['data'].get('league'))):
            parent['data'] = {**parent['data'],'country':data['country']}
            metadata_repairs += 1
    prefilter = [r["data"] for r in prefilter_rows]
    market_bundle_rows = [r for r in analysis_rows if r["stage"] == "market_bundle_v1" and r["data"]]
    market_bundles = [r["data"] for r in market_bundle_rows]
    prediction_rows = [r for r in analysis_rows
                       if r["stage"] in ("prediction", "prediction_phase4_prefilter_v1") and r["data"]]
    prefilter_prediction_rows = [r for r in prediction_rows
                                 if r["stage"] == "prediction_phase4_prefilter_v1"]
    predictions = [r["data"] for r in prediction_rows]
    outcome_rows = [r["data"] for r in analysis_rows if r["stage"] == "phase4_outcome_v1" and r["data"]]
    capture_issues = [r for r in analysis_rows if r["stage"] == "phase4_capture_issue_v1"]
    run_health = [r["data"] for r in analysis_rows if r["stage"] == "phase4_run_health_v1" and r["data"]]
    shadow_health = [r["data"] for r in analysis_rows if r["stage"] == "phase4_shadow_health_v1" and r["data"]]
    champion_shadow = [r for r in analysis_rows if r["stage"] == "shadow_phase4_prefilter_champion_v1"]
    market_shadow = [r for r in analysis_rows if r["stage"] == "shadow_phase4_prefilter_v1"]
    raw_markets = [r for r in analysis_rows if r["stage"] == "market"]
    collection_runs = set(valid_runs)
    groups = collections.Counter(x["eligibility_group"] for x in prefilter)
    league_rows = collections.defaultdict(lambda: collections.Counter())
    dimension_rows = collections.defaultdict(lambda: collections.Counter())
    strata_by_group = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.Counter()))
    for x in prefilter:
        group = x["eligibility_group"]
        group_family = "eligible" if group == "A_ELIGIBLE" else "non_eligible"
        league = str(x.get("league") or "UNKNOWN")
        league_rows[league][group] += 1
        for dimension in ("country", "season", "provider"):
            dimension_rows[dimension][str(x.get(dimension) or "UNKNOWN")] += 1
            strata_by_group[dimension][str(x.get(dimension) or "UNKNOWN")][group_family] += 1
        strata_by_group["league"][league][group_family] += 1
        strata_by_group["odds_bucket"][str(x.get("odds_bucket") or "UNKNOWN")][group_family] += 1
        try:
            event_dt = datetime.fromtimestamp(float(x.get("event_time")), tz=timezone.utc)
            weekday = event_dt.strftime("%A")
            hour_block = f"{(event_dt.hour // 6) * 6:02d}-{((event_dt.hour // 6) * 6 + 5):02d} UTC"
        except (TypeError, ValueError, OSError, OverflowError):
            weekday, hour_block = "UNKNOWN", "UNKNOWN"
        strata_by_group["weekday"][weekday][group_family] += 1
        strata_by_group["hour_block"][hour_block][group_family] += 1
    aux = collections.Counter()
    prefilter_keys = {(r["run_id"], r["match_id"]) for r in prefilter_rows}
    for row in market_bundle_rows:
        if (row["run_id"], row["match_id"]) not in prefilter_keys:
            continue
        bundle = row["data"]
        for family in bundle.get("available_market_families") or []:
            aux[family] += 1
    coverage = collections.Counter()
    original_coverage = collections.Counter()
    coverage_by_group = collections.defaultdict(collections.Counter)
    group_by_key = {(r["run_id"], r["match_id"]): r["data"]["eligibility_group"]
                    for r in prefilter_rows}
    for row in prefilter_prediction_rows:
        prediction = row["data"]
        families = ((prediction.get("data_coverage") or {}).get("families") or {})
        for family, available in families.items():
            if available is True:
                original_coverage[family] += 1
    family_by_key = {}
    free_shots_recovered = 0
    for row in prefilter_prediction_rows:
        key = (row['run_id'],row['match_id'])
        families = dict((row['data'].get('data_coverage') or {}).get('families') or {})
        f = row['data'].get('features') or {}
        # These counts were already frozen before kickoff; zeros of a measured
        # statistic are valid, but absence/NaN is not a measured observation.
        try:
            free_samples = [float(f.get(f'free_{side}_all_target_{direction}_games') or 0)
                            for side in ('home','away') for direction in ('for','against')]
            free_shots = all(math.isfinite(n) and n>=3 for n in free_samples)
        except (TypeError,ValueError):
            free_shots = False
        if free_shots and not families.get('shots_on_target'):
            families['shots_on_target'] = True
            free_shots_recovered += 1
        family_by_key[key] = families
    # Never let separately acquired covariates retroactively unlock the
    # preregistered study of frozen champion inputs.
    frozen_input_coverage = collections.Counter()
    for families in family_by_key.values():
        frozen_input_coverage.update(k for k,v in families.items() if v is True)
    covariates_valid = 0
    covariates_rejected = 0
    covariate_versions = collections.Counter()
    for row in analysis_rows:
        if row['stage'] != 'phase4_covariates_v2' or not row['data']:
            continue
        key = (row['run_id'],row['match_id']);parent=parent_by_key.get(key);data=row['data']
        captured=data.get('captured_at');kickoff=parent['data'].get('event_time') if parent else None
        valid = bool(parent and data.get('parent_sha256')==parent['sha256']
            and isinstance(captured,(int,float)) and isinstance(kickoff,(int,float))
            and parent['observed_at']<=captured<kickoff and row['observed_at']<kickoff
            and data.get('separate_from_champion_inputs') is True)
        evidence=data.get('evidence') or {};f=data.get('features') or {}
        families=(data.get('data_coverage') or {}).get('families') or {}
        if valid:
            try:
                for metric in ('xg','shots_on_target','big_chances'):
                    if families.get(metric) is not True:continue
                    for side in ('home','away'):
                        samples=evidence.get(f'{side}_{metric}') or []
                        ids={str(sample.get('match_id') or '') for sample in samples}
                        n=float(f.get(f'research_{side}_{metric}_games') or 0)
                        if (len(ids)<3 or '' in ids or row['match_id'] in ids
                                or not math.isfinite(n) or n<3
                                or any(not isinstance(sample.get('event_time'),(int,float))
                                       or sample['event_time']+3*3600>=captured for sample in samples)):
                            valid=False
            except (TypeError,ValueError,AttributeError):
                valid=False
        if not valid:
            covariates_rejected+=1;continue
        covariates_valid+=1;covariate_versions[str(data.get('version'))]+=1
        union=family_by_key.setdefault(key,{})
        for metric,available in families.items():
            if available is True:union[metric]=True
    for families in family_by_key.values():
        coverage.update(k for k,v in families.items() if v is True)
    for row in prefilter_prediction_rows:
        group = group_by_key.get((row["run_id"], row["match_id"]), "UNKNOWN")
        families = family_by_key.get((row['run_id'],row['match_id']), {})
        for family, available in families.items():
            coverage_by_group[group][family + ("_observed" if available is True else "_missing")] += 1
    bundle_by_key = {(r["run_id"], r["match_id"]): r["data"] for r in market_bundle_rows}
    prediction_by_key = {(r["run_id"], r["match_id"]): r["data"]
                         for r in prefilter_prediction_rows}
    league_detail = collections.defaultdict(collections.Counter)
    for row in prefilter_rows:
        item = row["data"]
        league = str(item.get("league") or "UNKNOWN")
        group = item["eligibility_group"]
        prefix = "eligible" if group == "A_ELIGIBLE" else "non_eligible"
        league_detail[league][prefix + "_n"] += 1
        league_detail[league]["market_1x2_valid"] += 1
        bundle = bundle_by_key.get((row["run_id"], row["match_id"]), {})
        if bundle.get("available_market_families"):
            league_detail[league]["aux_market_available"] += 1
        prediction = prediction_by_key.get((row["run_id"], row["match_id"]), {})
        families = family_by_key.get((row['run_id'],row['match_id']), {})
        league_detail[league]["xg_available"] += int(families.get("xg") is True)
        league_detail[league]["shots_on_target_available"] += int(families.get("shots_on_target") is True)
        league_detail[league]["big_chances_available"] += int(families.get("big_chances") is True)
    version, artifact_hash, cutoff = champion_identity(db_path)
    registry = challenger_definitions(version, artifact_hash, cutoff)
    plan = preregistration_document()
    trigger = plan["evaluation_trigger"]
    power = {
        "accuracy_baseline_39pct_n_each_group": {
            "plus_2pp": proportion_n(.39, .02),
            "plus_3pp": proportion_n(.39, .03),
            "plus_5pp": proportion_n(.39, .05),
        },
        "draw_rate_baseline_263pct_n_each_group": {
            "plus_2pp": proportion_n(.263, .02),
            "plus_3pp": proportion_n(.263, .03),
            "plus_5pp": proportion_n(.263, .05),
        },
        "standardized_entropy_difference_n_each_group": {
            "small_d_0_2": continuous_n(.2),
            "moderate_d_0_3": continuous_n(.3),
            "large_d_0_5": continuous_n(.5),
        },
        "alpha": .05, "power": .80,
        "interpretation": "Approximate balanced independent-group targets; final inference uses time blocks."
    }
    eligible = groups["A_ELIGIBLE"]
    noneligible = sum(v for k, v in groups.items() if k.startswith("B"))
    resolved_eligible = sum(x.get("eligibility_group") == "A_ELIGIBLE" for x in outcome_rows)
    resolved_noneligible = sum(str(x.get("eligibility_group", "")).startswith("B")
                               for x in outcome_rows)
    radars = len(collection_runs)
    target_families = ("team_form", "elo_rating", "home_away", "league_profile",
                       "xg", "shots_on_target", "big_chances")
    family_min = min((frozen_input_coverage.get(name, 0) for name in target_families), default=0)
    leagues_with_both = sum(
        counts.get("A_ELIGIBLE", 0) >= 20
        and sum(v for k, v in counts.items() if k.startswith("B")) >= 20
        for counts in league_rows.values()
    )
    def represented_values(dimension, group_name, minimum=1):
        return sum(counts.get(group_name, 0) >= minimum
                   for value,counts in strata_by_group[dimension].items() if value!='UNKNOWN')

    largest_league_share = {}
    for group_name, denominator in (("eligible", eligible), ("non_eligible", noneligible)):
        largest = max((counts.get(group_name, 0)
                       for counts in strata_by_group["league"].values()), default=0)
        largest_league_share[group_name] = (largest / denominator if denominator else None)
    representativeness_checks = {
        "five_balanced_leagues": leagues_with_both >= trigger["minimum_leagues_with_20_each_group"],
        "no_single_league_majority_eligible": bool(eligible) and largest_league_share["eligible"] <= .50,
        "no_single_league_majority_non_eligible": bool(noneligible) and largest_league_share["non_eligible"] <= .50,
        "five_countries_each_group": all(
            represented_values("country", group_name) >= 5
            for group_name in ("eligible", "non_eligible")),
        "four_weekdays_each_group": all(
            represented_values("weekday", group_name) >= 4
            for group_name in ("eligible", "non_eligible")),
        "three_time_blocks_each_group": all(
            represented_values("hour_block", group_name) >= 3
            for group_name in ("eligible", "non_eligible")),
        "known_provider_each_group": all(
            sum(counts.get(group_name, 0) for value, counts
                in strata_by_group["provider"].items() if value != "UNKNOWN")
            == denominator and denominator > 0
            for group_name, denominator in (("eligible", eligible), ("non_eligible", noneligible))),
    }
    representativeness_pass = all(representativeness_checks.values())
    family_status = {}
    for family in target_families:
        observed = int(coverage.get(family, 0))
        total = len(prefilter_prediction_rows)
        family_status[family] = {
            "observed": observed, "valid": observed,
            "missing": max(0, total - observed), "error": None,
            "coverage_percent": (100.0 * observed / total if total else 0.0),
            "target_n": trigger["minimum_family_observations"],
            "progress_percent": min(100.0, 100.0 * observed /
                                    trigger["minimum_family_observations"]),
        }
    for family in ("over_under", "btts", "asian_handicap", "double_chance", "draw_no_bet"):
        observed = int(aux.get(family, 0))
        total = len(prefilter)
        family_status[family] = {
            "observed": observed, "valid": observed,
            "missing": max(0, total - observed), "error": None,
            "coverage_percent": (100.0 * observed / total if total else 0.0),
            "target_n": trigger["minimum_family_observations"],
            "progress_percent": min(100.0, 100.0 * observed /
                                    trigger["minimum_family_observations"]),
        }
    schema_by_run = collections.defaultdict(collections.Counter)
    run_last_seen = {}
    for row in market_bundle_rows:
        schema_hash = str(row["data"].get("provider_schema_hash") or "UNKNOWN")
        schema_by_run[row["run_id"]][schema_hash] += 1
        run_last_seen[row["run_id"]] = max(run_last_seen.get(row["run_id"], 0), row["observed_at"])
    ordered_schema_runs = sorted(schema_by_run, key=lambda run: run_last_seen.get(run, 0))
    dominant_schema = {
        run: schema_by_run[run].most_common(1)[0][0] for run in ordered_schema_runs
        if schema_by_run[run]
    }
    provider_drift = (len(ordered_schema_runs) >= 2 and
                      dominant_schema.get(ordered_schema_runs[-1]) !=
                      dominant_schema.get(ordered_schema_runs[-2]))
    readiness = {
        "minimum_radars": {"current": radars, "target": trigger["minimum_radars"]},
        "valid_eligible": {"current": eligible, "target": trigger["minimum_valid_market_each_group"]},
        "valid_non_eligible": {"current": noneligible, "target": trigger["minimum_valid_market_each_group"]},
        "resolved_eligible": {"current": resolved_eligible, "target": trigger["minimum_resolved_eligible"],
                              "reason": "count only; outcome values remain blinded"},
        "resolved_non_eligible": {"current": resolved_noneligible, "target": trigger["minimum_resolved_non_eligible"],
                                  "reason": "count only; outcome values remain blinded"},
        "family_observations": {"current_min": family_min,
                                "target": trigger["minimum_family_observations"],
                                "source": "original frozen champion inputs only; separate covariates cannot unlock this holdout"},
        "leagues_with_20_each_group": {"current": leagues_with_both,
                                        "target": trigger["minimum_leagues_with_20_each_group"]},
        "representativeness_pass": {"current": int(representativeness_pass), "target": 1},
    }
    ready = all(
        item.get("current", item.get("current_min", 0)) >= item["target"]
        for item in readiness.values()
    )
    ready = bool(ready and health['integrity_ok'] and health['immutable_triggers']==2
        and not incomplete_runs and not provider_drift
        and all(x.get('available_before_decision') for x in prefilter))
    result = {
        "decision": "READY_TO_EVALUATE" if ready else "INSUFFICIENT_DATA",
        "outcomes_opened": False,
        "capture_health": health,
        "blind_collection_status": {
            "phase4_status": "COLLECTING" if collection_runs else "NOT_STARTED",
            "radars_completed": len(collection_runs),
            "raw_markets": len(raw_markets), "eligibility_rows": len(prefilter),
            "champion_shadow_rows": len(champion_shadow),
            "market_1x2_shadow_rows": len(market_shadow),
            "capture_issue_rows": len(capture_issues),
            "invalidated_runs": len(invalidated_runs),
            "invalidated_run_ids": sorted(invalidated_runs),
            "incomplete_runs": len(incomplete_runs),
            "incomplete_run_ids": sorted(incomplete_runs),
            "invalid_pregame_rows": sum(not bool(x.get("available_before_decision")) for x in prefilter),
            "run_health_alerts": [alert for item in (run_health + shadow_health)
                                  for alert in item.get("alerts", [])],
            "evaluation_locked": not ready,
        },
        "coverage_dashboard": {
            "prefilter_valid_1x2": len(prefilter), "prediction_snapshots": len(predictions),
            "prefilter_champion_predictions": len(prefilter_prediction_rows),
            "family_available_counts": dict(coverage), "auxiliary_market_counts": dict(aux),
            "original_champion_family_counts":dict(original_coverage),
            "corrected_frozen_input_family_counts":dict(frozen_input_coverage),
            "free_shots_recovered_from_frozen_inputs":free_shots_recovered,
            "separate_covariate_snapshots":covariates_valid,
            "rejected_covariate_snapshots":covariates_rejected,
            "covariate_versions":dict(covariate_versions),
            "separate_covariates_unlock_original_holdout":False,
            "family_status": family_status,
            "by_eligibility": {k: dict(v) for k, v in coverage_by_group.items()},
        },
        "coverage_by_league": [
            {"league": league, **dict(counts)}
            for league, counts in sorted(league_detail.items(),
                key=lambda item: -(item[1].get("eligible_n",0)+item[1].get("non_eligible_n",0)))
        ],
        "coverage_by_country_season_provider": {
            k: dict(v) for k, v in dimension_rows.items()
        },
        "representativeness": {
            "status": "PASS" if representativeness_pass else "INSUFFICIENT",
            "checks": representativeness_checks,
            "largest_league_share": largest_league_share,
            "strata": {
                dimension: {value: dict(counts) for value, counts in values.items()}
                for dimension, values in strata_by_group.items()
            },
            "uses_labels": False,
            "country_repairs_from_original_league":metadata_repairs,
        },
        "eligibility_dataset": {"groups": dict(groups), "n": len(prefilter),
                                "labels_opened": False},
        "odds_filter_structure": {
            "rule": "home_odd > 1.99 AND away_odd > 1.99",
            "eligible": eligible, "non_eligible": noneligible,
            "statistical_comparison": "BLOCKED_UNTIL_TRIGGER",
        },
        "pool_difficulty_readiness": {
            "available_now": ["market_entropy", "market_draw_probability", "top1_probability",
                              "top1_top2_gap", "home_away_gap", "overround", "eligibility",
                              "same_champion_prefilter_scoring_in_both_groups"],
            "waiting": ["resolved outcomes", "expected_goals_market", "asian_handicap_balance",
                        "btts_probability"],
            "score_active": False,
        },
        "challenger_registry": registry,
        "power_plan": power,
        "market_readiness": {"valid_1x2": len(prefilter), "ready": len(prefilter) >= 1500},
        "auxiliary_market_readiness": {"counts": dict(aux), "ready": bool(aux) and min(aux.values()) >= 500},
        "provider_health": {
            "providers": dict(collections.Counter(x.get("provider") or "UNKNOWN" for x in prefilter)),
            "schema_hashes": dict(collections.Counter(x.get("provider_schema_hash") or "UNKNOWN"
                                                       for x in market_bundles)),
            "capture_errors": len(capture_issues),
            "dominant_schema_by_run": dominant_schema,
            "provider_drift": provider_drift,
            "status": "PROVIDER_DRIFT" if provider_drift else "PASS",
        },
        "draw_readiness": {"valid_market_draw": len(prefilter),
                           "expected_goals": aux.get("over_under", 0),
                           "asian_handicap": aux.get("asian_handicap", 0),
                           "status": "INSUFFICIENT_SAMPLE"},
        "attack_defense_readiness": {"prediction_snapshots": len(predictions),
                                     "xg_available": coverage.get("xg", 0),
                                     "shots_on_target_available": coverage.get("shots_on_target", 0),
                                     "big_chances_available": coverage.get("big_chances", 0),
                                     "status": "INSUFFICIENT_SAMPLE"},
        "holdout_protocol": plan,
        "data_collection_health": {
            "status":"COVERAGE_BLOCKED" if family_min < trigger['minimum_family_observations'] else "PASS",
            "undercovered_frozen_families":[name for name in target_families
                if frozen_input_coverage.get(name,0)<trigger['minimum_family_observations']],
            "new_covariates_require_versioned_future_holdout":True,
        },
        "evaluation_trigger": {"checks": readiness, "ready": ready,
                               "representativeness_checked": representativeness_pass},
    }
    return result


def pct(current, target):
    return f"{(100 * current / target if target else 0):.1f}%"


def markdown(data):
    h = data["capture_health"]
    blind = data["blind_collection_status"]
    groups = data["eligibility_dataset"]["groups"]
    reg = data["challenger_registry"]
    power = data["power_plan"]
    checks = data["evaluation_trigger"]["checks"]
    lines = [
        "# Fase 4 — Primeira entrega: captura e pré-registro", "",
        f"PHASE4_STATUS = **{blind['phase4_status']}**. Decisão atual: **{data['decision']}**.", "",
        "Resultados esportivos não foram abertos. Este relatório mede somente infraestrutura, cobertura e prontidão.", "",
        "## A — Capture health", "",
        f"Status: **{h['status']}**. Sidecar existente: {h['exists']}. Registros: {h['records']}. "
        f"Radares: {h['runs']}. Integridade SHA-256: {h['integrity_ok']}. Triggers imutáveis: {h['immutable_triggers']}/2.", "",
        f"RAW_MARKETS={blind['raw_markets']}; ELIGIBILITY_ROWS={blind['eligibility_rows']}; "
        f"CHAMPION_SHADOW_ROWS={blind['champion_shadow_rows']}; MARKET_1X2_ROWS={blind['market_1x2_shadow_rows']}; "
        f"CAPTURE_ISSUES={blind['capture_issue_rows']}; INVALID_PREGAME={blind['invalid_pregame_rows']}; "
        f"INVALIDATED_RUNS={blind['invalidated_runs']}; INCOMPLETE_RUNS={blind['incomplete_runs']}; "
        f"EVALUATION_LOCKED={blind['evaluation_locked']}.", "",
        f"Alertas cegos: `{json.dumps(blind['run_health_alerts'], ensure_ascii=False)}`.", "",
        f"Provider health: **{data['provider_health']['status']}**; "
        f"erros de captura={data['provider_health']['capture_errors']}; "
        f"drift de schema={data['provider_health']['provider_drift']}.", "",
        "## B — Coverage dashboard", "",
        f"1X2 pré-filtro válido: {data['coverage_dashboard']['prefilter_valid_1x2']}. "
        f"Snapshots de previsão: {data['coverage_dashboard']['prediction_snapshots']}.", "",
        f"Famílias disponíveis: `{json.dumps(data['coverage_dashboard']['family_available_counts'], ensure_ascii=False)}`.", "",
        f"Coverage por eligibility: `{json.dumps(data['coverage_dashboard']['by_eligibility'], ensure_ascii=False)}`.", "",
        f"Cobertura original do campeão: `{json.dumps(data['coverage_dashboard']['original_champion_family_counts'], ensure_ascii=False)}`. "
        f"Covariáveis de pesquisa capturadas separadamente antes do jogo: {data['coverage_dashboard']['separate_covariate_snapshots']}; "
        "as previsões e os atributos originais do campeão permanecem congelados.", "",
        "## C — Coverage by league", "",
    ]
    league_rows=data["coverage_by_league"][:20]
    if league_rows:
        lines += ["| Liga | Eligible | Non-eligible | 1X2 | Aux | xG | SOT | Big chances |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for item in league_rows:
            lines.append(f"| {item['league']} | {item.get('eligible_n',0)} | {item.get('non_eligible_n',0)} | "
                         f"{item.get('market_1x2_valid',0)} | {item.get('aux_market_available',0)} | "
                         f"{item.get('xg_available',0)} | {item.get('shots_on_target_available',0)} | "
                         f"{item.get('big_chances_available',0)} |")
        lines.append("")
    else:
        lines += ["Ainda não existem observações prospectivas. A tabela será preenchida pelo primeiro radar.", ""]
    lines += [
        "### Representatividade sem labels", "",
        f"Status: **{data['representativeness']['status']}**. "
        f"Checks: `{json.dumps(data['representativeness']['checks'], ensure_ascii=False)}`.",
        "A distribuição é acompanhada por liga, país, dia, faixa horária, provider, odds e eligibility; resultados esportivos não são usados.", "",
    ]
    lines += [
        "## D — Eligibility dataset", "",
        f"A={groups.get('A_ELIGIBLE',0)}, B1={groups.get('B1_HOME_LOW',0)}, "
        f"B2={groups.get('B2_AWAY_LOW',0)}, B3={groups.get('B3_BOTH_LOW',0)}. Labels permanecem fechados.", "",
        "## E — Odds filter structure", "",
        "A regra foi congelada exatamente como `home_odd > 1.99 AND away_odd > 1.99`. "
        "Cada mercado válido passa a ser registrado antes do filtro com entropy, probabilidades de-vig, gaps, overround, bucket e distância do limiar. Nenhuma comparação foi executada sem amostra.", "",
        "## F — Pool difficulty readiness", "",
        "Disponível desde o primeiro radar: entropy, P_DRAW de mercado, top1, top2, gap top1-top2, gap casa-fora, overround, odds age e eligibility. "
        "O mesmo artefato campeão será pontuado no instante pré-filtro nos dois grupos, sem HTTP e com coverage registrada; a previsão operacional enriquecida continua separada. "
        "Aguardando O/U, AH, BTTS e resultados resolvidos. Não existe score com pesos manuais.", "",
        "## G — Challenger registry", "",
        "| Challenger | Versão | Status | Produz previsão agora |", "|---|---|---|---|",
    ]
    for item in reg:
        lines.append(f"| {item['challenger_id']} | {item['version']} | {item['status']} | "
                     f"{'sim' if item['challenger_id'] in ('CHAMPION','MARKET_1X2') else 'não'} |")
    lines += ["", "Somente CHAMPION e MARKET_1X2 possuem definição executável. Os demais ficam DRAFT; ausência é MISSING, nunca zero.", "",
              "## H — Power plan", "",
              "| Comparação | +2 p.p. | +3 p.p. | +5 p.p. |", "|---|---:|---:|---:|",
              f"| Accuracy, baseline 39%, N por grupo | {power['accuracy_baseline_39pct_n_each_group']['plus_2pp']} | {power['accuracy_baseline_39pct_n_each_group']['plus_3pp']} | {power['accuracy_baseline_39pct_n_each_group']['plus_5pp']} |",
              f"| Draw rate, baseline 26,3%, N por grupo | {power['draw_rate_baseline_263pct_n_each_group']['plus_2pp']} | {power['draw_rate_baseline_263pct_n_each_group']['plus_3pp']} | {power['draw_rate_baseline_263pct_n_each_group']['plus_5pp']} |",
              "", "Plano primário: 1.500 jogos resolvidos em cada grupo detectam aproximadamente 5 p.p. com 80% de poder; efeitos menores exigirão coleta maior.", "",
              "## I — Market readiness", "",
              f"1X2 válido: {data['market_readiness']['valid_1x2']}/1500 por grupo. Status: INSUFFICIENT SAMPLE.", "",
              "## J — Auxiliary market readiness", "",
              f"Contagens: `{json.dumps(data['auxiliary_market_readiness']['counts'], ensure_ascii=False)}`. Nenhum mercado ausente foi inventado.", "",
              "## K — Draw readiness", "",
              "D0 e D1 estão pré-definidos; D2–D11 aguardam coverage prospectiva. Não existe regra manual de empate.", "",
              "## L — Attack/Defense readiness", "",
              f"xG={data['attack_defense_readiness']['xg_available']}, shots on target={data['attack_defense_readiness']['shots_on_target_available']}, "
              f"big chances={data['attack_defense_readiness']['big_chances_available']}. Status: INSUFFICIENT SAMPLE.", "",
              "## M — Holdout protocol", "",
              "Versões congeladas; sem tuning, threshold, seleção de features ou inspeção de resultados intermediários. Mudança cria nova versão e novo holdout. "
              "Monitoramento permitido somente para saúde, volume, coverage e integridade.", "",
              "## N — Evaluation trigger", "",
              "| Requisito | Atual | Alvo | Progresso |", "|---|---:|---:|---:|" ]
    for name, item in checks.items():
        current = item.get("current", item.get("current_min", 0))
        lines.append(f"| {name} | {current} | {item['target']} | {pct(current,item['target'])} |")
    lines += ["", "A avaliação permanece bloqueada até todos os mínimos, resolução de labels e representatividade. "
              "A decisão científica desta primeira entrega é **INSUFFICIENT DATA**.", ""]
    return "\n".join(lines)


def run(db, sidecar, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = summarize(db, sidecar)
    (output / "phase4_readiness.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    report = markdown(data)
    (output / "FASE_4_PRIMEIRA_ENTREGA_A_N.md").write_text(report, encoding="utf-8")
    (output / "FASE_4B_STATUS_CEGO.md").write_text(report, encoding="utf-8")
    (output / "phase4_preregistration_v1.json").write_text(
        json.dumps(preregistration_document(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="ia_sports_v5.db")
    parser.add_argument("--sidecar", default="phase2_observations.db")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.db, args.sidecar, args.output_dir)
