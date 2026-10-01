import math
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date
from pathlib import Path

from phase4_readiness import continuous_n, proportion_n, summarize
from phase4_research import (
    capture_eligibility_candidate,
    eligibility_group,
    finalize_shadow_health,
    market_metrics,
    odds_bucket,
)
from phase4_labels import ingest_blind_outcomes
from phase2_observations import append


class _FakeLabelClient:
    def __init__(self):
        self.http_requests = 0

    def fetch_day(self, utc_day, force_refresh=False):
        self.http_requests += 1
        return [{
            "id": "source-1", "status": "ENDED",
            "teamA": {"score": {"f": 2}}, "teamB": {"score": {"f": 1}},
        }], {"complete": True, "pages_missing": 0}


class Phase4ResearchTests(unittest.TestCase):
    def test_strict_eligibility_boundaries(self):
        self.assertEqual(eligibility_group(2.0, 2.0), "A_ELIGIBLE")
        self.assertEqual(eligibility_group(1.99, 2.0), "B1_HOME_LOW")
        self.assertEqual(eligibility_group(2.0, 1.99), "B2_AWAY_LOW")
        self.assertEqual(eligibility_group(1.99, 1.99), "B3_BOTH_LOW")
        self.assertEqual(eligibility_group(1.0, 2.0), "INVALID")

    def test_market_metrics_are_devigged_and_missing_is_not_zero(self):
        metrics = market_metrics([2.5, 3.2, 2.8], 100, 200)
        self.assertAlmostEqual(sum(metrics[k] for k in (
            "p_market_home", "p_market_draw", "p_market_away")), 1.0)
        self.assertGreater(metrics["market_entropy"], 0)
        self.assertGreater(metrics["top1_market_probability"], 0)
        self.assertGreater(metrics["top2_market_probability"], 0)
        self.assertIsNone(metrics["expected_total_goals"])
        self.assertIsNone(metrics["asian_handicap_balance"])
        self.assertIsNone(metrics["btts_probability"])

    def test_prefilter_capture_is_append_only_and_research_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "main.db"
            result = capture_eligibility_candidate(
                db, "run", "match", [2.0, 3.0, 2.1], 200, 100,
                provider="test", home_team="A", away_team="B", league="L")
            self.assertEqual(result["group"], "A_ELIGIBLE")
            self.assertTrue(result["saved"])
            append(db, "run", "", "collection_started", {})
            append(db, "run", "", "collection_finished", {})
            append(db, "run", "", "phase4_run_health_v1", {"status": "PASS"})
            append(db, "run", "", "phase4_shadow_health_v1", {"status": "PASS"})
            sidecar = db.with_name("phase2_observations.db")
            data = summarize(db, sidecar)
            self.assertEqual(data["eligibility_dataset"]["groups"]["A_ELIGIBLE"], 1)
            self.assertFalse(data["outcomes_opened"])

    def test_buckets_and_power_are_predefined(self):
        self.assertEqual(odds_bucket(1.39, 3.0), "LT_1_40")
        self.assertEqual(odds_bucket(2.6, 2.7), "GE_2_50")
        self.assertGreater(proportion_n(.39, .02), proportion_n(.39, .05))
        self.assertEqual(continuous_n(.5), 63)

    def test_blind_label_ingestion_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "main.db"
            capture_eligibility_candidate(
                db, "run", "match", [2.0, 3.0, 2.1], 200, 100,
                provider="test", home_team="A", away_team="B", league="L",
                source_event_id="source-1")
            first = ingest_blind_outcomes(
                db, client=_FakeLabelClient(), now=20_000,
                grace_seconds=0, max_days=2)
            self.assertEqual(first["resolved_new"], 1)
            self.assertNotIn("actual_outcome", first)
            second = ingest_blind_outcomes(
                db, client=_FakeLabelClient(), now=20_000,
                grace_seconds=0, max_days=2)
            self.assertEqual(second["resolved_new"], 0)
            sidecar = db.with_name("phase2_observations.db")
            with closing(sqlite3.connect(sidecar)) as conn:
                row = conn.execute(
                    "SELECT payload FROM observations WHERE stage='phase4_outcome_v1'"
                ).fetchone()
            self.assertEqual(json.loads(row[0])["actual_outcome"], "HOME")

    def test_shadow_health_reports_only_infrastructure(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "main.db"
            health = finalize_shadow_health(
                db, "run", candidates=10, scored=9, failed=1)
            self.assertEqual(health["status"], "ALERT")
            self.assertIn("SHADOW_FAILURE_RATE_GT_5_PERCENT", health["alerts"])
            self.assertNotIn("accuracy", health)

    def test_invalidated_run_is_excluded_from_prospective_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "main.db"
            for run in ("failed", "valid"):
                capture_eligibility_candidate(
                    db, run, "match-" + run, [2.0, 3.0, 2.1], 200, 100,
                    provider="test", home_team="A", away_team="B", league="L",
                    source_event_id="source-" + run)
                append(db, run, "", "collection_started", {})
            append(db, "valid", "", "collection_finished", {})
            append(db, "valid", "", "phase4_run_health_v1", {"status": "PASS"})
            append(db, "valid", "", "phase4_shadow_health_v1", {"status": "PASS"})
            append(db, "failed", "", "phase4_pipeline_error_v1", {
                "run_completed": False, "error_type": "RegressionTest"})
            data = summarize(db, db.with_name("phase2_observations.db"))
            self.assertEqual(data["blind_collection_status"]["radars_completed"], 1)
            self.assertEqual(data["blind_collection_status"]["invalidated_runs"], 1)
            self.assertEqual(data["eligibility_dataset"]["groups"]["A_ELIGIBLE"], 1)


if __name__ == "__main__":
    unittest.main()
