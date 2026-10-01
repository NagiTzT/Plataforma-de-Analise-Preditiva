import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import Mock, patch

import pregame_metric_enrichment as collector
import sofascore_intelligence as sofa
from football_model_features import add_measured_sofa_duel


NOW = 2000000000
STATISTICS = {"statistics": [{"period": "ALL", "groups": [{"statisticsItems": [
    {"key": "expectedGoals", "homeValue": 1.5, "awayValue": 0},
    {"key": "shotsOnGoal", "homeValue": 4, "awayValue": 0},
    {"key": "bigChanceCreated", "homeValue": 1, "awayValue": 0},
]}]}]}


class RecentMetricsTests(unittest.TestCase):
    def setUp(self):
        handle, self.db = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        sofa.init_sofascore_db(self.db)

    def tearDown(self):
        sofa._initialized.discard(os.path.abspath(self.db))
        os.unlink(self.db)

    def history(self, mid="past", start=NOW-86400):
        return {"match_id": mid, "start_timestamp": start, "is_home": True,
                "team_id": "1", "team_name": "Alpha", "opponent_id": "2",
                "opponent_name": "Beta", "goals_for": 2, "goals_against": 0}

    def snapshot(self, history, provider="sofascore", side="home"):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("""INSERT OR REPLACE INTO pregame_recent_form_snapshots
                (match_id,side,provider,captured_at,cutoff_timestamp,games_json,features_json)
                VALUES ('target',?,?,?,? ,?,'{}')""",
                (side, provider, NOW-60, NOW+3600, json.dumps(history)))
            conn.commit()

    def run_collector(self, **kwargs):
        with patch.object(collector.time, "time", return_value=NOW):
            return collector.collect_recent_metric_profiles(
                self.db, [{"ID": "target", "Timestamp": NOW+3600}], **kwargs)

    @patch.object(collector, "_fetch_resource", return_value=(STATISTICS, False, 200))
    def test_excludes_current_live_future_and_reuses_profiles(self, fetch):
        self.snapshot([self.history("target", NOW-86400), self.history("live", NOW-3600),
                       self.history("future", NOW+86400), self.history()])
        summary = self.run_collector()
        self.assertEqual([c.args[1] for c in fetch.call_args_list], ["past"])
        self.assertEqual(summary["xg_matches"], 1)
        fetch.reset_mock()
        self.run_collector()
        fetch.assert_not_called()

    @patch.object(collector, "_fetch_resource", return_value=(STATISTICS, False, 200))
    def test_budget_and_deduplication(self, fetch):
        past = [self.history(str(i), NOW-86400*(i+1)) for i in range(5)]
        self.snapshot(past)
        self.snapshot(past, side="away")
        summary = self.run_collector(max_lookups=2)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(summary["deferred"], 3)
        self.assertEqual([c.args[1] for c in fetch.call_args_list], ["0", "1"])

    @patch.object(collector, "_fetch_resource", side_effect=OSError("network"))
    def test_failure_uses_existing_api_fallback(self, fetch):
        self.snapshot([self.history()])
        fallback = Mock(return_value={"resources": {"statistics": STATISTICS}, "http_requests": 1})
        result = self.run_collector(fallback=fallback)
        self.assertEqual(result["xg_matches"], 1)
        self.assertEqual(result["allsports_http"], 1)
        fallback.assert_called_once_with("past", ["statistics"])

    @patch.object(collector, "_fetch_resource", return_value=(None, False, 403))
    def test_provider_failure_does_not_stop_and_has_backoff(self, fetch):
        self.snapshot([self.history()])
        result = self.run_collector(fallback=Mock(side_effect=OSError("network")))
        self.assertEqual(result["unavailable"], 1)
        fetch.reset_mock()
        self.run_collector()
        fetch.assert_not_called()

    def test_complementary_fallback_keeps_first_source_xg(self):
        self.snapshot([self.history()])
        xg_only = {"statistics": [{"period": "ALL", "groups": [{"statisticsItems": [
            {"key": "expectedGoals", "homeValue": 2.7, "awayValue": 0},
        ]}]}]}
        with patch.object(collector, "_fetch_resource", return_value=(xg_only, False, 200)):
            result = self.run_collector(fallback=Mock(return_value={"resources": {"statistics": STATISTICS}}))
        self.assertEqual(result["xg_matches"], 1)
        with closing(sqlite3.connect(self.db)) as conn:
            xg, shots = conn.execute("SELECT xg_for,shots_on_target_for FROM sofascore_team_match_profiles WHERE team_name='Alpha'").fetchone()
        self.assertEqual(xg, 2.7)
        self.assertEqual(shots, 4)

    @patch.object(collector, "_fetch_resource")
    def test_unknown_namespace_never_uses_sofa_fixture_id(self, fetch):
        self.snapshot([self.history()], provider="soccerfootballinfo")
        self.run_collector()
        fetch.assert_not_called()

    def test_xg_duel_needs_both_measured_samples(self):
        features = {"sofa_roll_home_games": 8, "sofa_roll_away_games": 8}
        add_measured_sofa_duel(features)
        self.assertEqual(features["sofa_roll_duel_available"], 0)
        self.assertEqual(features["sofa_roll_duel_draw_signal"], 0)
        features.update(sofa_roll_home_xg_games=5, sofa_roll_away_xg_games=0,
                        sofa_roll_home_xg_for_avg=1.5)
        add_measured_sofa_duel(features)
        self.assertEqual(features["sofa_roll_duel_available"], 0)
        features["sofa_roll_away_xg_games"] = 3
        add_measured_sofa_duel(features)
        self.assertEqual(features["sofa_roll_duel_available"], 1)
        self.assertEqual(features["sofa_roll_duel_xg_reliability"], .6)


if __name__ == "__main__":
    unittest.main()
