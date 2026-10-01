"""Regression checks for absent statistics versus measured zero values."""

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing

import sofascore_intelligence as sofa


class PregameMetricCoverageTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        sofa.init_sofascore_db(self.path)

    def tearDown(self):
        sofa._initialized.discard(os.path.abspath(self.path))
        os.unlink(self.path)

    def event(self, timestamp=100):
        return {
            "startTimestamp": timestamp,
            "homeTeam": {"id": 1, "name": "Alpha"},
            "awayTeam": {"id": 2, "name": "Beta"},
            "homeScore": {"normaltime": 1},
            "awayScore": {"normaltime": 0},
        }

    def metrics(self, items):
        return sofa._statistics_metrics({"statistics": [{
            "period": "ALL", "groups": [{"statisticsItems": items}],
        }]})[0]

    def save(self, identifier, metrics, timestamp=100):
        sofa._save_team_profiles(self.path, identifier, self.event(timestamp), metrics, 0.6)

    def roll(self, cutoff=1000):
        return sofa._rolling_team_features(self.path, "home", "Alpha", cutoff)

    def test_legacy_missing_zeros_do_not_dilute_one_measured_game(self):
        with closing(sqlite3.connect(self.path)) as conn:
            for index in range(7):
                conn.execute(
                    """INSERT INTO sofascore_team_match_profiles
                       (match_id,team_key,team_name,start_timestamp,is_home,
                        goals_for,goals_against,xg_for,xg_against,dominance,
                        result_points,is_draw,captured_at)
                       VALUES (?, 'alpha', 'Alpha', ?,1,1,0,0,0,0.5,3,0,1)""",
                    (str(index), index + 1),
                )
            conn.commit()
        self.save("measured", self.metrics([
            {"key": "expectedGoals", "homeValue": 2.0, "awayValue": 0.5},
        ]))
        features = self.roll()
        self.assertEqual(features["sofa_roll_home_games"], 8)
        self.assertEqual(features["sofa_roll_home_xg_games"], 1)
        self.assertEqual(features["sofa_roll_home_xg_for_avg"], 2.0)
        self.assertEqual(features["sofa_roll_home_xg_against_avg"], 0.5)
        self.assertEqual(features["sofa_roll_home_dominance_games"], 1)
        self.assertEqual(features["sofa_roll_home_legacy_xg_for_avg"], .25)
        self.assertAlmostEqual(features["sofa_roll_home_legacy_dominance_avg"], .5125)

        # Exercise the actual changed vector produced by the database rollup,
        # not just adding coverage fields to otherwise identical inputs.
        both = {**features, "sofa_roll_available": 1}
        both.update({key.replace("sofa_roll_home_", "sofa_roll_away_"): value
                     for key, value in features.items() if key.startswith("sofa_roll_home_")})
        legacy = dict(both)
        for side in ("home", "away"):
            for metric in ("xg_for", "xg_against", "dominance"):
                legacy[f"sofa_roll_{side}_{metric}_avg"] = both[f"sofa_roll_{side}_legacy_{metric}_avg"]
        legacy = {key: value for key, value in legacy.items() if "_legacy_" not in key}
        base = [.38, .29, .33]
        self.assertEqual(
            sofa.analyze_pregame_context(base, legacy, "active-v5")["probabilities"],
            sofa.analyze_pregame_context(base, both, "active-v5")["probabilities"],
        )

    def test_genuine_zero_is_preserved_when_provider_reports_metric(self):
        self.save("zero", self.metrics([
            {"key": "expectedGoals", "homeValue": 0.0, "awayValue": 0.0},
            {"key": "bigChanceCreated", "homeValue": 0, "awayValue": 0},
        ]))
        features = self.roll()
        self.assertEqual(features["sofa_roll_home_xg_games"], 1)
        self.assertEqual(features["sofa_roll_home_xg_for_avg"], 0)
        self.assertEqual(features["sofa_roll_home_big_chances_for_games"], 1)

    def test_later_incomplete_source_does_not_erase_measured_profile(self):
        self.save("same", self.metrics([
            {"key": "expectedGoals", "homeValue": 2.1, "awayValue": 0},
        ]))
        self.save("same", self.metrics([
            {"key": "shotsOnGoal", "homeValue": 7, "awayValue": 0},
        ]))
        self.save("same", {})
        features = self.roll()
        self.assertEqual(features["sofa_roll_home_xg_for_avg"], 2.1)
        self.assertEqual(features["sofa_roll_home_xg_against_avg"], 0)
        self.assertEqual(features["sofa_roll_home_xg_games"], 1)
        self.assertEqual(features["sofa_roll_home_shots_on_target_for_avg"], 7)

    def test_shots_coverage_does_not_claim_xg_coverage(self):
        self.save("shots", self.metrics([
            {"key": "shotsOnGoal", "homeValue": 5, "awayValue": 0},
        ]))
        features = self.roll()
        self.assertEqual(features["sofa_roll_home_xg_games"], 0)
        self.assertEqual(features["sofa_roll_home_shots_on_target_for_games"], 1)
        self.assertEqual(features["sofa_roll_home_shots_on_target_against_games"], 1)
        with closing(sqlite3.connect(self.path)) as conn:
            value = conn.execute(
                "SELECT xg_for FROM sofascore_team_match_profiles WHERE team_name='Alpha'"
            ).fetchone()[0]
        self.assertIsNone(value)

    def test_null_dash_and_nan_are_not_measured_zero(self):
        for value in (None, "-", "NaN", "", float("inf")):
            metrics = self.metrics([
                {"key": "expectedGoals", "homeValue": value, "awayValue": 0},
            ])
            self.assertEqual(metrics["home_xg_available"], 0)
            self.assertEqual(metrics["away_xg_available"], 1)

    def test_metric_averages_have_independent_denominators_and_temporal_cutoff(self):
        self.save("xg", self.metrics([
            {"key": "expectedGoals", "homeValue": 2, "awayValue": 1},
        ]), 100)
        self.save("shots", self.metrics([
            {"key": "shotsOnGoal", "homeValue": 6, "awayValue": 2},
        ]), 200)
        self.save("future", self.metrics([
            {"key": "expectedGoals", "homeValue": 8, "awayValue": 7},
        ]), 400)
        features = self.roll(cutoff=300)
        self.assertEqual(features["sofa_roll_home_games"], 2)
        self.assertEqual(features["sofa_roll_home_xg_for_avg"], 2)
        self.assertEqual(features["sofa_roll_home_shots_on_target_for_avg"], 6)
        self.assertEqual(features["sofa_roll_home_xg_games"], 1)

    def test_coverage_fields_do_not_change_legacy_snapshot_overlay(self):
        features = {"sofa_roll_available": 1, "sofa_roll_home_games": 5,
                    "sofa_roll_away_games": 5, "sofa_roll_home_ppg": 1.7,
                    "sofa_roll_away_ppg": 1.2, "sofa_roll_home_xg_for_avg": 1.5,
                    "sofa_roll_home_xg_against_avg": 0.8,
                    "sofa_roll_away_xg_for_avg": 1.1,
                    "sofa_roll_away_xg_against_avg": 1.0}
        before = sofa.analyze_pregame_context([0.42, 0.28, 0.3], features, "active-v5")
        features.update(sofa_roll_home_xg_games=3, sofa_roll_away_xg_games=4)
        after = sofa.analyze_pregame_context([0.42, 0.28, 0.3], features, "active-v5")
        self.assertEqual(before["probabilities"], after["probabilities"])


if __name__ == "__main__":
    unittest.main()
