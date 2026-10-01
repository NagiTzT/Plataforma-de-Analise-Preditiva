import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing

import sofascore_intelligence as sofa
from football_model_features import add_measured_sofa_duel


class TacticalRollingFeatureTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        sofa.init_sofascore_db(self.path)

    def tearDown(self):
        sofa._initialized.discard(os.path.abspath(self.path))
        os.unlink(self.path)

    def test_windows_style_and_opponent_adjustment_are_temporal(self):
        measured = json.dumps([
            "goals_for", "goals_against", "result_points", "is_draw",
            "xg_for", "xg_against", "shots_for", "shots_against",
            "shots_on_target_for", "shots_on_target_against",
            "big_chances_for", "big_chances_against", "box_shots_for",
            "box_shots_against", "possession", "dominance",
        ])
        with closing(sqlite3.connect(self.path)) as conn:
            # Beta had a strong defence before every Alpha-Beta observation.
            for index in range(6):
                conn.execute(
                    """INSERT INTO sofascore_team_match_profiles
                       (match_id,team_key,team_name,start_timestamp,is_home,
                        goals_for,goals_against,result_points,is_draw,captured_at,
                        metric_presence_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (f"b{index}", "beta", "Beta", 10+index, 1, 1, .4, 2, 0, 50, measured),
                )
            for index in range(10):
                shots = 8 + index
                conn.execute(
                    """INSERT INTO sofascore_team_match_profiles
                       (match_id,team_key,team_name,opponent_key,start_timestamp,is_home,
                        goals_for,goals_against,xg_for,xg_against,shots_for,shots_against,
                        shots_on_target_for,shots_on_target_against,big_chances_for,
                        big_chances_against,box_shots_for,box_shots_against,possession,
                        dominance,result_points,is_draw,captured_at,metric_presence_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f"a{index}", "alpha", "Alpha", "beta", 100+index, 1,
                     2, 1, 1.2+index*.05, .8, shots, 7, 4, 2, 2, 1, 5, 3,
                     55+index*.2, .58, 3, 0, 200, measured),
                )
            conn.commit()
        features = sofa._rolling_team_features(self.path, "home", "Alpha", 1000)
        self.assertEqual(features["sofa_roll_home_window_5_games"], 5)
        self.assertEqual(features["sofa_roll_home_window_10_games"], 10)
        self.assertGreater(features["sofa_roll_home_shots_for_weighted_avg_5"],
                           features["sofa_roll_home_shots_for_avg_10"])
        self.assertGreater(features["sofa_roll_home_xg_for_opponent_adjusted_5"],
                           features["sofa_roll_home_xg_for_avg_5"])
        self.assertEqual(features["sofa_roll_home_style_available_5"], 1)
        self.assertGreater(features["sofa_roll_home_pressure_proxy_5"], .5)
        self.assertEqual(features["sofa_roll_home_venue_games_5"], 5)
        self.assertAlmostEqual(features["sofa_roll_home_venue_result_points_5"], 3.0)
        self.assertGreater(features["sofa_roll_home_goals_for_opponent_adjusted_5"], 2.0)
        self.assertEqual(features["sofa_roll_home_close_game_rate_5"], 1.0)

    def test_duel_exports_separate_five_and_ten_game_signals(self):
        features = {}
        for side, attack, defence in (("home", 1.7, .8), ("away", 1.1, 1.3)):
            for window in (5, 10):
                features[f"sofa_roll_{side}_goals_for_games_{window}"] = window
                features[f"sofa_roll_{side}_goals_against_games_{window}"] = window
                features[f"sofa_roll_{side}_goals_for_opponent_adjusted_{window}"] = attack
                features[f"sofa_roll_{side}_goals_against_opponent_adjusted_{window}"] = defence
                features[f"sofa_roll_{side}_draw_rate_weighted_{window}"] = .3
                features[f"sofa_roll_{side}_low_total_rate_{window}"] = .6
                features[f"sofa_roll_{side}_close_game_rate_{window}"] = .7
                features[f"sofa_roll_{side}_xg_for_games_{window}"] = window
                features[f"sofa_roll_{side}_xg_against_games_{window}"] = window
                features[f"sofa_roll_{side}_xg_for_opponent_adjusted_{window}"] = attack
                features[f"sofa_roll_{side}_xg_against_opponent_adjusted_{window}"] = defence
                features[f"sofa_roll_{side}_style_available_{window}"] = 1
                for metric in ("attack_intensity", "chance_quality", "box_shot_share",
                               "territorial_control", "pressure_proxy", "directness"):
                    features[f"sofa_roll_{side}_{metric}_{window}"] = 1 if side == "home" else .5
        add_measured_sofa_duel(features)
        self.assertEqual(features["sofa_roll_duel_window_5_available"], 1)
        self.assertEqual(features["sofa_roll_duel_goal_window_5_available"], 1)
        self.assertGreater(features["sofa_roll_duel_goal_window_5_draw_poisson"], 0)
        self.assertGreater(features["sofa_roll_duel_window_5_attack_gap"], 0)
        self.assertGreater(features["sofa_roll_duel_pressure_proxy_gap_10"], 0)


if __name__ == "__main__":
    unittest.main()
