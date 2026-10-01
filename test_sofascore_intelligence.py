import os
import json
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from sofascore_intelligence import (
    _parse_h2h_payload,
    _rolling_team_features,
    analyze_pregame_context,
    backfill_historical_pregame,
    capture_pregame_contexts,
    classify_postmortem,
    get_pregame_features,
    init_sofascore_db,
    monitor_pregame_context_sources,
    reclassify_stored_postmortems,
    save_prediction_snapshot,
    historical_backfill_status,
    seed_historical_score_profiles,
)


class SofaScoreIntelligenceTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        init_sofascore_db(self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def test_prediction_snapshot_requires_a_verifiable_future_kickoff(self):
        analysis = {
            "base_probabilities": [.4, .3, .3],
            "probabilities": [.4, .3, .3],
        }
        save_prediction_snapshot(
            self.db_path, "past", int(time.time()) - 1, {}, analysis, "MANDANTE"
        )
        save_prediction_snapshot(
            self.db_path, "unknown", 0, {}, analysis, "MANDANTE"
        )
        save_prediction_snapshot(
            self.db_path, "future", int(time.time()) + 3600, {}, analysis, "MANDANTE"
        )
        conn = sqlite3.connect(self.db_path)
        try:
            saved = [row[0] for row in conn.execute(
                "SELECT match_id FROM ml_prediction_snapshots ORDER BY match_id"
            )]
        finally:
            conn.close()
        self.assertEqual(saved, ["future"])

    def test_rolling_profile_keeps_tactical_statistics(self):
        observed = json.dumps([
            "goals_for", "goals_against", "shots_for", "shots_against",
            "box_shots_for", "box_shots_against", "possession", "dominance",
            "result_points", "is_draw",
        ])
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO sofascore_team_match_profiles
                   (match_id,team_key,team_name,opponent_key,start_timestamp,is_home,
                    goals_for,goals_against,shots_for,shots_against,
                    box_shots_for,box_shots_against,possession,dominance,
                    result_points,is_draw,captured_at,metric_presence_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("old-1", "alpha", "Alpha", "beta", 1000, 1, 2, 1,
                 14, 8, 9, 4, 61, .64, 3, 0, 2000, observed),
            )
            conn.commit()
        finally:
            conn.close()
        features = _rolling_team_features(self.db_path, "home", "Alpha", 5000)
        self.assertEqual(features["sofa_roll_home_shots_for_avg"], 14)
        self.assertEqual(features["sofa_roll_home_box_shots_for_avg"], 9)
        self.assertEqual(features["sofa_roll_home_possession_avg"], 61)
        self.assertEqual(features["sofa_roll_home_possession_games"], 1)

    def test_pregame_capture_is_cached_and_never_needs_postgame_data(self):
        start = int(time.time()) + 7200
        event = {
            "event": {
                "id": 123, "startTimestamp": start,
                "homeTeam": {"id": 1, "name": "Alpha FC"},
                "awayTeam": {"id": 2, "name": "Beta"},
            }
        }
        form = {
            "homeTeam": {"form": ["W", "D", "W", "L", "W"], "position": 2,
                         "value": "30", "avgRating": "6.95"},
            "awayTeam": {"form": ["D", "L", "D", "W", "L"], "position": 9,
                         "value": "19", "avgRating": "6.61"},
        }

        def fake_request(url):
            if url.endswith("/pregame-form"):
                return 200, form, None
            if url.endswith("/team-streaks"):
                return 200, {"general": []}, None
            return 200, event, None

        game = {"ID": "123", "Timestamp": start, "Time Casa": "Alpha",
                "Time Fora": "Beta", "Liga": "Teste"}
        with patch("sofascore_intelligence._request_json", side_effect=fake_request) as mocked:
            first = capture_pregame_contexts(self.db_path, [game])
            second = capture_pregame_contexts(self.db_path, [game])
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(first["captured"], 1)
        self.assertEqual(second["context_cache_hits"], 1)
        features = get_pregame_features(
            self.db_path, "123", "Alpha", "Beta", cutoff_timestamp=start
        )
        self.assertEqual(features["sofa_pre_available"], 1.0)
        self.assertGreater(features["sofa_pre_ppg_diff"], 0)

    def test_match_link_rejects_one_good_side_and_wrong_category(self):
        start = int(time.time()) + 7200
        wrong_event = {
            "event": {
                "id": 777, "startTimestamp": start,
                "homeTeam": {"id": 1, "name": "Sunderland"},
                "awayTeam": {"id": 2, "name": "Completely Different"},
            }
        }

        def fake_request(url):
            if url.endswith("/pregame-form"):
                return 200, {"homeTeam": {"form": ["W"]},
                             "awayTeam": {"form": ["L"]}}, None
            if url.endswith("/team-streaks") or url.endswith("/h2h/events"):
                return 200, {}, None
            return 200, wrong_event, None

        game = {"ID": "777", "Timestamp": start,
                "Time Casa": "Sunderland U21", "Time Fora": "Beta",
                "Liga": "Teste"}
        with patch("sofascore_intelligence._request_json", side_effect=fake_request):
            result = capture_pregame_contexts(self.db_path, [game])
        self.assertEqual(result["link_mismatch"], 1)
        features = get_pregame_features(
            self.db_path, "777", "Sunderland U21", "Beta", start
        )
        self.assertEqual(features.get("sofa_pre_available", 0), 0)

    def test_match_link_rejects_event_at_unrelated_time(self):
        start = int(time.time()) + 7200
        wrong_event = {"event": {
            "id": 778, "startTimestamp": start + 2 * 86400,
            "homeTeam": {"id": 1, "name": "Alpha"},
            "awayTeam": {"id": 2, "name": "Beta"},
        }}

        def fake_request(url):
            if url.endswith("/pregame-form"):
                return 200, {"homeTeam": {"form": ["W"]},
                             "awayTeam": {"form": ["L"]}}, None
            if url.endswith("/team-streaks") or url.endswith("/h2h/events"):
                return 200, {}, None
            return 200, wrong_event, None

        game = {"ID": "778", "Timestamp": start, "Time Casa": "Alpha",
                "Time Fora": "Beta", "Liga": "Teste"}
        with patch("sofascore_intelligence._request_json", side_effect=fake_request):
            result = capture_pregame_contexts(self.db_path, [game])
        self.assertEqual(result["link_mismatch"], 1)

    def test_malformed_h2h_timestamp_is_ignored(self):
        result = _parse_h2h_payload(
            {"events": [{"startTimestamp": "not-a-timestamp"}]},
            "Alpha", "Beta", int(time.time()),
        )
        self.assertEqual(result["sofa_h2h_available"], 0.0)

    def test_reader_rejects_context_from_an_id_collision(self):
        start = int(time.time()) + 7200
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""INSERT OR REPLACE INTO sofascore_pregame_context
                (match_id,captured_at,start_timestamp,home_name,away_name,
                 features_json,coverage,source_version)
                VALUES (?,?,?,?,?,?,?,?)""", (
                    "collision", int(time.time()), start, "Senior Alpha", "Senior Beta",
                    json.dumps({"sofa_pre_available": 1.0}), 1.0, "test",
                ))
            conn.commit()
        finally:
            conn.close()
        features = get_pregame_features(
            self.db_path, "collision", "Senior Alpha U21", "Senior Beta U21", start
        )
        self.assertEqual(features.get("sofa_pre_available", 0), 0)

    def test_pregame_capture_uses_allsports_fallback_without_postgame_fields(self):
        start = int(time.time()) + 7200
        event = {"event": {"id": 321, "startTimestamp": start,
                           "homeTeam": {"id": 1, "name": "Alpha"},
                           "awayTeam": {"id": 2, "name": "Beta"}}}

        def fake_request(url):
            if url.endswith("/pregame-form") or url.endswith("/team-streaks"):
                return 403, None, "blocked"
            if url.endswith("/h2h/events"):
                return 404, None, "missing"
            return 200, event, None

        fallback_calls = []

        def fallback(game, need_form, need_streaks):
            fallback_calls.append((need_form, need_streaks))
            return {
                "form": {"homeTeam": {"form": ["W", "D"], "position": 2},
                         "awayTeam": {"form": ["L", "D"], "position": 8}},
                "streaks": {"general": []},
                "features": {"allsports_goal_home_available": 1.0,
                             "allsports_goal_away_available": 1.0},
                "http_requests": 4, "cache_hits": 0,
            }

        game = {"ID": "321", "Timestamp": start, "Time Casa": "Alpha",
                "Time Fora": "Beta", "Liga": "Teste"}
        with patch("sofascore_intelligence._request_json", side_effect=fake_request):
            result = capture_pregame_contexts(
                self.db_path, [game], pregame_payload_fallback=fallback
            )
        features = get_pregame_features(
            self.db_path, "321", "Alpha", "Beta", cutoff_timestamp=start
        )
        self.assertEqual(fallback_calls, [(True, True)])
        self.assertEqual(result["allsports_pregame_http"], 4)
        self.assertEqual(features["sofa_pre_available"], 1.0)
        self.assertEqual(features["allsports_goal_home_available"], 1.0)
        self.assertNotIn("statistics", features)

    def test_context_overlay_handles_draw_risk_without_using_odds(self):
        features = {
            "context_sfi_available": 1.0,
            "form_sfi_home_ppg": 1.4, "form_sfi_away_ppg": 1.4,
            "form_sfi_home_recent_points_3": 0.44,
            "form_sfi_away_recent_points_3": 0.44,
            "form_sfi_home_draw_rate": 0.6, "form_sfi_away_draw_rate": 0.6,
            "liga_prior_empate": 0.32,
        }
        result = analyze_pregame_context([0.38, 0.27, 0.35], features)
        self.assertGreater(result["draw_risk"], 0.27)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0)
        self.assertNotIn("odd_casa", features)

    def test_context_source_monitor_uses_frozen_pregame_snapshot(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO ml_prediction_snapshots
                   (match_id,captured_at,feature_version,features_json,
                    predicted_outcome,source_version)
                   VALUES (?,?,?,?,?,?)""",
                ("m1", int(time.time()), 8, json.dumps({
                    "allsports_goal_home_available": 1.0,
                    "allsports_goal_away_available": 1.0,
                    "sofa_roll_home_games": 5.0,
                }), "MANDANTE", "test"),
            )
            conn.execute(
                """INSERT INTO match_postmortems
                   (match_id,audited_at,predicted_outcome,actual_outcome,
                    verdict,learning_weight,source_version)
                   VALUES (?,?,?,?,?,?,?)""",
                ("m1", int(time.time()), "MANDANTE", "MANDANTE",
                 "ACERTO", 1.0, "test"),
            )
            conn.commit()
        finally:
            conn.close()
        result = monitor_pregame_context_sources(self.db_path)
        self.assertEqual(result["allsports_goal_distribution"]["resolved"], 1)
        self.assertEqual(result["allsports_goal_distribution"]["accuracy"], 1.0)
        self.assertEqual(result["sofascore_recent_form"]["snapshots"], 1)

    def test_active_v5_profile_is_identified_without_replay_label(self):
        result = analyze_pregame_context(
            [0.38, 0.27, 0.35], {}, analysis_profile="active-v5"
        )
        self.assertEqual(result["analysis_version"], "sofa-context-v5")

    def test_live_last_five_falls_back_and_respects_temporal_cutoff(self):
        start = int(time.time()) + 7200
        current_event = {
            "event": {
                "id": 900, "startTimestamp": start,
                "homeTeam": {"id": 1, "name": "Alpha FC"},
                "awayTeam": {"id": 2, "name": "Beta"},
            }
        }

        def recent_payload(team_id, team_name, opponent_prefix, wins):
            events = []
            for index in range(5):
                is_win = index < wins
                events.append({
                    "id": f"{team_id}-{index}",
                    "startTimestamp": start - (index + 2) * 86400,
                    "status": {"type": "finished"},
                    "homeTeam": {"id": team_id, "name": team_name},
                    "awayTeam": {"id": f"o-{team_id}-{index}",
                                 "name": f"{opponent_prefix} {index}"},
                    "homeScore": {"normaltime": 2 if is_win else 0},
                    "awayScore": {"normaltime": 0 if is_win else 1},
                })
            # Este jogo nunca pode entrar porque ocorre depois do confronto.
            events.append({
                "id": f"future-{team_id}", "startTimestamp": start + 86400,
                "status": {"type": "finished"},
                "homeTeam": {"id": team_id, "name": team_name},
                "awayTeam": {"id": "future-opponent", "name": "Future"},
                "homeScore": {"normaltime": 9},
                "awayScore": {"normaltime": 0},
            })
            return {"events": events}

        def fake_request(url):
            if "/team/" in url:
                return 403, None, "blocked"
            if url.endswith("/pregame-form"):
                return 404, None, "missing"
            if url.endswith("/team-streaks") or url.endswith("/h2h/events"):
                return 404, None, "missing"
            return 200, current_event, None

        def fallback(game, side, provider_team_id):
            if side == "home":
                return {"provider": "allsports", "team_id": "1",
                        "payload": recent_payload("1", "Alpha FC", "Strong", 4)}
            return {"provider": "allsports", "team_id": "2",
                    "payload": recent_payload("2", "Beta", "Weak", 1)}

        game = {"ID": "900", "Timestamp": start, "Time Casa": "Alpha",
                "Time Fora": "Beta", "Liga": "Teste"}
        with patch("sofascore_intelligence._request_json", side_effect=fake_request):
            summary = capture_pregame_contexts(
                self.db_path, [game], recent_form_fallback=fallback
            )
        features = get_pregame_features(
            self.db_path, "900", "Alpha", "Beta", cutoff_timestamp=start
        )
        self.assertEqual(summary["live_available"], 1)
        self.assertEqual(summary["live_provider_allsports"], 2)
        self.assertEqual(features["live_recent_home_games"], 5.0)
        self.assertEqual(features["live_recent_away_games"], 5.0)
        self.assertGreater(features["live_recent_home_weighted_ppg"],
                           features["live_recent_away_weighted_ppg"])
        conn = sqlite3.connect(self.db_path)
        try:
            future = conn.execute(
                """SELECT COUNT(*) FROM sofascore_team_match_profiles
                   WHERE match_id LIKE 'future-%'"""
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(future, 0)

    def test_live_recent_form_materially_corrects_stale_base_reading(self):
        features = {
            "live_recent_available": 1.0,
            "live_recent_home_games": 5.0,
            "live_recent_away_games": 5.0,
            "live_recent_home_strength_adjusted_ppg": 2.5,
            "live_recent_away_strength_adjusted_ppg": 0.4,
            "live_recent_home_weighted_ppg": 2.6,
            "live_recent_away_weighted_ppg": 0.3,
            "live_recent_home_same_venue_ppg": 2.4,
            "live_recent_away_same_venue_ppg": 0.5,
            "live_recent_home_trend": .4,
            "live_recent_away_trend": -.4,
            "live_recent_home_goals_for_avg": 2.0,
            "live_recent_home_goals_against_avg": .6,
            "live_recent_away_goals_for_avg": .5,
            "live_recent_away_goals_against_avg": 1.9,
            "live_recent_home_draw_rate": .2,
            "live_recent_away_draw_rate": .2,
        }
        result = analyze_pregame_context([.30, .25, .45], features)
        self.assertGreater(result["context_conflict"], .18)
        self.assertGreater(result["probabilities"][0], .30)
        self.assertLess(result["probabilities"][2], .45)

    def test_strong_context_conflict_has_material_influence(self):
        features = {
            "context_sfi_available": 1.0,
            "form_sfi_home_ppg": 0.4, "form_sfi_away_ppg": 2.5,
            "form_sfi_home_recent_points_3": 0.1,
            "form_sfi_away_recent_points_3": 0.9,
            "form_sfi_home_goals_for_avg": 0.6,
            "form_sfi_home_goals_against_avg": 2.1,
            "form_sfi_away_goals_for_avg": 2.2,
            "form_sfi_away_goals_against_avg": 0.7,
            "form_sfi_home_draw_rate": 0.2, "form_sfi_away_draw_rate": 0.2,
            "liga_prior_empate": 0.28,
        }
        result = analyze_pregame_context([0.49, 0.25, 0.26], features)
        self.assertGreaterEqual(result["context_conflict"], 0.18)
        self.assertGreaterEqual(result["blend"], 0.30)
        self.assertLess(result["probabilities"][0], 0.44)

    def test_corroborated_draw_is_not_flipped_by_one_medium_source(self):
        features = {
            "context_sfi_available": 1.0,
            "form_sfi_home_games": 5, "form_sfi_away_games": 5,
            "form_sfi_home_ppg": 1.6, "form_sfi_away_ppg": 2.4,
            "form_sfi_home_recent_points_3": .44,
            "form_sfi_away_recent_points_3": .67,
            "form_sfi_home_goals_for_avg": .8,
            "form_sfi_home_goals_against_avg": .2,
            "form_sfi_away_goals_for_avg": 3.8,
            "form_sfi_away_goals_against_avg": 1.8,
            "form_sfi_home_draw_rate": .4, "form_sfi_away_draw_rate": 0,
            "liga_prior_empate": .32, "context_empate_composto": .34,
            "context_paridade_ppg": .8,
        }
        result = analyze_pregame_context([.3370, .3372, .3258], features)
        self.assertEqual(max(range(3), key=lambda i: result["probabilities"][i]), 1)
        self.assertEqual(result["blend"], 0.0)

    def test_second_leg_leader_is_not_treated_like_regular_away_win(self):
        features = {
            "is_qualifier": 1.0,
            "sofa_h2h_available": 1.0,
            "sofa_h2h_games": 1.0,
            "sofa_h2h_home_ppg": 0.0,
            "sofa_h2h_away_ppg": 3.0,
            "sofa_h2h_draw_rate": 0.0,
            "sofa_h2h_recent_reverse": 1.0,
            "sofa_h2h_home_aggregate_deficit": 1.0,
        }
        result = analyze_pregame_context([.31, .30, .39], features)
        self.assertGreater(result["second_leg_tactical_adjustment"], 0)
        self.assertEqual(max(range(3), key=lambda i: result["probabilities"][i]), 1)

    def test_low_sample_symmetric_sides_can_promote_corroborated_draw(self):
        features = {
            "context_sfi_available": 1.0,
            "form_sfi_home_games": 0, "form_sfi_away_games": 1,
            "form_sfi_home_ppg": 0.0, "form_sfi_away_ppg": 1.0,
            "form_sfi_home_draw_rate": 1.0, "form_sfi_away_draw_rate": 1.0,
            "liga_prior_empate": .34, "context_empate_composto": .34,
            "context_paridade_ppg": 1.0, "context_baixa_intensidade": .6,
        }
        result = analyze_pregame_context([.371, .257, .372], features)
        self.assertEqual(max(range(3), key=lambda i: result["probabilities"][i]), 1)
        self.assertEqual(result["low_sample_draw_override"], 1.0)

    def test_near_tie_draw_is_promoted_only_with_independent_risk(self):
        features = {
            "context_sfi_available": 1.0,
            "form_sfi_home_games": 5, "form_sfi_away_games": 5,
            "form_sfi_home_ppg": 1.4, "form_sfi_away_ppg": 1.4,
            "form_sfi_home_draw_rate": .4, "form_sfi_away_draw_rate": .4,
            "liga_prior_empate": .34, "context_empate_composto": .34,
            "context_paridade_ppg": .9, "context_baixa_intensidade": .6,
        }
        result = analyze_pregame_context([.334, .332, .334], features)
        self.assertEqual(max(range(3), key=lambda i: result["probabilities"][i]), 1)
        self.assertEqual(result["near_tie_draw_override"], 1.0)

        decisive = analyze_pregame_context([.37, .32, .31], features)
        self.assertEqual(decisive["near_tie_draw_override"], 0.0)

    def test_duel_uses_venue_and_attack_against_defense_to_correct_side(self):
        features = {
            "live_recent_available": 1.0,
            "live_recent_home_games": 5, "live_recent_away_games": 5,
            "live_recent_home_strength_adjusted_ppg": 1.2,
            "live_recent_away_strength_adjusted_ppg": 1.4,
            "live_recent_home_weighted_ppg": 1.2,
            "live_recent_away_weighted_ppg": 1.4,
            "live_recent_home_same_venue_games": 3,
            "live_recent_away_same_venue_games": 3,
            "live_recent_home_same_venue_ppg": 2.4,
            "live_recent_away_same_venue_ppg": .4,
            "live_recent_home_goals_for_avg": 2.1,
            "live_recent_home_goals_against_avg": .8,
            "live_recent_away_goals_for_avg": .7,
            "live_recent_away_goals_against_avg": 1.9,
            "live_recent_home_draw_rate": .1,
            "live_recent_away_draw_rate": .1,
        }
        result = analyze_pregame_context([.31, .25, .44], features)
        self.assertEqual(max(range(3), key=lambda i: result["probabilities"][i]), 0)
        self.assertEqual(result["duel_side_override"], 1.0)
        self.assertGreater(result["duel_venue_signal"], 0)
        self.assertGreater(result["duel_attack_defense_signal"], 0)

    def test_duel_never_replaces_a_draw_already_on_top(self):
        features = {
            "live_recent_available": 1.0,
            "live_recent_home_games": 5, "live_recent_away_games": 5,
            "live_recent_home_same_venue_games": 3,
            "live_recent_away_same_venue_games": 3,
            "live_recent_home_same_venue_ppg": 0,
            "live_recent_away_same_venue_ppg": 3,
            "live_recent_home_goals_for_avg": .4,
            "live_recent_home_goals_against_avg": 2.0,
            "live_recent_away_goals_for_avg": 2.2,
            "live_recent_away_goals_against_avg": .5,
            "live_recent_home_draw_rate": .4,
            "live_recent_away_draw_rate": .4,
            "liga_prior_empate": .34,
        }
        result = analyze_pregame_context([.33, .34, .33], features)
        self.assertEqual(max(range(3), key=lambda i: result["probabilities"][i]), 1)
        self.assertEqual(result["duel_side_override"], 0.0)

    def test_legacy_base_conflict_is_not_mistaken_for_final_pick_conflict(self):
        result = classify_postmortem(
            "VISITANTE", "EMPATE", "RED", {}, 0.0,
            {"context_conflict": .40, "side_strength": -.65,
             "information_quality": .8, "sample_reliability": 1.0,
             "draw_risk": .20, "draw_corroboration": 0},
        )
        self.assertEqual(result["verdict"], "INSUFFICIENT_DATA")

    def test_red_card_while_level_is_incident_variance(self):
        result = classify_postmortem(
            "VISITANTE", "MANDANTE", "RED", {
                "away_red_cards": 1.0, "away_red_while_level": 1.0,
                "home_red_cards": 0.0,
            }, 0.0, {},
        )
        self.assertEqual(result["verdict"], "MATCH_INCIDENT_VARIANCE")
        self.assertLess(result["learning_weight"], 1.0)

    def test_decisive_xg_and_shots_work_with_partial_coverage(self):
        result = classify_postmortem(
            "VISITANTE", "EMPATE", "RED", {
                "home_xg": 2.29, "away_xg": .55,
                "home_shots": 20, "away_shots": 9,
            }, .25, {},
        )
        self.assertEqual(result["verdict"], "MODEL_READING_ERROR")

    def test_local_score_seed_gives_temporal_history_without_http(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""CREATE TABLE training_data (
                match_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                data_jogo TEXT, home_score REAL, away_score REAL)""")
            conn.execute(
                "INSERT INTO training_data VALUES (?,?,?,?,?,?)",
                ("seed-1", "Alpha", "Beta", "2025-01-01 18:00:00", 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        result = seed_historical_score_profiles(self.db_path)
        self.assertEqual(result["profiles_added"], 2)
        before = get_pregame_features(
            self.db_path, "future-0", "Alpha", "Beta", cutoff_timestamp=1
        )
        after = get_pregame_features(
            self.db_path, "future-1", "Alpha", "Beta",
            cutoff_timestamp=int(time.time()),
        )
        self.assertEqual(before["sofa_roll_available"], 0.0)
        self.assertEqual(after["sofa_roll_available"], 1.0)
        self.assertEqual(after["sofa_roll_home_draw_rate"], 1.0)

    def test_clear_statistical_superiority_is_model_error_not_zebra(self):
        metrics = {
            "home_xg": 1.26, "away_xg": 3.50,
            "home_big_chances": 1, "away_big_chances": 5,
            "home_shots": 15, "away_shots": 15,
            "home_shots_on_target": 2, "away_shots_on_target": 10,
            "home_box_shots": 6, "away_box_shots": 11,
            "home_possession": 57, "away_possession": 43,
        }
        result = classify_postmortem(
            "MANDANTE", "VISITANTE", "RED ❌", metrics, 1.0
        )
        self.assertEqual(result["verdict"], "MODEL_READING_ERROR")
        self.assertGreater(result["learning_weight"], 1.0)

    def test_dominant_loser_is_variance_and_gets_lower_weight(self):
        metrics = {
            "home_xg": 2.4, "away_xg": 0.8,
            "home_big_chances": 4, "away_big_chances": 1,
            "home_shots": 22, "away_shots": 6,
            "home_shots_on_target": 8, "away_shots_on_target": 3,
            "home_box_shots": 15, "away_box_shots": 4,
            "home_possession": 65, "away_possession": 35,
        }
        result = classify_postmortem(
            "MANDANTE", "VISITANTE", "RED ❌", metrics, 1.0
        )
        self.assertEqual(result["verdict"], "ZEBRA_VARIANCE")
        self.assertLess(result["learning_weight"], 1.0)

    def test_draw_is_not_overlearned_without_pregame_corroboration(self):
        metrics = {
            "home_xg": 1.1, "away_xg": 1.0,
            "home_shots": 10, "away_shots": 9,
            "home_shots_on_target": 4, "away_shots_on_target": 4,
            "home_possession": 51, "away_possession": 49,
        }
        result = classify_postmortem(
            "MANDANTE", "EMPATE", "RED", metrics, 1.0,
            {"draw_risk": 0.31, "draw_corroboration": 1,
             "sample_reliability": 0.4, "information_quality": 0.5},
        )
        self.assertEqual(result["verdict"], "DRAW_UNCERTAIN")
        self.assertLess(result["learning_weight"], 1.2)

    def test_high_context_score_alone_does_not_make_draw_foreseeable(self):
        result = classify_postmortem(
            "MANDANTE", "EMPATE", "RED", {}, 0.0,
            {"draw_risk": .38, "probabilities": [.45, .28, .27],
             "draw_corroboration": 4, "sample_reliability": 1.0},
        )
        self.assertEqual(result["verdict"], "INSUFFICIENT_DATA")

    def test_near_top_draw_with_three_signals_is_learned_as_missed(self):
        result = classify_postmortem(
            "MANDANTE", "EMPATE", "RED", {}, 0.0,
            {"probabilities": [.345, .33, .325],
             "draw_corroboration": 3, "sample_reliability": .9},
        )
        self.assertEqual(result["verdict"], "DRAW_RISK_MISSED")
        self.assertGreater(result["learning_weight"], 1.0)

    def test_legacy_postmortem_reclassification_uses_no_http(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""INSERT INTO ml_prediction_snapshots
                (match_id,captured_at,start_timestamp,feature_version,features_json,
                 base_probabilities_json,final_probabilities_json,predicted_outcome,
                 information_quality,draw_risk,context_conflict,analysis_json,source_version)
                VALUES ('x',1,1,4,'{}','[.4,.3,.3]','[.4,.3,.3]',
                        'MANDANTE',.5,.31,0,'{}','old')""")
            conn.execute("""INSERT INTO match_postmortems
                (match_id,audited_at,prediction_status,predicted_outcome,actual_outcome,
                 scoreline,verdict,process_score,chosen_dominance,opponent_dominance,
                 learning_weight,error_margin,data_coverage,metrics_json,reasons_json,source_version)
                VALUES ('x',1,'RED','MANDANTE','EMPATE','1-1','DRAW_RISK_MISSED',
                        .5,.5,.5,1.45,.55,1.0,'{}','[]','old')""")
            conn.commit()
        finally:
            conn.close()
        with patch("sofascore_intelligence._request_json") as request:
            result = reclassify_stored_postmortems(self.db_path, ["x"])
        self.assertEqual(request.call_count, 0)
        self.assertEqual(result["reclassified"], 1)
        conn = sqlite3.connect(self.db_path)
        try:
            verdict = conn.execute(
                "SELECT verdict FROM match_postmortems WHERE match_id='x'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(verdict, "DRAW_UNCERTAIN")

    def test_historical_backfill_is_resumable_and_keeps_errors_pending(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""CREATE TABLE training_data (
                match_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                data_jogo TEXT)""")
            conn.executemany(
                "INSERT INTO training_data VALUES (?,?,?,?)",
                [
                    ("101", "Alpha", "Beta", "2025-01-01 18:00:00"),
                    ("102", "Gamma", "Delta", "2025-01-02 18:00:00"),
                    ("103", "Epsilon", "Zeta", "2025-01-03 18:00:00"),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        form = {
            "homeTeam": {"form": ["W", "D", "W"], "position": 2, "value": "20"},
            "awayTeam": {"form": ["L", "D", "L"], "position": 8, "value": "10"},
        }

        def first_pass(url):
            if "/101/" in url:
                return 200, form, None
            if "/102/" in url:
                return 404, None, "not found"
            return 503, None, "temporary"

        with patch("sofascore_intelligence._request_json", side_effect=first_pass):
            result = backfill_historical_pregame(self.db_path)
        self.assertEqual(result["available"], 1)
        self.assertEqual(result["no_coverage"], 1)
        self.assertEqual(result["error"], 1)
        status = historical_backfill_status(self.db_path)
        self.assertEqual(status["processed"], 2)
        self.assertEqual(status["remaining"], 1)

        with patch("sofascore_intelligence._request_json", return_value=(200, form, None)) as retried:
            second = backfill_historical_pregame(self.db_path)
        self.assertEqual(second["queued"], 1)
        self.assertEqual(retried.call_count, 1)
        final = historical_backfill_status(self.db_path)
        self.assertEqual(final["processed"], 3)
        self.assertEqual(final["remaining"], 0)

    def test_historical_backfill_opens_circuit_after_three_forbidden_responses(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""CREATE TABLE training_data (
                match_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                data_jogo TEXT)""")
            conn.executemany(
                "INSERT INTO training_data VALUES (?,?,?,?)",
                [(str(i), f"H{i}", f"A{i}", "2025-01-01 18:00:00")
                 for i in range(10)],
            )
            conn.commit()
        finally:
            conn.close()
        with patch(
            "sofascore_intelligence._request_json",
            return_value=(403, None, "forbidden"),
        ) as blocked:
            result = backfill_historical_pregame(self.db_path)
        self.assertEqual(blocked.call_count, 3)
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["circuit_open"], 1)
        self.assertEqual(result["remaining_batch"], 7)
        self.assertEqual(historical_backfill_status(self.db_path)["remaining"], 10)


if __name__ == "__main__":
    unittest.main()
