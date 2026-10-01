import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone

import allsports_api
from allsports_api import (
    fetch_allsports_postmatch_resources,
    fetch_allsports_pregame_context,
    goal_distribution_url,
    fetch_football_events_for_date,
    match_odds_url,
    matches_odds_date_url,
    scheduled_tournaments_url,
    tournament_scheduled_events_url,
)


class AllSportsApiV2Tests(unittest.TestCase):
    def setUp(self):
        self._old_cache_db = allsports_api.SCHEDULE_CACHE_DB_PATH
        self._temp_dir = tempfile.TemporaryDirectory()
        allsports_api.SCHEDULE_CACHE_DB_PATH = os.path.join(
            self._temp_dir.name, "schedule-cache.db"
        )
        with allsports_api._schedule_cache_lock:
            allsports_api._schedule_cache.clear()

    def tearDown(self):
        with allsports_api._schedule_cache_lock:
            allsports_api._schedule_cache.clear()
        allsports_api.SCHEDULE_CACHE_DB_PATH = self._old_cache_db
        self._temp_dir.cleanup()

    def test_current_documented_urls(self):
        host = "allsportsapi2.p.rapidapi.com"
        self.assertEqual(allsports_api.SCHEDULE_MAX_TOURNAMENTS, 0)
        self.assertEqual(
            match_odds_url(host, 14081808),
            "https://allsportsapi2.p.rapidapi.com/api/match/14081808/odds/1/all",
        )
        self.assertEqual(
            matches_odds_date_url(host, "20/08/2026"),
            "https://allsportsapi2.p.rapidapi.com/api/matches/odds/20/8/2026",
        )
        self.assertEqual(
            scheduled_tournaments_url(host, "2026-08-20", 1),
            "https://allsportsapi2.p.rapidapi.com/api/scheduled-tournaments/20/8/2026/page/1",
        )
        self.assertEqual(
            tournament_scheduled_events_url(host, 7, "20/08/2026"),
            "https://allsportsapi2.p.rapidapi.com/api/tournament/7/scheduled-events/2026-08-20",
        )
        self.assertEqual(
            goal_distribution_url(host, 11, 242, 86668),
            "https://allsportsapi2.p.rapidapi.com/api/team/11/tournament/242/season/86668/goal-distributions",
        )

    def test_pregame_context_parses_venue_goal_distribution_and_caches(self):
        host = "pregame-allsports.test"
        calls = []

        def fake_safe_get(url):
            calls.append(url)
            if url.endswith("/form"):
                return {"homeTeam": {"form": ["W"]}, "awayTeam": {"form": ["L"]}}
            if url.endswith("/streaks"):
                return {"general": [{"name": "No wins", "team": "away", "value": "4"}]}
            if "/team/11/" in url:
                venue = "home"; scored, conceded = 18, 8
            else:
                venue = "away"; scored, conceded = 7, 15
            return {"goalDistributions": [{
                "type": venue, "matches": 10, "scoredGoals": scored,
                "concededGoals": conceded,
                "periods": [{"periodEnd": 45, "scoredGoals": 4, "concededGoals": 3}],
            }]}

        game = {"ID": 900, "Home_ID": 11, "Away_ID": 22,
                "Unique_Tournament_ID": 242, "Season_ID": 86668}
        first = fetch_allsports_pregame_context(host, game, fake_safe_get)
        second = fetch_allsports_pregame_context(
            host, game, fake_safe_get, need_form=False, need_streaks=False
        )
        self.assertEqual(first["features"]["allsports_goal_home_gf_avg"], 1.8)
        self.assertEqual(first["features"]["allsports_goal_away_ga_avg"], 1.5)
        self.assertGreater(first["features"]["allsports_goal_attack_matchup_diff"], 0)
        self.assertEqual(second["cache_hits"], 2)
        self.assertEqual(second["http_requests"], 0)
        self.assertEqual(len(calls), 4)

    def test_pregame_context_hydrates_missing_ids_from_match_detail(self):
        host = "hydrate-allsports.test"
        calls = []

        def fake_safe_get(url):
            calls.append(url)
            if url.endswith("/api/match/901"):
                return {"event": {
                    "competitionType": 1,
                    "roundInfo": {"round": 8},
                    "tournament": {"uniqueTournament": {"id": 242}},
                    "season": {"id": 86668},
                    "homeTeam": {"id": 11, "userCount": 100},
                    "awayTeam": {"id": 22, "userCount": 80},
                }}
            return {"goalDistributions": [{
                "type": "home" if "/team/11/" in url else "away",
                "matches": 5, "scoredGoals": 7, "concededGoals": 4,
                "periods": [],
            }]}

        game = {"ID": 901, "Home_ID": "", "Away_ID": "",
                "Unique_Tournament_ID": "", "Season_ID": ""}
        result = fetch_allsports_pregame_context(
            host, game, fake_safe_get, need_form=False, need_streaks=False
        )
        self.assertEqual(result["http_requests"], 3)
        self.assertEqual(game["Unique_Tournament_ID"], "242")
        self.assertEqual(game["Season_ID"], "86668")
        self.assertEqual(result["features"]["allsports_round"], 8.0)
        self.assertEqual(result["features"]["allsports_goal_home_available"], 1.0)
        self.assertEqual(result["features"]["allsports_goal_away_available"], 1.0)

    def test_postmatch_fallback_is_persistent_and_budgeted(self):
        calls = []

        def fake_safe_get(url):
            calls.append(url)
            return {"statistics": [{"period": "ALL", "groups": []}]}

        first = fetch_allsports_postmatch_resources(
            "postmatch.test", "991", ["statistics", "invalid"], fake_safe_get
        )
        second = fetch_allsports_postmatch_resources(
            "postmatch.test", "991", ["statistics"], fake_safe_get
        )
        self.assertEqual(first["http_requests"], 1)
        self.assertIn("statistics", first["resources"])
        self.assertEqual(second["http_requests"], 0)
        self.assertEqual(second["cache_hits"], 1)
        self.assertEqual(len(calls), 1)

    def test_recent_form_resolves_team_id_from_current_match(self):
        host = "fake-allsports.test"
        calls = []

        def fake_safe_get(url):
            calls.append(url)
            if url.endswith("/api/match/900"):
                return {"event": {
                    "homeTeam": {"id": 11}, "awayTeam": {"id": 22}
                }}
            if url.endswith("/api/team/11/matches/previous/0"):
                return {"events": [{"id": 1}]}
            return None

        game = {"ID": "900", "Home_ID": "", "Away_ID": ""}
        result = allsports_api.fetch_recent_team_events_for_game(
            host, game, "home", fake_safe_get
        )
        self.assertEqual(result["provider"], "allsports")
        self.assertEqual(result["team_id"], "11")
        self.assertEqual(game["Home_ID"], "11")
        self.assertEqual(len(calls), 2)

    def test_schedule_flow_uses_inner_tournament_id_and_deduplicates(self):
        host = "fake-allsports.test"
        calls = []
        calls_lock = threading.Lock()

        def scheduled(wrapper_id, tournament_id):
            return {
                "tournament": {
                    "id": wrapper_id,
                    "tournament": {"id": tournament_id},
                }
            }

        responses = {
            f"https://{host}/api/scheduled-tournaments/20/8/2026/page/1": {
                "scheduled": [scheduled(1007, 7), scheduled(1008, 8)],
                "hasNextPage": True,
            },
            f"https://{host}/api/scheduled-tournaments/20/8/2026/page/2": {
                "scheduled": [scheduled(2007, 7), scheduled(1009, 9)],
                "hasNextPage": False,
            },
            f"https://{host}/api/tournament/7/scheduled-events/2026-08-20": {
                "events": [{"id": 2, "startTimestamp": 200}]
            },
            f"https://{host}/api/tournament/8/scheduled-events/2026-08-20": {
                "events": [{"id": 1, "startTimestamp": 100}]
            },
            f"https://{host}/api/tournament/9/scheduled-events/2026-08-20": {
                "events": [
                    {"id": 2, "startTimestamp": 200},
                    {"id": 3, "startTimestamp": 300},
                ]
            },
        }

        def fake_safe_get(url, **_kwargs):
            with calls_lock:
                calls.append(url)
            return responses.get(url)

        result = fetch_football_events_for_date(
            fake_safe_get, host, "20/08/2026", force_refresh=True
        )

        self.assertEqual([event["id"] for event in result["events"]], [1, 2, 3])
        self.assertEqual(result["_meta"]["pages_consulted"], 2)
        self.assertEqual(result["_meta"]["tournaments_consulted"], 3)
        self.assertEqual(result["_meta"]["estimated_http_requests"], 5)
        self.assertNotIn(
            f"https://{host}/api/tournament/1007/scheduled-events/2026-08-20",
            calls,
        )
        self.assertIn(
            f"https://{host}/api/tournament/7/scheduled-events/2026-08-20",
            calls,
        )

    def test_schedule_flow_caps_requests_and_prefers_priority(self):
        host = "priority-allsports.test"
        calls = []
        old_limit = allsports_api.SCHEDULE_MAX_TOURNAMENTS
        old_target = allsports_api.SCHEDULE_TARGET_EVENTS
        allsports_api.SCHEDULE_MAX_TOURNAMENTS = 2
        allsports_api.SCHEDULE_TARGET_EVENTS = 999

        scheduled = []
        for tournament_id, priority in ((1, 10), (2, 900), (3, 500)):
            scheduled.append({
                "tournament": {
                    "id": 1000 + tournament_id,
                    "name": f"Liga {tournament_id}",
                    "priority": priority,
                    "tournament": {"id": tournament_id},
                },
                "timezoneEventCount": {"0": 4},
            })

        def fake_safe_get(url, **_kwargs):
            calls.append(url)
            if "scheduled-tournaments" in url:
                return {"scheduled": scheduled, "hasNextPage": False}
            tournament_id = int(url.split("/tournament/")[1].split("/")[0])
            return {"events": [{"id": tournament_id, "startTimestamp": tournament_id}]}

        try:
            result = fetch_football_events_for_date(
                fake_safe_get, host, "21/08/2026", force_refresh=True
            )
        finally:
            allsports_api.SCHEDULE_MAX_TOURNAMENTS = old_limit
            allsports_api.SCHEDULE_TARGET_EVENTS = old_target

        self.assertEqual(result["_meta"]["tournaments_available"], 3)
        self.assertEqual(result["_meta"]["tournaments_consulted"], 2)
        self.assertEqual(result["_meta"]["estimated_http_requests"], 3)
        self.assertEqual({event["id"] for event in result["events"]}, {2, 3})
        self.assertFalse(any("/tournament/1/" in url for url in calls))

    def test_persistent_cache_reuses_overlapping_date_without_api_calls(self):
        host = "persistent-allsports.test"
        calls = []

        def fake_safe_get(url, **_kwargs):
            calls.append(url)
            if "scheduled-tournaments" in url:
                return {
                    "scheduled": [{
                        "tournament": {
                            "id": 1007,
                            "name": "Liga Principal",
                            "priority": 900,
                            "tournament": {"id": 7},
                        },
                        "timezoneEventCount": {"0": 1},
                    }],
                    "hasNextPage": False,
                }
            return {"events": [{"id": 77, "startTimestamp": 123}]}

        first = fetch_football_events_for_date(
            fake_safe_get, host, "22/08/2026", force_refresh=True
        )
        first_call_count = len(calls)
        self.assertEqual(first["_meta"]["completion_rate"], 1.0)

        # Simula outro processo/dia: a RAM foi perdida, mas o SQLite permanece.
        with allsports_api._schedule_cache_lock:
            allsports_api._schedule_cache.clear()
        second = fetch_football_events_for_date(
            fake_safe_get, host, "22/08/2026"
        )

        self.assertEqual(len(calls), first_call_count)
        self.assertEqual(second["_meta"]["cache_hit"], "persistent")
        self.assertEqual(second["_meta"]["estimated_http_requests"], 0)
        self.assertEqual([event["id"] for event in second["events"]], [77])

    def test_brt_window_reuses_one_date_and_advances_only_the_next_one(self):
        brt = timezone(timedelta(hours=-3))
        first_start, first_end = allsports_api.radar_window_brt(
            datetime(2026, 8, 21, 17, 0, tzinfo=brt))
        second_start, second_end = allsports_api.radar_window_brt(
            datetime(2026, 8, 22, 17, 0, tzinfo=brt))
        self.assertEqual(first_start.strftime("%d/%m/%Y %H:%M"), "21/08/2026 22:00")
        self.assertEqual(second_start.strftime("%d/%m/%Y %H:%M"), "22/08/2026 22:00")
        first_dates = allsports_api.schedule_dates_for_brt_window(first_start, first_end)
        second_dates = allsports_api.schedule_dates_for_brt_window(second_start, second_end)
        self.assertEqual([str(day) for day in first_dates], ["2026-08-21", "2026-08-22"])
        self.assertEqual([str(day) for day in second_dates], ["2026-08-22", "2026-08-23"])
        self.assertEqual(set(first_dates) & set(second_dates), {first_dates[1]})

    def test_unlimited_scope_does_not_reuse_old_truncated_cache(self):
        host = "scope-allsports.test"
        calls = []
        old_limit = allsports_api.SCHEDULE_MAX_TOURNAMENTS
        scheduled = [{
            "tournament": {"id": 1000 + tid, "name": f"Liga {tid}",
                           "tournament": {"id": tid}},
            "timezoneEventCount": {"0": 1},
        } for tid in (1, 2, 3)]

        def fake_safe_get(url, **_kwargs):
            calls.append(url)
            if "scheduled-tournaments" in url:
                return {"scheduled": scheduled, "hasNextPage": False}
            tid = int(url.split("/tournament/")[1].split("/")[0])
            return {"events": [{"id": tid, "startTimestamp": tid}]}

        try:
            allsports_api.SCHEDULE_MAX_TOURNAMENTS = 2
            first = fetch_football_events_for_date(
                fake_safe_get, host, "23/08/2026", force_refresh=True)
            self.assertEqual(first["_meta"]["tournaments_consulted"], 2)
            with allsports_api._schedule_cache_lock:
                allsports_api._schedule_cache.clear()
            before = len(calls)
            allsports_api.SCHEDULE_MAX_TOURNAMENTS = 0
            second = fetch_football_events_for_date(fake_safe_get, host, "23/08/2026")
            self.assertGreater(len(calls), before)
            self.assertEqual(second["_meta"]["tournaments_consulted"], 3)
        finally:
            allsports_api.SCHEDULE_MAX_TOURNAMENTS = old_limit

    def test_pagination_limit_never_marks_partial_day_complete(self):
        host = "pagination-allsports.test"
        old_pages = allsports_api.SCHEDULE_MAX_PAGES
        allsports_api.SCHEDULE_MAX_PAGES = 1

        def fake_safe_get(url, **_kwargs):
            if "scheduled-tournaments" in url:
                return {
                    "scheduled": [{"tournament": {"id": 1007,
                        "tournament": {"id": 7}}}],
                    "hasNextPage": True,
                }
            return {"events": [{"id": 77, "startTimestamp": 123}]}

        try:
            result = fetch_football_events_for_date(
                fake_safe_get, host, "24/08/2026", force_refresh=True)
        finally:
            allsports_api.SCHEDULE_MAX_PAGES = old_pages
        self.assertTrue(result["_meta"]["pagination_truncated"])
        self.assertFalse(result["_meta"]["complete"])

    def test_partial_tournament_collection_keeps_events_and_reports_exact_missing_calls(self):
        host = "partial-allsports.test"
        scheduled = [{"tournament": {"id": 1000 + tid,
                      "tournament": {"id": tid}}} for tid in (1, 2, 3)]

        def fake_safe_get(url, **_kwargs):
            if "scheduled-tournaments" in url:
                return {"scheduled": scheduled, "hasNextPage": False}
            tid = int(url.split("/tournament/")[1].split("/")[0])
            if tid == 2:
                return None
            return {"events": [{"id": tid, "startTimestamp": tid}]}

        result = fetch_football_events_for_date(
            fake_safe_get, host, "25/08/2026", force_refresh=True)
        self.assertFalse(result["_meta"]["complete"])
        self.assertEqual(result["_meta"]["tournaments_succeeded"], 2)
        self.assertEqual(result["_meta"]["requests_missing_to_complete"], 1)
        self.assertTrue(result["_meta"]["requests_missing_count_exact"])
        self.assertEqual({event["id"] for event in result["events"]}, {1, 3})

    def test_listing_failure_reports_minimum_instead_of_false_exact_total(self):
        host = "partial-pages-allsports.test"

        def fake_safe_get(url, **_kwargs):
            if "page/1" in url:
                return {"scheduled": [{"tournament": {"id": 1007,
                        "tournament": {"id": 7}}}], "hasNextPage": True}
            if "page/2" in url:
                return None
            return {"events": [{"id": 77, "startTimestamp": 123}]}

        result = fetch_football_events_for_date(
            fake_safe_get, host, "26/08/2026", force_refresh=True)
        self.assertFalse(result["_meta"]["complete"])
        self.assertFalse(result["_meta"]["requests_missing_count_exact"])
        self.assertGreaterEqual(result["_meta"]["requests_missing_to_complete"], 1)
        self.assertEqual([event["id"] for event in result["events"]], [77])


if __name__ == "__main__":
    unittest.main()
