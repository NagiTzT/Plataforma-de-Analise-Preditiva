import os
import tempfile
import unittest
from datetime import datetime, timezone

import soccer_football_info as sfi


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers, params, timeout):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return self.responses.pop(0)


class SoccerFootballInfoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp.name, "context.db")
        self.old_keys = sfi.SOCCER_API_KEYS
        self.old_limit = sfi.SOCCER_API_DAILY_LIMIT
        self.old_interval = sfi.SOCCER_API_MIN_INTERVAL_SECONDS
        sfi.SOCCER_API_KEYS = ["key-a", "key-b"]
        sfi.SOCCER_API_DAILY_LIMIT = 2
        sfi.SOCCER_API_MIN_INTERVAL_SECONDS = 0
        sfi._rotation_cursor = 0
        sfi._last_request_at = 0
        sfi.init_soccer_context_db(self.db_path)

    def tearDown(self):
        sfi.SOCCER_API_KEYS = self.old_keys
        sfi.SOCCER_API_DAILY_LIMIT = self.old_limit
        sfi.SOCCER_API_MIN_INTERVAL_SECONDS = self.old_interval
        sfi._rotation_cursor = 0
        self.temp.cleanup()

    def _event(self, home, away, event_id="m1"):
        return {
            "id": event_id,
            "date": "2026-08-22T15:00:00",
            "home_team": {"id": f"h-{event_id}", "name": home, "perf": "WWDLW"},
            "away_team": {"id": f"a-{event_id}", "name": away, "perf": "LDWDL"},
            "championship": {"name": "League"},
        }

    def _game(self, home, away, game_id="a1"):
        timestamp = int(datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc).timestamp())
        return {
            "ID": game_id,
            "Time Casa": home,
            "Time Fora": away,
            "Liga": "League",
            "Timestamp": timestamp,
            "Odd Casa": 2.05,
            "Odd Fora": 2.30,
        }

    def test_rotates_after_local_limit_without_exceeding_it(self):
        session = _FakeSession([_FakeResponse({"result": []}) for _ in range(3)])
        client = sfi.SoccerFootballInfoClient(self.db_path, session=session)
        for page in (1, 2, 3):
            self.assertIsNotNone(client.fetch_page("20260822", page, force_refresh=True))
        used_keys = [call["headers"]["x-rapidapi-key"] for call in session.calls]
        self.assertEqual(used_keys, ["key-a", "key-a", "key-b"])
        status = sfi.soccer_quota_status(self.db_path)
        self.assertEqual(status["keys"][0]["used"], 2)
        self.assertEqual(status["keys"][0]["available"], 0)

    def test_name_matching_handles_suffixes_accents_and_short_names(self):
        pairs = [
            ("Wycombe", "Plymouth", "Wycombe Wanderers", "Plymouth Argyle"),
            ("Çorum FK", "Kasımpaşa", "Corum", "Kasimpasa"),
            ("Toulouse", "Lyon", "Toulouse FC", "Olympique Lyonnais"),
            ("Antwerp", "Genk", "Royal Antwerp FC", "KRC Genk"),
            ("West Bromwich Albion", "Burnley", "West Brom", "Burnley"),
        ]
        for index, (source_home, source_away, target_home, target_away) in enumerate(pairs):
            matched = sfi.match_allsports_game(
                self._game(source_home, source_away, f"a{index}"),
                [self._event(target_home, target_away, f"s{index}")],
            )
            self.assertIsNotNone(matched, (source_home, source_away))
            self.assertGreaterEqual(matched["confidence"], 0.72)

    def test_name_similarity_never_crosses_squad_categories(self):
        mismatches = [
            ("Sunderland U21", "Sunderland"),
            ("Brazil U20 (W)", "Brazil"),
            ("Brazil U20 (W)", "Brazil U20"),
            ("Orlando City II", "Orlando City SC"),
            ("Brighton U21", "Brighton SC"),
            ("Bayern Munich Women", "Bayern Munich II"),
        ]
        for left, right in mismatches:
            with self.subTest(left=left,right=right):
                self.assertEqual(sfi.team_name_similarity(left,right),0.0)
        self.assertGreater(sfi.team_name_similarity("Brazil U20 (W)","Brazil U20 Women"),.9)

    def test_changed_team_form_refreshes_even_before_short_ttl(self):
        game = self._game("Millwall", "Opponent")
        event = self._event("Millwall FC", "Opponent Club")
        matched = sfi.match_allsports_game(game, [event])
        self.assertIsNotNone(matched)
        now = 1_800_000_000
        self.assertEqual(sfi._save_match_link(self.db_path, game, matched, now), 2)

        # Conteúdo novo vence o TTL: manter LLLLL como WWWDD por sete dias
        # produziria exatamente a leitura antiga que o radar deve evitar.
        event["home_team"]["perf"] = "LLLLL"
        self.assertEqual(sfi._save_match_link(self.db_path, game, matched, now + 3600), 1)
        features = sfi.get_soccer_context_features(
            self.db_path, game["ID"], "Millwall", "Opponent", "League"
        )
        self.assertEqual(features["context_sfi_available"], 1.0)
        self.assertEqual(features["form_sfi_home_ppg"], 0.0)
        self.assertFalse(any("odd" in key.lower() for key in features))

    def test_lower_confidence_alias_cannot_replace_verified_provider(self):
        with sfi._db(self.db_path) as conn:
            conn.execute(
                """INSERT INTO soccer_team_aliases
                   (source_normalized, league_normalized, source_name,
                    provider_team_id, provider_name, confidence, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sfi.normalize_team_name("West Bromwich Albion"), "league",
                 "West Bromwich Albion", "verified-id", "West Bromwich Albion",
                 0.99, 1),
            )
        game = self._game("West Bromwich Albion", "Burnley")
        event = self._event("West Brom", "Burnley", "different-provider")
        matched = sfi.match_allsports_game(game, [event])
        self.assertIsNotNone(matched)
        self.assertLess(matched["home_similarity"], 0.99)
        sfi._save_match_link(self.db_path, game, matched, 2)
        with sfi._db(self.db_path) as conn:
            row = conn.execute(
                """SELECT provider_team_id, confidence FROM soccer_team_aliases
                   WHERE source_normalized=? AND league_normalized=?""",
                (sfi.normalize_team_name("West Bromwich Albion"), "league"),
            ).fetchone()
        self.assertEqual(row[0], "verified-id")
        self.assertEqual(row[1], 0.99)

    def test_reused_match_id_with_different_teams_rejects_stale_link(self):
        game = self._game("Millwall", "Burnley", "reused-id")
        event = self._event("Millwall", "Burnley", "original")
        matched = sfi.match_allsports_game(game, [event])
        sfi._save_match_link(self.db_path, game, matched, 1_800_000_000)
        features = sfi.get_soccer_context_features(
            self.db_path, "reused-id", "Santos", "Palmeiras", "Other League"
        )
        self.assertEqual(features["context_sfi_available"], 0.0)
        self.assertEqual(features["context_sfi_link_confidence"], 0.0)

    def test_parser_accepts_current_team_and_pagination_shape(self):
        payload = {
            "pagination": [{"page": 1, "per_page": 25, "items": 1678}],
            "result": [{
                "id": "live-shape",
                "date": "2026-08-22 00:00:00",
                "teamA": {"id": "home", "name": "Patriotas FC",
                          "perf": {"l_5_matches": "DDWLL"}},
                "teamB": {"id": "away", "name": "Atletico Cali FC",
                          "perf": {"l_5_matches": "DDLLL"}},
            }],
        }
        pages, per_page = sfi._pagination(payload, 25)
        self.assertEqual((pages, per_page), (68, 25))
        event = payload["result"][0]
        self.assertEqual(sfi.event_home_team(event)["id"], "home")
        self.assertEqual(sfi.event_away_team(event)["id"], "away")
        self.assertEqual(sfi._performance_sequence(event["teamA"]), list("DDWLL"))

    def test_radar_uses_soccer_schedule_and_current_allsports_odds_by_fid(self):
        event = self._event("Home FC", "Away FC", "soccer-match")
        event["bet365_url"] = "https://www.bet365.EXT/#/AC/B1/C1/D8/E199123456/F3/I0/"

        class FakeClient:
            def fetch_window(self, start, end):
                return [event], {
                    "complete": True, "days": [], "events_downloaded": 1,
                    "events_in_window": 1, "http_requests": 0,
                    "cache_hits": 2, "pages_missing": 0, "quota": {},
                }

        odds = {"odds": {"allsports-match": {
            "fid": 199123456,
            "marketName": "Full time",
            "suspended": False,
            "choices": [
                {"name": "1", "initialFractionalValue": "1/2", "fractionalValue": "11/10"},
                {"name": "X", "initialFractionalValue": "10/1", "fractionalValue": "5/2"},
                {"name": "2", "initialFractionalValue": "1/3", "fractionalValue": "6/5"},
            ],
        }}}
        start = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
        games, meta = sfi.collect_soccer_radar_games(
            self.db_path, start, start.replace(hour=16), [odds], client=FakeClient()
        )
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["ID"], "allsports-match")
        self.assertAlmostEqual(games[0]["Odd Casa"], 2.1)
        self.assertAlmostEqual(games[0]["Odd Fora"], 2.2)
        self.assertEqual(games[0]["Odds_Source"], "AllSports atual")
        self.assertEqual(meta["linked_to_current_odds"], 1)
        self.assertEqual(meta["qualified_games"], 1)
        features = sfi.get_soccer_context_features(
            self.db_path, "allsports-match", "Home FC", "Away FC", "League"
        )
        self.assertEqual(features["context_sfi_available"], 1.0)

    def test_radar_falls_back_to_allsports_match_id_when_fid_is_missing(self):
        event = self._event("Sanfrecce Hiroshima", "Okinawa SV", "soccer-japan")
        event["championship"] = {"name": "Japan FA Cup"}
        fallback_event = self._event(
            "Sanfrecce Hiroshima", "Okinawa SV", "allsports-japan"
        )
        fallback_event["championship"] = {"name": "Japan Emperor Cup"}
        fallback_event["tournament"] = {
            "id": 901, "uniqueTournament": {"id": 90},
        }
        fallback_event["season"] = {"id": 2026}

        class FakeClient:
            def fetch_window(self, start, end):
                return [event], {
                    "complete": True, "days": [], "events_downloaded": 1,
                    "events_in_window": 1, "http_requests": 0,
                    "cache_hits": 1, "pages_missing": 0, "quota": {},
                }

        odds = {"odds": {"allsports-japan": {
            "marketName": "Full time", "suspended": False,
            "choices": [
                {"name": "1", "fractionalValue": "11/10"},
                {"name": "X", "fractionalValue": "5/2"},
                {"name": "2", "fractionalValue": "6/5"},
            ],
        }}}
        start = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
        games, meta = sfi.collect_soccer_radar_games(
            self.db_path, start, start.replace(hour=16), [odds],
            client=FakeClient(),
            allsports_fallback_loader=lambda names: {
                "events": [fallback_event],
                "_meta": {"estimated_http_requests": 3},
            },
        )
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["ID"], "allsports-japan")
        self.assertEqual(games[0]["Tournament_ID"], "901")
        self.assertEqual(games[0]["Season_ID"], "2026")
        self.assertEqual(games[0]["Unique_Tournament_ID"], "90")
        self.assertEqual(meta["missing_bet365_fid"], 1)
        self.assertEqual(meta["fallback_linked"], 1)

    def test_exact_local_league_mapping_fills_ids_without_request(self):
        with sfi._db(self.db_path) as conn:
            conn.execute("""CREATE TABLE mapeamento_ligas (
                liga_nome TEXT, tournament_id TEXT, season_id TEXT)""")
            conn.execute(
                "INSERT INTO mapeamento_ligas VALUES (?,?,?)",
                ("South Korea - K League 1", "77", "2026"),
            )
        self.assertEqual(
            sfi._resolve_local_league_mapping(self.db_path, "South Korea K League 1"),
            ("77", "2026"),
        )

    def test_ambiguous_league_mapping_is_not_guessed(self):
        with sfi._db(self.db_path) as conn:
            conn.execute("""CREATE TABLE mapeamento_ligas (
                liga_nome TEXT, tournament_id TEXT, season_id TEXT)""")
            conn.executemany(
                "INSERT INTO mapeamento_ligas VALUES (?,?,?)",
                [("Brazil Serie B", "1", "2026"), ("Brazil - Serie B", "2", "2026")],
            )
        self.assertEqual(
            sfi._resolve_local_league_mapping(self.db_path, "Brazil Serie B"), ("", "")
        )


if __name__ == "__main__":
    unittest.main()
