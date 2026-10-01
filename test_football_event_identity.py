import unittest

from football_event_identity import event_matches_prediction, safe_event_timestamp


class FootballEventIdentityTests(unittest.TestCase):
    def event(self, home="Alpha", away="Beta", start=1_700_000_000, match_id="1"):
        return {"id": match_id, "startTimestamp": start,
                "homeTeam": {"name": home}, "awayTeam": {"name": away}}

    def test_accepts_aliases_of_both_ordered_teams(self):
        self.assertTrue(event_matches_prediction(
            self.event("Alpha FC", "Beta CF"), "1", "Alpha vs Beta", 1_700_000_000
        ))

    def test_rejects_provider_id_collision(self):
        self.assertFalse(event_matches_prediction(
            self.event("Other", "Beta"), "1", "Alpha vs Beta", 1_700_000_000
        ))

    def test_rejects_youth_to_senior_collision(self):
        self.assertFalse(event_matches_prediction(
            self.event("Sunderland", "Beta"), "1",
            "Sunderland U21 vs Beta", 1_700_000_000
        ))

    def test_rejects_unrelated_kickoff(self):
        self.assertFalse(event_matches_prediction(
            self.event(start=1_700_100_000), "1", "Alpha vs Beta", 1_700_000_000
        ))

    def test_numeric_string_timestamp_is_accepted(self):
        self.assertEqual(safe_event_timestamp("1700000000.0"), 1_700_000_000)

    def test_malformed_timestamp_is_zero(self):
        self.assertEqual(safe_event_timestamp("bad"), 0)


if __name__ == "__main__":
    unittest.main()
