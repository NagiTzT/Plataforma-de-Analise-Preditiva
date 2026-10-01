import unittest
from datetime import datetime, timedelta, timezone

from pregame_archive_integrity import archive_matches_training_fixture


class PregameArchiveIntegrityTests(unittest.TestCase):
    def test_accepts_same_fixture_with_small_time_adjustment(self):
        start = int(datetime(2026, 9, 1, 18, 0, tzinfo=timezone(timedelta(hours=-3))).timestamp())
        self.assertTrue(archive_matches_training_fixture(
            "Barcelona SC", "Emelec", "2026-09-01 17:30:00",
            "Barcelona Sporting Club", "CS Emelec", start,
        ))

    def test_rejects_wrong_team_category(self):
        start = int(datetime(2026, 9, 1, 18, 0, tzinfo=timezone(timedelta(hours=-3))).timestamp())
        self.assertFalse(archive_matches_training_fixture(
            "Sunderland U21", "Emelec", "2026-09-01 18:00:00",
            "Sunderland", "Emelec", start,
        ))

    def test_rejects_rescheduled_fixture_outside_tolerance(self):
        start = int(datetime(2026, 9, 3, 18, 0, tzinfo=timezone(timedelta(hours=-3))).timestamp())
        self.assertFalse(archive_matches_training_fixture(
            "Alpha", "Beta", "2026-09-01 18:00:00",
            "Alpha FC", "Beta FC", start,
        ))


if __name__ == "__main__":
    unittest.main()
