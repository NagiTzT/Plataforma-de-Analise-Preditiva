import os
import sqlite3
import tempfile
import unittest

from historical_identity import (
    clear_identity_cache,
    resolve_training_team_name,
    training_team_aliases,
)


class HistoricalIdentityTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE TABLE training_data(home_team TEXT, away_team TEXT)"
        )
        self.conn.executemany(
            "INSERT INTO training_data VALUES (?,?)",
            [
                ("Colorado Springs Switchbacks FC", "Barcelona SC Guayaquil"),
                ("Colorado Springs Switchbacks FC", "CF Pachuca"),
                ("Alpha United", "Beta City"),
                ("Sunderland", "Brazil"),
                ("Orlando City SC", "Brighton SC"),
                ("Brazil U20 Women", "Sunderland U21"),
                ("Barcelona SC  Guayaquil", "Opponent"),
            ],
        )
        self.conn.commit()
        clear_identity_cache()

    def tearDown(self):
        self.conn.close()
        clear_identity_cache()
        os.unlink(self.db_path)

    def test_normalized_and_high_confidence_aliases_are_linked(self):
        self.assertEqual(
            resolve_training_team_name(self.conn, "Barcelona Guayaquil")[0],
            "Barcelona SC Guayaquil",
        )
        name, confidence = resolve_training_team_name(
            self.conn, "Colorado Switchbacks FC"
        )
        self.assertEqual(name, "Colorado Springs Switchbacks FC")
        self.assertGreaterEqual(confidence, .90)

    def test_unknown_or_ambiguous_name_is_not_guessed(self):
        self.assertEqual(
            resolve_training_team_name(self.conn, "Unknown XYZ"),
            ("Unknown XYZ", 0.0),
        )

    def test_main_youth_reserve_and_women_squads_are_never_fused(self):
        for source in ("Orlando City II", "Brighton U21"):
            with self.subTest(source=source):
                self.assertEqual(resolve_training_team_name(self.conn, source), (source, 0.0))
        self.assertEqual(resolve_training_team_name(self.conn,"Sunderland U21")[0],"Sunderland U21")
        self.assertNotEqual(resolve_training_team_name(self.conn,"Sunderland U21")[0],"Sunderland")

    def test_same_explicit_category_can_match_alias(self):
        self.assertEqual(resolve_training_team_name(self.conn, "Brazil U20 (W)")[0],
                         "Brazil U20 Women")

    def test_exact_normalized_aliases_are_returned_for_sql_history(self):
        aliases = training_team_aliases(self.conn, "Barcelona SC Guayaquil")
        self.assertIn("Barcelona SC Guayaquil", aliases)
        self.assertIn("Barcelona SC  Guayaquil", aliases)


if __name__ == "__main__":
    unittest.main()
