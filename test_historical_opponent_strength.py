import os
import sqlite3
import tempfile
import unittest

from historical_opponent_strength import (
    clear_opponent_strength_cache,
    current_elo_ratings,
    historical_opponent_strengths,
)


class HistoricalOpponentStrengthTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        conn = sqlite3.connect(self.path)
        conn.execute("""CREATE TABLE training_data (
            match_id TEXT PRIMARY KEY, data_jogo TEXT, home_team TEXT,
            away_team TEXT, home_score REAL, away_score REAL)""")
        conn.executemany("INSERT INTO training_data VALUES (?,?,?,?,?,?)", [
            ("1", "2026-01-01 10:00:00", "A", "B", 1, 0),
            ("2", "2026-01-02 10:00:00", "C", "A", 0, 2),
            ("3", "2026-01-03 10:00:00", "B", "C", 1, 1),
        ])
        conn.commit()
        conn.close()
        clear_opponent_strength_cache()

    def tearDown(self):
        clear_opponent_strength_cache()
        os.unlink(self.path)

    def test_values_are_strictly_before_each_match(self):
        conn = sqlite3.connect(self.path)
        try:
            index = historical_opponent_strengths(conn)
        finally:
            conn.close()
        self.assertEqual(index["1"], (1.35, 1.35))
        self.assertAlmostEqual(index["2"][0], (3 + 1.35 * 8) / 9)
        self.assertEqual(index["2"][1], 1.35)

    def test_later_results_do_not_rewrite_earlier_strength(self):
        conn = sqlite3.connect(self.path)
        try:
            before = historical_opponent_strengths(conn)["1"]
            conn.execute("INSERT INTO training_data VALUES (?,?,?,?,?,?)",
                         ("4", "2026-01-04 10:00:00", "B", "A", 5, 0))
            conn.commit()
            after = historical_opponent_strengths(conn)["1"]
        finally:
            conn.close()
        self.assertEqual(before, after)

    def test_current_elo_advances_but_preserves_zero_sum(self):
        conn = sqlite3.connect(self.path)
        try:
            ratings = current_elo_ratings(conn)
        finally:
            conn.close()
        self.assertEqual(set(ratings), {"A", "B", "C"})
        self.assertAlmostEqual(sum(ratings.values()), 3 * 1500.0, places=6)
        self.assertGreater(ratings["A"], ratings["B"])

    def test_self_match_identity_corruption_is_ignored(self):
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("INSERT INTO training_data VALUES (?,?,?,?,?,?)",
                         ("bad", "2026-01-04 10:00:00", "D", "D", 4, 1))
            conn.commit()
            clear_opponent_strength_cache()
            index = historical_opponent_strengths(conn)
            ratings = current_elo_ratings(conn)
        finally:
            conn.close()
        self.assertNotIn("bad", index)
        self.assertNotIn("D", ratings)


if __name__ == "__main__":
    unittest.main()
