import os
import sqlite3
import tempfile
import unittest

from operational_backtest import operational_snapshot_report


class OperationalBacktestTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        conn = sqlite3.connect(self.path)
        conn.executescript("""
            CREATE TABLE ml_prediction_snapshots (
                match_id TEXT PRIMARY KEY,predicted_outcome TEXT,source_version TEXT,
                captured_at INTEGER,start_timestamp INTEGER);
            CREATE TABLE match_postmortems (
                match_id TEXT PRIMARY KEY,actual_outcome TEXT);
            CREATE TABLE previsoes (
                id INTEGER PRIMARY KEY,radar_run_id TEXT,ticket_id TEXT,match_id TEXT);
        """)
        predictions = ["MANDANTE", "EMPATE", "VISITANTE", "MANDANTE"]
        actuals = ["MANDANTE", "EMPATE", "VISITANTE", "EMPATE"]
        for index, (predicted, actual) in enumerate(zip(predictions, actuals), 1):
            match_id = f"m{index}"
            conn.execute("INSERT INTO ml_prediction_snapshots VALUES (?,?,?,?,?)",
                         (match_id, predicted, "v5", 100, 200))
            conn.execute("INSERT INTO match_postmortems VALUES (?,?)", (match_id, actual))
            conn.execute("INSERT INTO previsoes VALUES (?,?,?,?)",
                         (index, "r1", "t1", match_id))
        # A duplicate persisted leg cannot turn the ticket into five legs.
        conn.execute("INSERT INTO previsoes VALUES (?,?,?,?)",
                     (5, "r1", "t1", "m1"))
        # Um palpite registrado após o início nunca entra no relatório honesto.
        conn.execute("INSERT INTO ml_prediction_snapshots VALUES (?,?,?,?,?)",
                     ("late", "MANDANTE", "v5", 300, 200))
        conn.execute("INSERT INTO match_postmortems VALUES (?,?)", ("late", "MANDANTE"))
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_scores_only_frozen_predictions(self):
        report = operational_snapshot_report(self.path)
        self.assertEqual(report["resolved_predictions"], 4)
        self.assertEqual(report["correct_predictions"], 3)
        self.assertEqual(report["complete_tickets"], 1)
        self.assertEqual(report["green_tickets"], 0)
        self.assertEqual(report["ticket_hits_distribution"]["3"], 1)


if __name__ == "__main__":
    unittest.main()
