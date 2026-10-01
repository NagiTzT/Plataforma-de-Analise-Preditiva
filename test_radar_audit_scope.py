import sqlite3
import unittest

from radar_audit_scope import ensure_radar_audit_schema, get_latest_radar_run_ids


class RadarAuditScopeTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            """CREATE TABLE previsoes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   match_id TEXT,
                   timestamp TEXT,
                   status_resultado TEXT DEFAULT 'PENDENTE'
               )"""
        )

    def tearDown(self):
        self.conn.close()

    def test_backfill_groups_legacy_rows_by_radar_timestamp(self):
        self.conn.executemany(
            "INSERT INTO previsoes(match_id, timestamp) VALUES (?, ?)",
            [("a", "2026-08-24 18:00:00"), ("b", "2026-08-24 18:00:00"),
             ("c", "2026-08-25 18:00:00")],
        )
        ensure_radar_audit_schema(self.conn)

        self.assertEqual(
            get_latest_radar_run_ids(self.conn, 2),
            ["legacy:2026-08-25 18:00:00", "legacy:2026-08-24 18:00:00"],
        )

    def test_returns_only_the_two_latest_new_radar_runs(self):
        ensure_radar_audit_schema(self.conn)
        self.conn.executemany(
            """INSERT INTO previsoes(match_id, timestamp, radar_run_id)
               VALUES (?, ?, ?)""",
            [("a", "2026-08-23 18:00:00", "20260823_180000"),
             ("b", "2026-08-24 18:00:00", "20260824_180000"),
             ("c", "2026-08-25 18:00:00", "20260825_180000")],
        )

        self.assertEqual(
            get_latest_radar_run_ids(self.conn, 2),
            ["20260825_180000", "20260824_180000"],
        )


if __name__ == "__main__":
    unittest.main()
