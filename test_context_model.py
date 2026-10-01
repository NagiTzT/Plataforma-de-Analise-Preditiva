import os
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from datetime import datetime, timezone
from unittest.mock import patch

import robo_auto


class ContextModelTests(unittest.TestCase):
    def test_audit_reconciles_provider_alias_error_with_frozen_side(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("""CREATE TABLE previsoes(
                match_id TEXT, ticket_id TEXT, status_resultado TEXT,
                radar_run_id TEXT, anulado INTEGER)""")
            conn.execute("""CREATE TABLE ml_prediction_snapshots(
                match_id TEXT, predicted_outcome TEXT)""")
            conn.execute("""CREATE TABLE match_postmortems(
                match_id TEXT, actual_outcome TEXT, prediction_status TEXT)""")
            conn.execute(
                "INSERT INTO previsoes VALUES ('1','ticket','RED ❌','run',0)"
            )
            conn.execute(
                "INSERT INTO ml_prediction_snapshots VALUES ('1','VISITANTE')"
            )
            conn.execute(
                "INSERT INTO match_postmortems VALUES ('1','VISITANTE','RED ❌')"
            )
            changed, tickets = robo_auto._reconciliar_status_com_snapshot(
                conn.cursor(), ["run"]
            )
            self.assertEqual(changed, 1)
            self.assertEqual(tickets, {"ticket"})
            self.assertEqual(
                conn.execute("SELECT status_resultado FROM previsoes").fetchone()[0],
                "GREEN ✅",
            )
        finally:
            conn.close()

    def test_context_source_must_cover_each_chronological_period(self):
        registros = []
        for index in range(300):
            registros.append({
                'sofa_pre_available': 1.0 if index < 100 else 0.0,
                'sofa_pre_home_ppg': 2.0 if index < 100 else 0.0,
            })
        self.assertFalse(robo_auto._context_feature_has_temporal_coverage(
            registros, 'sofa_pre_home_ppg', 20
        ))
        for index in range(0, 300, 10):
            registros[index]['sofa_pre_available'] = 1.0
        for index in range(100, 300, 10):
            registros[index]['sofa_pre_available'] = 1.0
        self.assertTrue(robo_auto._context_feature_has_temporal_coverage(
            registros, 'sofa_pre_home_ppg', 20
        ))

    def test_radar_requires_both_sides_above_199(self):
        start = int(datetime(2026, 8, 23, 12, tzinfo=timezone.utc).timestamp())
        payload = {"events": [{
            "id": 123,
            "startTimestamp": start,
            "homeTeam": {"id": 1, "name": "Casa"},
            "awayTeam": {"id": 2, "name": "Fora"},
            "tournament": {"id": 3, "name": "Liga", "category": {"name": "Brasil"}},
            "season": {"id": 4},
        }]}
        with patch.object(robo_auto, "salvar_ids_liga"):
            sem_odds = robo_auto.extrair_dados_allsports(payload, {}, start - 1, start + 1)
            odds = {"123": {"choices": [
                {"name": "1", "fractionalValue": "11/10"},
                {"name": "X", "fractionalValue": "12/5"},
                {"name": "2", "fractionalValue": "6/5"},
            ]}}
            jogos = robo_auto.extrair_dados_allsports(payload, odds, start - 1, start + 1)
        self.assertEqual(sem_odds, [])
        self.assertEqual(len(jogos), 1)
        self.assertGreater(jogos[0]["Odd Casa"], 1.99)
        self.assertGreater(jogos[0]["Odd Fora"], 1.99)

    def test_market_features_are_removed(self):
        features = robo_auto._sem_features_de_odds({
            "odd_casa_prejogo": 2.1,
            "prob_mercado_casa": 0.45,
            "form_home_10_ppg": 1.8,
        })
        self.assertEqual(features, {"form_home_10_ppg": 1.8})

    def test_style_duel_uses_form_not_odds(self):
        features = {
            "form_home_10_gf": 2.0, "form_home_10_ga": 0.8,
            "form_away_10_gf": 0.7, "form_away_10_ga": 1.7,
        }
        robo_auto._enriquecer_duelo_de_estilos(features)
        self.assertAlmostEqual(features["context_ataque_home_vs_defesa_away"], 0.3)
        self.assertGreater(features["context_encaixe_ofensivo_gap"], 0)

    def _quota_db(self):
        temp = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp.name, "quota.db")
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""CREATE TABLE rapidapi_key_usage (
                key_id TEXT, usage_date TEXT, request_count INTEGER,
                status TEXT, blocked_until REAL, last_http_status TEXT,
                PRIMARY KEY (key_id, usage_date))""")
            conn.commit()
        return temp, db_path

    def test_expired_server_reset_reactivates_key_on_same_brt_day(self):
        temp, db_path = self._quota_db()
        key = "test-key"
        usage_date = robo_auto.get_brt_time().strftime("%Y-%m-%d")
        key_id = robo_auto._rapidapi_key_id(key)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""INSERT INTO rapidapi_key_usage VALUES
                (?, ?, 100, 'exhausted', ?, '429')""",
                (key_id, usage_date, time.time() - 1))
            conn.commit()
        try:
            with patch.object(robo_auto, "DB_NAME", db_path), \
                 patch.object(robo_auto, "RAPIDAPI_KEYS", [key]), \
                 patch.object(robo_auto, "RAPIDAPI_DAILY_LIMIT", 100):
                robo_auto._key_rotation_cursor = 0
                reserved, _, _ = robo_auto._reserve_rapidapi_key()
            self.assertEqual(reserved, key)
            with closing(sqlite3.connect(db_path)) as conn:
                count, status = conn.execute(
                    "SELECT request_count, status FROM rapidapi_key_usage"
                ).fetchone()
            self.assertEqual((count, status), (1, "active"))
        finally:
            temp.cleanup()

    def test_success_response_reconciles_local_count_with_server_header(self):
        temp, db_path = self._quota_db()
        key = "test-key"
        usage_date = robo_auto.get_brt_time().strftime("%Y-%m-%d")
        key_id = robo_auto._rapidapi_key_id(key)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""INSERT INTO rapidapi_key_usage VALUES
                (?, ?, 100, 'exhausted', 0, '429')""", (key_id, usage_date))
            conn.commit()

        class Response:
            status_code = 200
            headers = {
                "x-ratelimit-requests-remaining": "40",
                "x-ratelimit-requests-limit": "100",
                "x-ratelimit-requests-reset": "3600",
            }

        try:
            with patch.object(robo_auto, "DB_NAME", db_path), \
                 patch.object(robo_auto, "RAPIDAPI_KEYS", [key]):
                robo_auto._sync_rapidapi_response(key_id, 0, Response())
            with closing(sqlite3.connect(db_path)) as conn:
                count, status, reset_at = conn.execute(
                    "SELECT request_count, status, blocked_until FROM rapidapi_key_usage"
                ).fetchone()
            self.assertEqual((count, status), (60, "active"))
            self.assertGreater(reset_at, time.time() + 3500)
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
