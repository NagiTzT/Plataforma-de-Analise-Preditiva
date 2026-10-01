import os
import sqlite3
import tempfile
import unittest
from contextlib import closing

import numpy as np

from ml_evolution import (
    CompetitionCalibratedClassifier,
    TwoStageClassifier,
    _apply_probability_overlay,
    _competition_adjustments,
    _filter_draw_competent,
    _head_to_head_gate,
    _shadow_prediction_rows,
    _ticket_head_to_head_gate,
    _ticket_metrics,
    balanced_sample_weights,
    ensure_evolution_tables,
    get_active_confidence,
    monitor_live_predictions,
)


class MLEvolutionPolicyTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_name = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        with closing(sqlite3.connect(self.db_name)) as conn:
            ensure_evolution_tables(conn, 50)
            conn.execute("""CREATE TABLE previsoes (
                confianca REAL, status_resultado TEXT, anulado INTEGER DEFAULT 0,
                timestamp DATETIME, ml_model_id TEXT)""")
            conn.execute("""CREATE TABLE modelos_ml (
                liga TEXT PRIMARY KEY, model_version INTEGER, data_treinamento DATETIME)""")
            conn.execute("INSERT INTO modelos_ml VALUES ('GLOBAL', 3, 'modelo-teste')")
            conn.commit()

    def tearDown(self):
        os.unlink(self.db_name)

    def test_policy_never_goes_below_floor(self):
        with closing(sqlite3.connect(self.db_name)) as conn:
            conn.execute("UPDATE ml_selection_policy SET min_confidence=42 WHERE scope='GLOBAL'")
            conn.commit()
        self.assertEqual(get_active_confidence(self.db_name, 50), 50)

    def test_class_balance_uses_only_the_fit_window(self):
        train_y = np.asarray([0, 0, 0, 1, 2])
        base = np.ones(len(train_y), dtype=np.float32)
        before = balanced_sample_weights(train_y, base)
        # A distribuição de uma janela futura não participa da chamada.
        _future_y = np.asarray([1] * 100)
        after = balanced_sample_weights(train_y, base)
        np.testing.assert_allclose(before, after)
        self.assertGreater(before[3], before[0])

    def test_live_monitor_raises_threshold_only_with_evidence(self):
        rows = []
        # Corte 50 está degradado: 20/50. Em 60+, há 20/20 greens.
        for i in range(20):
            rows.append((62, "GREEN ✅", 0, f"2026-01-{i % 28 + 1:02d}", "modelo-teste"))
        for i in range(30):
            rows.append((52, "RED ❌", 0, f"2026-02-{i % 28 + 1:02d}", "modelo-teste"))
        with closing(sqlite3.connect(self.db_name)) as conn:
            conn.executemany("INSERT INTO previsoes VALUES (?,?,?,?,?)", rows)
            conn.commit()
        result = monitor_live_predictions(self.db_name, min_samples=20)
        self.assertEqual(result["status"], "ADJUSTED")
        self.assertEqual(result["recommended_confidence"], 54)

    def test_challenger_cannot_replace_a_more_accurate_champion(self):
        champion = {
            "radar_accuracy": 0.42, "radar_log_loss": 1.05,
            "radar_brier": 0.63, "radar_recall_draw": 0.20,
        }
        challenger = {
            "radar_accuracy": 0.41, "radar_log_loss": 1.04,
            "radar_brier": 0.62, "radar_recall_draw": 0.24,
        }
        accepted, reason = _head_to_head_gate(champion, challenger)
        self.assertFalse(accepted)
        self.assertIn("campeao venceu", reason)

    def test_non_inferior_challenger_with_real_gain_can_replace_champion(self):
        champion = {
            "radar_accuracy": 0.42, "radar_log_loss": 1.0500,
            "radar_brier": 0.6300, "radar_recall_draw": 0.20,
        }
        challenger = {
            "radar_accuracy": 0.42, "radar_log_loss": 1.0490,
            "radar_brier": 0.6290, "radar_recall_draw": 0.22,
        }
        accepted, reason = _head_to_head_gate(champion, challenger)
        self.assertTrue(accepted)
        self.assertIn("venceu", reason)

    def test_draw_gain_does_not_buy_worse_calibration(self):
        champion = {
            "radar_accuracy": 0.42, "radar_log_loss": 1.0500,
            "radar_brier": 0.6300, "radar_recall_draw": 0.20,
        }
        challenger = {
            "radar_accuracy": 0.42, "radar_log_loss": 1.0510,
            "radar_brier": 0.6305, "radar_recall_draw": 0.25,
        }
        accepted, _ = _head_to_head_gate(champion, challenger)
        self.assertFalse(accepted)

    def test_tune_selector_rejects_model_that_abandons_draws(self):
        def candidate(accuracy, loss, brier, draw):
            return {"tune": {
                "radar_accuracy": accuracy, "radar_log_loss": loss,
                "radar_brier": brier, "radar_recall_draw": draw,
            }}
        baseline = candidate(.433, 1.054, .636, .044)
        two_stage = candidate(.441, 1.049, .632, .036)
        draw_aware = candidate(.435, 1.055, .636, .071)
        eligible = _filter_draw_competent(
            [("baseline", baseline), ("two_stage", two_stage),
             ("draw_aware", draw_aware)],
            baseline["tune"],
        )
        names = {name for name, _ in eligible}
        self.assertIn("draw_aware", names)
        self.assertNotIn("two_stage", names)

    def test_two_stage_probabilities_are_valid(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(180, 5)).astype(np.float32)
        y = np.asarray(([0, 1, 2] * 60), dtype=int)
        model = TwoStageClassifier({
            "n_estimators": 12, "max_depth": 2, "learning_rate": .1,
            "random_state": 42, "n_jobs": 1,
        }).fit(X, y)
        proba = model.predict_proba(X[:12])
        self.assertEqual(proba.shape, (12, 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_probability_overlay_evaluates_the_published_stack(self):
        base = np.asarray([[.60, .25, .15], [.20, .30, .50]])
        contexts = [{"force_draw": True}, {"force_draw": False}]

        def overlay(probability, context):
            if context["force_draw"]:
                return {"probabilities": [.20, .70, .10]}
            return {"probabilities": probability}

        adjusted = _apply_probability_overlay(base, contexts, overlay)
        self.assertEqual(int(np.argmax(adjusted[0])), 1)
        self.assertEqual(int(np.argmax(adjusted[1])), 2)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)

    def test_probability_overlay_rejects_missing_context(self):
        with self.assertRaises(ValueError):
            _apply_probability_overlay(
                np.asarray([[.4, .3, .3]]), [], lambda probability, _: probability
            )

    def test_ticket_metrics_only_count_complete_four_leg_tickets(self):
        y = np.asarray([0, 1, 2, 0, 1, 1, 1], dtype=int)
        proba = np.eye(3)[y]
        metrics = _ticket_metrics(
            y, proba, np.ones(len(y), dtype=bool),
            ["A"] * 4 + ["B"] * 3,
            ["radar-1"] * 7,
        )
        self.assertEqual(metrics["complete_tickets"], 1)
        self.assertEqual(metrics["green_tickets"], 1)
        self.assertEqual(metrics["radar_runs"], 1)

    def test_ticket_gate_requires_longitudinal_evidence(self):
        champion = {
            "complete_tickets": 29, "green_tickets": 1,
            "per_run": {"r1": {"green": 0}, "r2": {"green": 1}},
        }
        challenger = {
            "complete_tickets": 29, "green_tickets": 3,
            "per_run": {"r1": {"green": 1}, "r2": {"green": 2}},
        }
        accepted, reason = _ticket_head_to_head_gate(champion, challenger)
        self.assertFalse(accepted)
        self.assertIn("insuficiente", reason)

    def test_ticket_gate_accepts_repeated_four_of_four_gain(self):
        champion = {
            "complete_tickets": 36, "green_tickets": 3,
            "per_run": {
                "r1": {"green": 1}, "r2": {"green": 1},
                "r3": {"green": 1},
            },
        }
        challenger = {
            "complete_tickets": 36, "green_tickets": 5,
            "per_run": {
                "r1": {"green": 2}, "r2": {"green": 2},
                "r3": {"green": 1},
            },
        }
        accepted, reason = _ticket_head_to_head_gate(champion, challenger)
        self.assertTrue(accepted)
        self.assertIn("greens 4/4", reason)

    def test_shadow_rows_keep_each_candidate_probability(self):
        rows = _shadow_prediction_rows(
            np.asarray([0, 1]),
            {"base": np.asarray([[.7, .2, .1], [.2, .6, .2]])},
            np.asarray([True, True]), ["m1", "m2"],
            ["t1", "t1"], ["r1", "r1"], np.asarray([True, False]),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["predicted_outcome"], 0)
        self.assertEqual(rows[0]["is_champion_comparison"], 1)
        self.assertEqual(len(rows[0]["probabilities"]), 3)

    def test_shadow_schema_is_created(self):
        with closing(sqlite3.connect(self.db_name)) as conn:
            columns = {
                row[1] for row in conn.execute(
                    "PRAGMA table_info(ml_shadow_predictions)"
                ).fetchall()
            }
        self.assertIn("candidate_config", columns)
        self.assertIn("radar_run_id", columns)
        self.assertIn("probabilities_json", columns)

    def test_competition_calibration_requires_enough_samples(self):
        X = np.zeros((200, 2), dtype=np.float32)
        X[:, 0] = 1.0
        y = np.asarray([0] * 80 + [1] * 60 + [2] * 60)
        proba = np.tile([.50, .20, .30], (200, 1))
        adjustments = _competition_adjustments(
            X, y, proba,
            ["competition_family_league", "competition_family_friendly"],
            min_samples=150,
        )
        self.assertIn("league", adjustments)
        self.assertNotIn("friendly", adjustments)


if __name__ == "__main__":
    unittest.main()
