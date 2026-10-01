import ast
import unittest
from pathlib import Path

from radar_prediction_integrity import deduplicate_predictions


class RadarPredictionIntegrityTests(unittest.TestCase):
    def test_radar_does_not_shadow_module_executor_import(self):
        source = Path("robo_auto.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        radar = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "job_radar_e_analise")
        local_concurrent_imports = [
            node for node in ast.walk(radar)
            if isinstance(node, ast.ImportFrom) and node.module == "concurrent.futures"
        ]
        self.assertEqual(local_concurrent_imports, [])

    def test_same_label_different_match_ids_are_not_collapsed(self):
        rows = [
            {"ID": "1", "Confronto": "A vs B", "Confiança": 40},
            {"ID": "2", "Confronto": "A vs B", "Confiança": 60},
        ]
        self.assertEqual([row["ID"] for row in deduplicate_predictions(rows)], ["1", "2"])

    def test_same_match_id_keeps_higher_confidence(self):
        rows = [
            {"ID": "1", "Confronto": "A vs B", "Confiança": 40},
            {"ID": "1", "Confronto": "Alias A vs B", "Confiança": 55},
        ]
        result = deduplicate_predictions(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Confiança"], 55)

    def test_fallback_includes_league_and_time(self):
        rows = [
            {"Confronto": "A vs B", "Liga": "League", "Timestamp": 10},
            {"Confronto": "A vs B", "Liga": "League", "Timestamp": 20},
        ]
        self.assertEqual(len(deduplicate_predictions(rows)), 2)


if __name__ == "__main__":
    unittest.main()
