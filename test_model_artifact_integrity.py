import json
import unittest
from io import BytesIO

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from model_artifact_integrity import load_model_artifact


class ModelArtifactIntegrityTests(unittest.TestCase):
    @staticmethod
    def _blob(model):
        buffer = BytesIO()
        joblib.dump(model, buffer)
        return buffer.getvalue()

    def setUp(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.5, 2.0]])
        y = np.array([0, 1, 1, 0])
        self.model = LogisticRegression().fit(x, y)

    def test_loads_consistent_artifact(self):
        model, scaler, order = load_model_artifact(
            self._blob(self.model),
            json.dumps({"mean": [0.0, 0.0], "scale": [1.0, 2.0]}),
            json.dumps(["a", "b"]),
        )
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(model.predict_proba(scaler.transform([[1.0, 2.0]])).shape, (1, 2))

    def test_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimensões"):
            load_model_artifact(
                self._blob(self.model),
                json.dumps({"mean": [0.0], "scale": [1.0]}),
                json.dumps(["a", "b"]),
            )

    def test_rejects_duplicate_feature_names(self):
        with self.assertRaisesRegex(ValueError, "duplicados"):
            load_model_artifact(
                self._blob(self.model),
                json.dumps({"mean": [0.0, 0.0], "scale": [1.0, 1.0]}),
                json.dumps(["a", "a"]),
            )

    def test_rejects_model_feature_mismatch(self):
        with self.assertRaisesRegex(ValueError, "modelo diverge"):
            load_model_artifact(
                self._blob(self.model),
                json.dumps({"mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}),
                json.dumps(["a", "b", "c"]),
            )


if __name__ == "__main__":
    unittest.main()
