"""Validação defensiva dos artefatos de ML armazenados no SQLite."""

from io import BytesIO
import json

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


MAX_MODEL_BYTES = 200 * 1024 * 1024


def load_model_artifact(model_blob, scaler_params_json, feature_order_json):
    """Desserializa um modelo sem alterar o banco e valida seu contrato de entrada."""
    if not isinstance(model_blob, (bytes, bytearray, memoryview)):
        raise ValueError("modelo_blob ausente ou inválido")
    model_bytes = bytes(model_blob)
    if not model_bytes:
        raise ValueError("modelo_blob vazio")
    if len(model_bytes) > MAX_MODEL_BYTES:
        raise ValueError(
            f"modelo_blob excede o limite defensivo de {MAX_MODEL_BYTES // (1024 * 1024)} MB"
        )

    feature_order = json.loads(feature_order_json)
    scaler_params = json.loads(scaler_params_json)
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError("feature_order vazio ou inválido")
    if any(not isinstance(name, str) or not name for name in feature_order):
        raise ValueError("feature_order contém nome inválido")
    if len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order contém atributos duplicados")
    if not isinstance(scaler_params, dict):
        raise ValueError("scaler_params inválido")

    mean = np.asarray(scaler_params.get("mean"), dtype=float)
    scale = np.asarray(scaler_params.get("scale"), dtype=float)
    expected = len(feature_order)
    if mean.ndim != 1 or scale.ndim != 1 or len(mean) != expected or len(scale) != expected:
        raise ValueError("dimensões do scaler não correspondem ao feature_order")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("scaler contém valores não finitos")
    if np.any(scale <= 0):
        raise ValueError("scaler contém escala nula ou negativa")

    model = joblib.load(BytesIO(model_bytes))
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("artefato não implementa predict_proba")
    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and int(model_feature_count) != expected:
        raise ValueError("quantidade de atributos do modelo diverge do feature_order")

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = np.square(scale)
    scaler.n_features_in_ = expected
    scaler.n_samples_seen_ = 1
    return model, scaler, feature_order
