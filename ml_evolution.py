"""Aprendizagem contínua e validação temporal do modelo global.

Este módulo não conhece Streamlit, Telegram ou RapidAPI. Ele recebe matrizes já
construídas sem vazamento, compara configurações em duas janelas cronológicas e
só devolve um artefato promovível quando o último período, nunca usado para
escolher a configuração, confirma a qualidade das previsões selecionadas.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from io import BytesIO

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = 8
TARGET_INDIVIDUAL_ACCURACY = 0.50
DEFAULT_CONFIDENCE_FLOOR = 0.50
MIN_CHAMPION_COMPARISON = 30
MIN_TICKET_COMPARISON = 30
MIN_RADAR_COMPARISON = 3

MODEL_CONFIGS = {
    "estavel_v3": {
        "max_depth": 3, "learning_rate": 0.03, "n_estimators": 350,
        "subsample": 0.85, "colsample_bytree": 0.75, "gamma": 0.05,
        "reg_alpha": 0.2, "reg_lambda": 4.0, "min_child_weight": 10,
    },
    "conservador": {
        "max_depth": 2, "learning_rate": 0.025, "n_estimators": 450,
        "subsample": 0.88, "colsample_bytree": 0.70, "gamma": 0.10,
        "reg_alpha": 0.35, "reg_lambda": 6.0, "min_child_weight": 14,
    },
    "regularizado": {
        "max_depth": 3, "learning_rate": 0.02, "n_estimators": 500,
        "subsample": 0.80, "colsample_bytree": 0.70, "gamma": 0.15,
        "reg_alpha": 0.45, "reg_lambda": 7.0, "min_child_weight": 16,
    },
    "empate_calibrado": {
        "max_depth": 3, "learning_rate": 0.02, "n_estimators": 500,
        "subsample": 0.80, "colsample_bytree": 0.70, "gamma": 0.15,
        "reg_alpha": 0.45, "reg_lambda": 7.0, "min_child_weight": 16,
        "draw_weight_multiplier": 1.03,
    },
    "duas_etapas_empate": {
        "architecture": "two_stage",
        "max_depth": 2, "learning_rate": 0.025, "n_estimators": 450,
        "subsample": 0.88, "colsample_bytree": 0.70, "gamma": 0.10,
        "reg_alpha": 0.35, "reg_lambda": 6.0, "min_child_weight": 14,
    },
    "duas_etapas_competicao": {
        "architecture": "two_stage",
        "competition_calibration": True,
        "draw_weight_multiplier": 1.10,
        "max_depth": 2, "learning_rate": 0.025, "n_estimators": 450,
        "subsample": 0.88, "colsample_bytree": 0.70, "gamma": 0.10,
        "reg_alpha": 0.35, "reg_lambda": 6.0, "min_child_weight": 14,
    },
    "roteador_contextual": {
        "architecture": "stacked_router",
    },
}


def balanced_sample_weights(y, weights):
    """Balance classes using only labels visible in the current fit window."""
    labels = np.asarray(y, dtype=int)
    result = np.asarray(weights, dtype=np.float32).copy()
    if not len(labels):
        return result
    counts = np.bincount(labels, minlength=3).astype(float)
    factors = np.power(
        len(labels) / np.maximum(1.0, 3.0 * counts), 0.25
    )
    factors = np.clip(factors, 0.85, 1.20)
    return result * factors[labels].astype(np.float32)

BLEND_CONFIGS = {
    # O modelo em duas etapas melhora combinações 4/4, mas tende a ignorar
    # empates. Os blends devolvem parte da probabilidade do classificador
    # calibrado sem criar uma regra manual de pick.
    "blend_side_draw_70_30": {
        "models": ("duas_etapas_empate", "empate_calibrado"),
        "weights": (0.70, 0.30),
    },
    "blend_side_draw_55_45": {
        "models": ("duas_etapas_empate", "empate_calibrado"),
        "weights": (0.55, 0.45),
    },
    "blend_side_stable_60_40": {
        "models": ("duas_etapas_empate", "estavel_v3"),
        "weights": (0.60, 0.40),
    },
    "blend_comp_draw_70_30": {
        "models": ("duas_etapas_competicao", "empate_calibrado"),
        "weights": (0.70, 0.30),
    },
}


class TwoStageClassifier:
    """Primeiro estima empate; depois separa mandante de visitante."""

    def __init__(self, params):
        self.params = dict(params)
        self.draw_model = None
        self.side_model = None
        self.classes_ = np.asarray([0, 1, 2])

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=int)
        weights = (np.ones(len(y), dtype=np.float32) if sample_weight is None
                   else np.asarray(sample_weight, dtype=np.float32))
        binary_params = dict(self.params)
        binary_params.update(objective="binary:logistic", eval_metric="logloss")
        binary_params.pop("num_class", None)
        self.draw_model = xgb.XGBClassifier(**binary_params)
        self.draw_model.fit(X, (y == 1).astype(int), sample_weight=weights)
        side = y != 1
        self.side_model = xgb.XGBClassifier(**binary_params)
        self.side_model.fit(X[side], (y[side] == 2).astype(int),
                            sample_weight=weights[side])
        return self

    def predict_proba(self, X):
        draw = np.clip(self.draw_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        away_given_side = np.clip(
            self.side_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6
        )
        non_draw = 1.0 - draw
        return np.column_stack((
            non_draw * (1.0 - away_given_side), draw,
            non_draw * away_given_side,
        ))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _competition_group_indices(X, feature_order):
    names = (
        "friendly", "qualifier", "knockout", "cup_group", "league",
    )
    result = np.full(len(X), "league", dtype=object)
    # Prioridade impede sobreposição em nomes como "World Cup Qualifier".
    for name in reversed(names):
        key = f"competition_family_{name}"
        if key in feature_order:
            # As flags chegam padronizadas ao modelo: presença fica positiva e
            # ausência negativa. Zero cobre colunas constantes/sem informação.
            result[np.asarray(X[:, feature_order.index(key)] > 0.0)] = name
    return result


def _competition_adjustments(X, y, proba, feature_order,
                             min_samples=150, shrinkage=300.0):
    """Razões real/previsto suavizadas, aprendidas só na cauda cronológica."""
    groups = _competition_group_indices(X, feature_order)
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    global_actual = (np.bincount(y, minlength=3) + 3.0) / (len(y) + 9.0)
    global_predicted = np.mean(proba, axis=0)
    adjustments = {}
    for group in sorted(set(groups)):
        mask = groups == group
        count = int(np.sum(mask))
        if count < int(min_samples):
            continue
        actual = (np.bincount(y[mask], minlength=3)
                  + shrinkage * global_actual) / (count + shrinkage)
        predicted = (np.sum(proba[mask], axis=0)
                     + shrinkage * global_predicted) / (count + shrinkage)
        adjustments[group] = np.clip(
            actual / np.maximum(predicted, 1e-6), .80, 1.25
        ).tolist()
    return adjustments


class CompetitionCalibratedClassifier:
    def __init__(self, base_model, feature_order, adjustments):
        self.base_model = base_model
        self.feature_order = list(feature_order)
        self.adjustments = dict(adjustments)
        self.classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, X):
        proba = np.asarray(self.base_model.predict_proba(X), dtype=float)
        groups = _competition_group_indices(X, self.feature_order)
        for group, ratios in self.adjustments.items():
            mask = groups == group
            if np.any(mask):
                proba[mask] *= np.asarray(ratios, dtype=float)
        return proba / np.maximum(np.sum(proba, axis=1, keepdims=True), 1e-9)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class ProbabilityBlendClassifier:
    """Combina probabilidades de modelos treinados no mesmo corte temporal."""

    def __init__(self, models, weights):
        self.models = list(models)
        normalized = np.asarray(weights, dtype=float)
        self.weights = normalized / max(float(np.sum(normalized)), 1e-9)
        self.classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, X):
        proba = sum(
            weight * np.asarray(model.predict_proba(X), dtype=float)
            for model, weight in zip(self.models, self.weights)
        )
        return proba / np.maximum(np.sum(proba, axis=1, keepdims=True), 1e-9)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class StackedRouterClassifier:
    """Aprende qual combinação de modelos funciona em cada contexto."""

    def __init__(self, models, router, context_indices):
        self.models = list(models)
        self.router = router
        self.context_indices = list(context_indices)
        self.classes_ = np.asarray([0, 1, 2])

    def _router_matrix(self, X):
        probabilities = [
            np.asarray(model.predict_proba(X), dtype=float)
            for model in self.models
        ]
        pieces = probabilities
        if self.context_indices:
            pieces.append(np.asarray(X)[:, self.context_indices])
        return np.column_stack(pieces)

    def predict_proba(self, X):
        probabilities = np.asarray(
            self.router.predict_proba(self._router_matrix(X)), dtype=float
        )
        return probabilities / np.maximum(
            np.sum(probabilities, axis=1, keepdims=True), 1e-9
        )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def ensure_evolution_tables(conn: sqlite3.Connection, confidence_floor=50):
    conn.execute("""CREATE TABLE IF NOT EXISTS ml_evolution_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at DATETIME, finished_at DATETIME, total_samples INTEGER,
        new_samples INTEGER, status TEXT, chosen_config TEXT, reason TEXT,
        min_confidence REAL, tune_accuracy REAL, tune_selected INTEGER,
        final_accuracy REAL, final_selected INTEGER, final_coverage REAL,
        global_accuracy REAL, log_loss REAL, brier_score REAL,
        baseline_accuracy REAL, baseline_log_loss REAL, promoted INTEGER DEFAULT 0
    )""")
    run_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(ml_evolution_runs)").fetchall()
    }
    if "details_json" not in run_columns:
        conn.execute("ALTER TABLE ml_evolution_runs ADD COLUMN details_json TEXT")
    conn.execute("""CREATE TABLE IF NOT EXISTS ml_selection_policy (
        scope TEXT PRIMARY KEY, min_confidence REAL NOT NULL,
        target_accuracy REAL NOT NULL, validated_accuracy REAL,
        selected_samples INTEGER DEFAULT 0, coverage REAL DEFAULT 0,
        status TEXT NOT NULL, source TEXT, updated_at DATETIME,
        model_version INTEGER, reason TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ml_live_monitoring (
        id INTEGER PRIMARY KEY AUTOINCREMENT, evaluated_at DATETIME,
        samples INTEGER, min_confidence REAL, accuracy REAL,
        recommended_confidence REAL, status TEXT, details TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ml_model_backups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backed_up_at TEXT NOT NULL,
        liga TEXT, data_treinamento TEXT, num_amostras INTEGER,
        modelo_blob BLOB, scaler_params TEXT, feature_order TEXT,
        acuracia REAL, log_loss REAL, roc_auc REAL, model_version INTEGER
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ml_shadow_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evolution_run_id INTEGER NOT NULL,
        candidate_config TEXT NOT NULL,
        match_id TEXT NOT NULL,
        radar_run_id TEXT,
        ticket_id TEXT,
        actual_outcome INTEGER NOT NULL,
        predicted_outcome INTEGER NOT NULL,
        probabilities_json TEXT NOT NULL,
        is_champion_comparison INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE(evolution_run_id, candidate_config, match_id)
    )""")
    conn.execute("""INSERT OR IGNORE INTO ml_selection_policy
        (scope, min_confidence, target_accuracy, validated_accuracy,
         selected_samples, coverage, status, source, updated_at, model_version, reason)
        VALUES ('GLOBAL', ?, 0.50, NULL, 0, 0, 'INITIAL', 'configuracao',
                CURRENT_TIMESTAMP, ?, 'Piso conservador ainda sem nova validacao v8')""",
                 (float(confidence_floor), MODEL_VERSION))
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ml_evolution_finished ON ml_evolution_runs(finished_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ml_live_evaluated ON ml_live_monitoring(evaluated_at)")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_ml_shadow_candidate
                    ON ml_shadow_predictions(candidate_config, created_at)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_ml_shadow_radar_ticket
                    ON ml_shadow_predictions(radar_run_id, ticket_id)""")


def get_active_confidence(db_name, floor_percent=50):
    """Retorna o corte operacional; nunca permite valor abaixo do piso."""
    try:
        with closing(sqlite3.connect(db_name, timeout=30)) as conn:
            ensure_evolution_tables(conn, floor_percent)
            row = conn.execute(
                "SELECT min_confidence FROM ml_selection_policy WHERE scope='GLOBAL'"
            ).fetchone()
            conn.commit()
        return max(float(floor_percent), float(row[0]) if row else float(floor_percent))
    except (sqlite3.Error, TypeError, ValueError):
        return float(floor_percent)


def get_active_model_identity(db_name):
    """Identidade imutável do artefato que gerará as previsões do radar."""
    try:
        with closing(sqlite3.connect(db_name, timeout=30)) as conn:
            row = conn.execute("""SELECT COALESCE(model_version, 1), data_treinamento
                FROM modelos_ml WHERE liga='GLOBAL'""").fetchone()
        return (int(row[0]), str(row[1])) if row else (0, "SEM_MODELO")
    except (sqlite3.Error, TypeError, ValueError):
        return 0, "SEM_MODELO"


def _xgb_params(config):
    config = {
        key: value for key, value in config.items()
        if key not in {"draw_weight_multiplier", "architecture",
                       "competition_calibration"}
    }
    return {
        "objective": "multi:softprob", "num_class": 3,
        "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1,
        **config,
    }


def _fit_core(X, y, weights, config):
    weights = balanced_sample_weights(y, weights)
    if config.get("architecture") == "two_stage":
        model = TwoStageClassifier(_xgb_params(config))
        return model.fit(X, y, sample_weight=weights)
    model = xgb.XGBClassifier(**_xgb_params(config))
    calibrated = model
    if len(X) >= 180 and min(np.bincount(y, minlength=3)) >= 8:
        try:
            calibrated = CalibratedClassifierCV(
                model, method="sigmoid", cv=TimeSeriesSplit(n_splits=3)
            )
            calibrated.fit(X, y, sample_weight=weights)
            return calibrated
        except (ValueError, xgb.core.XGBoostError):
            pass
    model.fit(X, y, sample_weight=weights)
    return model


def _router_context_indices(feature_order):
    preferred = {
        "liga_prior_empate", "context_empate_composto",
        "context_paridade_ppg", "context_paridade_mando",
        "context_paridade_elo", "context_equilibrio_ataques",
        "context_baixa_intensidade", "temporada_expected_total",
        "temporada_attack_defense_gap", "temporada_sample_quality",
        "sofa_roll_duel_draw_signal", "sofa_roll_duel_xg_parity",
        "sofa_roll_duel_attack_gap", "live_recent_home_draw_rate",
        "live_recent_away_draw_rate", "live_recent_home_same_venue_ppg",
        "live_recent_away_same_venue_ppg",
        "competition_family_league", "competition_family_cup_group",
        "competition_family_knockout", "competition_family_qualifier",
        "competition_family_friendly",
    }
    return [index for index, name in enumerate(feature_order) if name in preferred]


def _fit_stacked_router(X, y, weights, feature_order):
    """Treina o roteador em previsões internas fora da amostra."""
    if len(X) < 1200:
        return _fit_core(X, y, weights, MODEL_CONFIGS["estavel_v3"])
    inner_end = int(len(X) * 0.80)
    component_names = (
        "estavel_v3", "empate_calibrado", "duas_etapas_empate",
    )
    context_indices = _router_context_indices(feature_order or [])
    inner_models = [
        _fit_calibrated(
            X[:inner_end], y[:inner_end], weights[:inner_end],
            MODEL_CONFIGS[name], feature_order,
        )
        for name in component_names
    ]
    meta_parts = [model.predict_proba(X[inner_end:]) for model in inner_models]
    if context_indices:
        meta_parts.append(np.asarray(X[inner_end:])[:, context_indices])
    meta_X = np.column_stack(meta_parts)
    router = LogisticRegression(
        C=0.25, max_iter=1000, solver="lbfgs", random_state=42,
    )
    router.fit(meta_X, y[inner_end:], sample_weight=weights[inner_end:])
    full_models = [
        _fit_calibrated(
            X, y, weights, MODEL_CONFIGS[name], feature_order,
        )
        for name in component_names
    ]
    return StackedRouterClassifier(full_models, router, context_indices)


def _fit_calibrated(X, y, weights, config, feature_order=None):
    if config.get("architecture") == "stacked_router":
        return _fit_stacked_router(X, y, weights, feature_order or [])
    weights = np.asarray(weights, dtype=np.float32).copy()
    draw_multiplier = float(config.get("draw_weight_multiplier", 1.0))
    if draw_multiplier != 1.0:
        weights[np.asarray(y, dtype=int) == 1] *= draw_multiplier
    if config.get("competition_calibration") and feature_order and len(X) >= 1000:
        calibration_start = int(len(X) * .80)
        probe = _fit_core(
            X[:calibration_start], y[:calibration_start],
            weights[:calibration_start], config,
        )
        probe_proba = probe.predict_proba(X[calibration_start:])
        adjustments = _competition_adjustments(
            X[calibration_start:], y[calibration_start:], probe_proba,
            feature_order,
        )
        final_model = _fit_core(X, y, weights, config)
        return CompetitionCalibratedClassifier(
            final_model, feature_order, adjustments
        )
    return _fit_core(X, y, weights, config)


def _radar_mask(X, feature_order):
    """Máscara de avaliação.

    Odds não pertencem à matriz do ML. O filtro operacional >1,99 continua no
    radar, mas a validação do classificador não pode reintroduzir mercado como
    atributo escondido.
    """
    try:
        idx_home = feature_order.index("odd_casa_prejogo")
        idx_away = feature_order.index("odd_fora_prejogo")
    except ValueError:
        return np.ones(len(X), dtype=bool)
    return (X[:, idx_home] > 1.99) & (X[:, idx_away] > 1.99)


def _brier_multiclass(y, proba):
    expected = np.eye(3, dtype=float)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((np.asarray(proba) - expected) ** 2, axis=1)))


def _selection_metrics(y, proba, radar_mask, threshold):
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-9, 1.0)
    proba /= np.maximum(np.sum(proba, axis=1, keepdims=True), 1e-9)
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    # A política operacional publica todas as classes, inclusive empates. Tirar
    # empates da métrica escondia justamente a classe que mais gerou REDs.
    selected = np.asarray(radar_mask, dtype=bool) & (confidence >= threshold)
    n_radar = int(np.sum(radar_mask))
    n_selected = int(np.sum(selected))
    selected_accuracy = (float(accuracy_score(y[selected], pred[selected]))
                         if n_selected else 0.0)
    radar_mask = np.asarray(radar_mask, dtype=bool)
    radar_y = y[radar_mask]
    radar_proba = proba[radar_mask]
    radar_pred = pred[radar_mask]
    recalls = {}
    for class_id, name in ((0, "home"), (1, "draw"), (2, "away")):
        mask = y == class_id
        recalls[f"recall_{name}"] = (
            float(np.mean(pred[mask] == class_id)) if np.any(mask) else 0.0
        )
        radar_class = radar_y == class_id
        recalls[f"radar_recall_{name}"] = (
            float(np.mean(radar_pred[radar_class] == class_id))
            if np.any(radar_class) else 0.0
        )
    radar_accuracy = (
        float(accuracy_score(radar_y, radar_pred)) if len(radar_y) else 0.0
    )
    radar_loss = (
        float(log_loss(radar_y, radar_proba, labels=[0, 1, 2]))
        if len(radar_y) else float("inf")
    )
    radar_brier = (
        _brier_multiclass(radar_y, radar_proba) if len(radar_y) else float("inf")
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": _brier_multiclass(y, proba),
        "selected_accuracy": selected_accuracy,
        "selected": n_selected,
        "radar": n_radar,
        "coverage": (n_selected / n_radar) if n_radar else 0.0,
        "threshold": float(threshold),
        "radar_accuracy": radar_accuracy,
        "radar_log_loss": radar_loss,
        "radar_brier": radar_brier,
        **recalls,
    }


def _choose_threshold(y, proba, radar_mask, floor=DEFAULT_CONFIDENCE_FLOOR,
                      target=TARGET_INDIVIDUAL_ACCURACY):
    n_radar = int(np.sum(radar_mask))
    min_selected = max(40, min(100, int(math.ceil(n_radar * 0.03))))
    reports = []
    for threshold in np.arange(float(floor), 0.751, 0.02):
        report = _selection_metrics(y, proba, radar_mask, round(float(threshold), 2))
        reports.append(report)
        if report["selected"] >= min_selected and report["selected_accuracy"] >= target:
            report["validated"] = True
            report["minimum_selected"] = min_selected
            return report
    # Não fabrica 50%: mantém o piso e marca explicitamente como não validado.
    fallback = reports[0] if reports else _selection_metrics(y, proba, radar_mask, floor)
    fallback["validated"] = False
    fallback["minimum_selected"] = min_selected
    return fallback


def _head_to_head_gate(champion: dict, challenger: dict) -> tuple[bool, str]:
    """Só aceita troca quando o desafiante não piora no mesmo conjunto novo."""
    eps = 1e-9
    non_inferior = (
        challenger["radar_accuracy"] + eps >= champion["radar_accuracy"]
        and challenger["radar_log_loss"] <= champion["radar_log_loss"] + eps
        and challenger["radar_brier"] <= champion["radar_brier"] + eps
        and challenger["radar_recall_draw"] + eps >= champion["radar_recall_draw"]
    )
    improvement = (
        challenger["radar_accuracy"] > champion["radar_accuracy"] + eps
        or challenger["radar_log_loss"] < champion["radar_log_loss"] - eps
        or challenger["radar_brier"] < champion["radar_brier"] - eps
        or challenger["radar_recall_draw"] > champion["radar_recall_draw"] + eps
    )
    if not non_inferior:
        return False, (
            "campeao venceu no conjunto novo: "
            f"acc {champion['radar_accuracy']:.2%} x {challenger['radar_accuracy']:.2%}; "
            f"loss {champion['radar_log_loss']:.4f} x {challenger['radar_log_loss']:.4f}"
        )
    if not improvement:
        return False, "desafiante não apresentou ganho mensurável sobre o campeão"
    return True, "desafiante venceu o campeão nas partidas inéditas para o modelo atual"


def _ticket_metrics(y, proba, mask, ticket_groups, radar_run_groups):
    """Mede somente bilhetes reais e completos de quatro seleções."""
    y = np.asarray(y, dtype=int)
    pred = np.argmax(np.asarray(proba, dtype=float), axis=1)
    mask = np.asarray(mask, dtype=bool)
    if ticket_groups is None or radar_run_groups is None:
        ticket_groups = [None] * len(y)
        radar_run_groups = [None] * len(y)
    if len(ticket_groups) != len(y) or len(radar_run_groups) != len(y):
        raise ValueError("grupos de radar/bilhete desalinhados")
    grouped = {}
    for index, (run_id, ticket_id) in enumerate(zip(radar_run_groups, ticket_groups)):
        run_id = str(run_id or "").strip()
        ticket_id = str(ticket_id or "").strip()
        if not run_id or not ticket_id or ticket_id.upper() == "SEM_TICKET":
            continue
        grouped.setdefault((run_id, ticket_id), []).append(index)
    per_run = {}
    complete = green = 0
    for (run_id, _), indices in grouped.items():
        # Não completa artificialmente bilhetes parcialmente presentes na janela.
        if len(indices) != 4 or not all(mask[index] for index in indices):
            continue
        is_green = all(pred[index] == y[index] for index in indices)
        complete += 1
        green += int(is_green)
        run = per_run.setdefault(run_id, {"complete": 0, "green": 0})
        run["complete"] += 1
        run["green"] += int(is_green)
    return {
        "complete_tickets": complete,
        "green_tickets": green,
        "green_rate": (green / complete) if complete else 0.0,
        "radar_runs": len(per_run),
        "per_run": per_run,
    }


def _ticket_head_to_head_gate(champion: dict, challenger: dict) -> tuple[bool, str]:
    """Exige ganho 4/4 repetido em mais de um radar antes de promover."""
    comparable = min(
        int(champion.get("complete_tickets", 0)),
        int(challenger.get("complete_tickets", 0)),
    )
    common_runs = sorted(
        set(champion.get("per_run", {})) & set(challenger.get("per_run", {}))
    )
    if comparable < MIN_TICKET_COMPARISON or len(common_runs) < MIN_RADAR_COMPARISON:
        return False, (
            "evidência 4/4 insuficiente: "
            f"{comparable}/{MIN_TICKET_COMPARISON} bilhetes completos e "
            f"{len(common_runs)}/{MIN_RADAR_COMPARISON} radares"
        )
    champion_green = int(champion.get("green_tickets", 0))
    challenger_green = int(challenger.get("green_tickets", 0))
    if challenger_green <= champion_green:
        return False, (
            "desafiante não aumentou greens 4/4: "
            f"campeão {champion_green} x desafiante {challenger_green}"
        )
    gains = losses = 0
    for run_id in common_runs:
        champion_run = champion["per_run"][run_id]["green"]
        challenger_run = challenger["per_run"][run_id]["green"]
        gains += int(challenger_run > champion_run)
        losses += int(challenger_run < champion_run)
    if gains < 2 or losses > 1:
        return False, (
            "ganho 4/4 sem consistência entre radares: "
            f"melhorou {gains}, piorou {losses}, empatou {len(common_runs)-gains-losses}"
        )
    return True, (
        f"greens 4/4 {champion_green}->{challenger_green}, "
        f"com ganho em {gains} radares e perda em {losses}"
    )


def _shadow_prediction_rows(y, candidate_probas, mask, match_ids,
                            ticket_groups, radar_run_groups,
                            comparison_mask=None):
    """Serializa previsões cegas por candidato para acompanhamento longitudinal."""
    if match_ids is None:
        return []
    size = len(y)
    if any(len(values) != size for values in
           (match_ids, ticket_groups, radar_run_groups)):
        raise ValueError("metadados das previsões sombra desalinhados")
    mask = np.asarray(mask, dtype=bool)
    comparison = (np.zeros(size, dtype=bool) if comparison_mask is None
                  else np.asarray(comparison_mask, dtype=bool))
    rows = []
    for name, probabilities in candidate_probas.items():
        probabilities = np.asarray(probabilities, dtype=float)
        predicted = np.argmax(probabilities, axis=1)
        for index in np.flatnonzero(mask):
            match_id = str(match_ids[index] or "").strip()
            if not match_id:
                continue
            rows.append({
                "candidate_config": str(name), "match_id": match_id,
                "radar_run_id": str(radar_run_groups[index] or "").strip() or None,
                "ticket_id": str(ticket_groups[index] or "").strip() or None,
                "actual_outcome": int(y[index]),
                "predicted_outcome": int(predicted[index]),
                "probabilities": [float(value) for value in probabilities[index]],
                "is_champion_comparison": int(comparison[index]),
            })
    return rows


def _filter_draw_competent(eligible, baseline_tune):
    """Mantém só modelos equivalentes que não abandonam a classe empate."""
    if not eligible:
        return eligible
    best_accuracy = max(item["tune"]["radar_accuracy"] for _, item in eligible)
    best_loss = min(item["tune"]["radar_log_loss"] for _, item in eligible)
    best_brier = min(item["tune"]["radar_brier"] for _, item in eligible)
    comparable = [
        (name, item) for name, item in eligible
        if item["tune"]["radar_accuracy"] >= best_accuracy - .01
        and item["tune"]["radar_log_loss"] <= best_loss + .006
        and item["tune"]["radar_brier"] <= best_brier + .006
    ]
    if not comparable:
        return eligible
    best_draw_recall = max(
        item["tune"]["radar_recall_draw"] for _, item in comparable
    )
    draw_floor = max(
        baseline_tune["radar_recall_draw"] - .01,
        best_draw_recall - .025,
    )
    draw_competent = [
        (name, item) for name, item in comparable
        if item["tune"]["radar_recall_draw"] >= draw_floor
    ]
    return draw_competent or eligible


def _champion_probabilities(X, feature_order, champion_artifact):
    champion_features = list(champion_artifact.get("feature_order") or [])
    scaler = champion_artifact.get("scaler_params") or {}
    means = np.asarray(scaler.get("mean") or [], dtype=np.float32)
    scales = np.asarray(scaler.get("scale") or [], dtype=np.float32)
    if not champion_features or len(means) != len(champion_features) or len(scales) != len(champion_features):
        raise ValueError("artefato campeão incompatível com seu scaler")
    current_indices = {name: index for index, name in enumerate(feature_order)}
    # Feature ausente no desafiante equivale à média do campeão (zero após o
    # scaler), não ao valor bruto zero. Isso é essencial ao remover uma variável.
    aligned = np.tile(means, (len(X), 1)).astype(np.float32)
    for champion_index, name in enumerate(champion_features):
        current_index = current_indices.get(name)
        if current_index is not None:
            aligned[:, champion_index] = X[:, current_index]
    scales = np.where(np.abs(scales) < 1e-12, 1.0, scales)
    transformed = (aligned - means) / scales
    model = joblib.load(BytesIO(champion_artifact["model_blob"]))
    return model.predict_proba(transformed)


def _apply_probability_overlay(probabilities, context_rows=None, overlay=None):
    """Evaluate the exact probability stack that production publishes."""
    base = np.asarray(probabilities, dtype=float)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("matriz de probabilidades deve ter três classes")
    if overlay is None:
        result = base.copy()
    else:
        if context_rows is None or len(context_rows) != len(base):
            raise ValueError("contexto pré-jogo desalinhado com as probabilidades")
        rows = []
        for probability, context in zip(base, context_rows):
            adjusted = overlay(probability.copy(), dict(context or {}))
            if isinstance(adjusted, dict):
                adjusted = adjusted.get("probabilities")
            row = np.asarray(adjusted, dtype=float)
            if row.shape != (3,) or not np.all(np.isfinite(row)):
                raise ValueError("overlay retornou probabilidades inválidas")
            rows.append(row)
        result = np.asarray(rows, dtype=float)
    result = np.clip(result, 1e-9, 1.0)
    result /= np.maximum(np.sum(result, axis=1, keepdims=True), 1e-9)
    return result


def evaluate_evolution(X, y, feature_order, weights, confidence_floor=0.50,
                       target_accuracy=0.50, radar_mask=None,
                       champion_artifact=None, comparison_mask=None,
                       match_ids=None, ticket_groups=None,
                       radar_run_groups=None, context_rows=None,
                       probability_overlay=None):
    """Executa seleção em tune e confirmação no holdout final cronológico."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    weights = np.asarray(weights, dtype=np.float32)
    if len(X) < 500 or set(y) != {0, 1, 2}:
        return {"promote": False, "status": "REJECTED",
                "reason": "Amostras ou classes insuficientes para validacao temporal v8"}
    if probability_overlay is not None and (
        context_rows is None or len(context_rows) != len(X)
    ):
        return {
            "promote": False, "status": "REJECTED",
            "reason": "Contexto do overlay de produção ausente/desalinhado",
        }

    train_end = int(len(X) * 0.70)
    tune_end = int(len(X) * 0.85)
    if train_end < 300 or tune_end >= len(X):
        return {"promote": False, "status": "REJECTED",
                "reason": "Janelas cronologicas insuficientes"}

    if radar_mask is None:
        operational_mask = _radar_mask(X, feature_order)
    else:
        operational_mask = np.asarray(radar_mask, dtype=bool)
        if len(operational_mask) != len(X):
            return {
                "promote": False, "status": "REJECTED",
                "reason": "Mascara operacional desalinhada com as amostras",
            }
    if int(np.sum(operational_mask[tune_end:])) < 40:
        return {
            "promote": False, "status": "REJECTED",
            "reason": "Jogos do radar insuficientes no holdout cronologico",
        }

    scaler_eval = StandardScaler()
    X_train = scaler_eval.fit_transform(X[:train_end])
    X_tune = scaler_eval.transform(X[train_end:tune_end])
    X_final = scaler_eval.transform(X[tune_end:])
    radar_tune = operational_mask[train_end:tune_end]
    radar_final = operational_mask[tune_end:]
    final_size = len(y) - tune_end
    final_match_ids = (list(match_ids[tune_end:]) if match_ids is not None
                       else [None] * final_size)
    final_ticket_groups = (list(ticket_groups[tune_end:]) if ticket_groups is not None
                           else [None] * final_size)
    final_radar_groups = (list(radar_run_groups[tune_end:])
                          if radar_run_groups is not None else [None] * final_size)
    if not all(len(values) == final_size for values in
               (final_match_ids, final_ticket_groups, final_radar_groups)):
        return {
            "promote": False, "status": "REJECTED",
            "reason": "Metadados de radar/bilhete desalinhados com as amostras",
        }

    tune_size = tune_end - train_end
    tune_ticket_groups = (
        list(ticket_groups[train_end:tune_end]) if ticket_groups is not None
        else [None] * tune_size
    )
    tune_radar_groups = (
        list(radar_run_groups[train_end:tune_end]) if radar_run_groups is not None
        else [None] * tune_size
    )
    evaluated = {}
    candidate_tune_base = {}
    candidate_final_base = {}
    candidate_tune_probas = {}
    candidate_final_probas = {}
    fitted_candidates = {}
    for name, config in MODEL_CONFIGS.items():
        model = _fit_calibrated(
            X_train, y[:train_end], weights[:train_end], config, feature_order
        )
        fitted_candidates[name] = model
        tune_base = model.predict_proba(X_tune)
        final_base = model.predict_proba(X_final)
        candidate_tune_base[name] = tune_base
        candidate_final_base[name] = final_base
        tune_proba = _apply_probability_overlay(
            tune_base,
            context_rows[train_end:tune_end] if context_rows is not None else None,
            probability_overlay,
        )
        candidate_tune_probas[name] = tune_proba
        tune = _choose_threshold(
            y[train_end:tune_end], tune_proba, radar_tune,
            floor=confidence_floor, target=target_accuracy,
        )
        tune["tickets"] = _ticket_metrics(
            y[train_end:tune_end], tune_proba, radar_tune,
            tune_ticket_groups, tune_radar_groups,
        )
        final_proba = _apply_probability_overlay(
            final_base,
            context_rows[tune_end:] if context_rows is not None else None,
            probability_overlay,
        )
        candidate_final_probas[name] = final_proba
        final = _selection_metrics(y[tune_end:], final_proba, radar_final, tune["threshold"])
        final["tickets"] = _ticket_metrics(
            y[tune_end:], final_proba, radar_final,
            final_ticket_groups, final_radar_groups,
        )
        evaluated[name] = {"tune": tune, "final": final}

    for name, blend in BLEND_CONFIGS.items():
        components = tuple(blend["models"])
        blend_weights = np.asarray(blend["weights"], dtype=float)
        blend_weights /= np.sum(blend_weights)
        tune_base = sum(
            weight * candidate_tune_base[component]
            for component, weight in zip(components, blend_weights)
        )
        final_base = sum(
            weight * candidate_final_base[component]
            for component, weight in zip(components, blend_weights)
        )
        tune_proba = _apply_probability_overlay(
            tune_base,
            context_rows[train_end:tune_end] if context_rows is not None else None,
            probability_overlay,
        )
        final_proba = _apply_probability_overlay(
            final_base,
            context_rows[tune_end:] if context_rows is not None else None,
            probability_overlay,
        )
        candidate_tune_probas[name] = tune_proba
        candidate_final_probas[name] = final_proba
        tune = _choose_threshold(
            y[train_end:tune_end], tune_proba, radar_tune,
            floor=confidence_floor, target=target_accuracy,
        )
        tune["tickets"] = _ticket_metrics(
            y[train_end:tune_end], tune_proba, radar_tune,
            tune_ticket_groups, tune_radar_groups,
        )
        final = _selection_metrics(
            y[tune_end:], final_proba, radar_final, tune["threshold"]
        )
        final["tickets"] = _ticket_metrics(
            y[tune_end:], final_proba, radar_final,
            final_ticket_groups, final_radar_groups,
        )
        evaluated[name] = {"tune": tune, "final": final}

    comparison_final_for_shadow = (
        np.asarray(comparison_mask, dtype=bool)[tune_end:] & radar_final
        if comparison_mask is not None and len(comparison_mask) == len(X)
        else np.zeros(final_size, dtype=bool)
    )
    shadow_predictions = _shadow_prediction_rows(
        y[tune_end:], candidate_final_probas, radar_final,
        final_match_ids, final_ticket_groups, final_radar_groups,
        comparison_final_for_shadow,
    )

    baseline = evaluated["estavel_v3"]
    # A configuração é escolhida no universo que realmente chega ao radar. A
    # máscara usa odds só como critério de inclusão; odds nunca entram em X.
    baseline_tune = baseline["tune"]
    eligible = [
        (name, item) for name, item in evaluated.items()
        if item["tune"]["radar_accuracy"] >= baseline_tune["radar_accuracy"] - .003
        and item["tune"]["radar_log_loss"] <= baseline_tune["radar_log_loss"] + .005
        and item["tune"]["radar_brier"] <= baseline_tune["radar_brier"] + .005
        and item["tune"]["radar_recall_draw"] >= baseline_tune["radar_recall_draw"] - .02
    ]
    if not eligible:
        eligible = [("estavel_v3", baseline)]
    # Não deixa uma pequena vantagem de accuracy esconder novamente a classe
    # empate. Primeiro identifica o melhor recall entre candidatos ainda
    # equivalentes em accuracy/calibração e elimina arquiteturas que ficam mais
    # de 2,5 p.p. atrás. O holdout e o confronto com o campeão continuam sendo
    # barreiras posteriores e independentes.
    eligible = _filter_draw_competent(eligible, baseline_tune)
    best_loss = min(item["tune"]["radar_log_loss"] for _, item in eligible)
    best_accuracy = max(item["tune"]["radar_accuracy"] for _, item in eligible)
    # Entre modelos estatisticamente equivalentes, prefere quem recupera mais
    # empates. Assim o ganho não é comprado com degradação material do conjunto.
    pareto = [
        (name, item) for name, item in eligible
        if item["tune"]["radar_log_loss"] <= best_loss + 0.002
        and item["tune"]["radar_accuracy"] >= best_accuracy - 0.003
    ]
    if pareto:
        chosen_name, chosen = max(
            pareto,
            key=lambda item: (
                item[1]["tune"].get("tickets", {}).get("green_tickets", 0),
                item[1]["tune"]["radar_accuracy"],
                item[1]["tune"]["radar_recall_draw"],
                -item[1]["tune"]["radar_log_loss"],
            ),
        )
    else:
        # Pode não haver interseção entre o campeão de accuracy e o de loss.
        # Nesse caso usa distância normalizada ao canto ideal, sem consultar o
        # holdout final (que continua totalmente cego para a escolha).
        chosen_name, chosen = max(
            eligible,
            key=lambda item: (
                -((best_accuracy - item[1]["tune"]["radar_accuracy"]) / .01)
                -((item[1]["tune"]["radar_log_loss"] - best_loss) / .01)
                + .10 * item[1]["tune"]["radar_recall_draw"]
            ),
        )

    tune = chosen["tune"]
    final = chosen["final"]
    chosen_final_proba = candidate_final_probas[chosen_name]
    non_inferior = (
        final["radar_accuracy"] >= baseline["final"]["radar_accuracy"] - 0.003
        and final["radar_log_loss"] <= baseline["final"]["radar_log_loss"] + 0.005
        and final["radar_brier"] <= baseline["final"]["radar_brier"] + 0.005
        and final["radar_recall_draw"] >= baseline["final"]["radar_recall_draw"] - 0.02
    )
    material_improvement = (
        chosen_name == "estavel_v3"
        or final["radar_accuracy"] >= baseline["final"]["radar_accuracy"] + 0.001
        or final["radar_log_loss"] <= baseline["final"]["radar_log_loss"] - 0.001
        or final["radar_brier"] <= baseline["final"]["radar_brier"] - 0.001
        or (
            final["radar_recall_draw"] >= baseline["final"]["radar_recall_draw"] + 0.01
            and final["radar_log_loss"] <= baseline["final"]["radar_log_loss"] + 0.002
        )
    )
    quality_gate = non_inferior and material_improvement
    if not quality_gate:
        reason = (
            "Desafiante rejeitado no holdout final: "
            f"acc radar={final['radar_accuracy']:.2%}, "
            f"loss radar={final['radar_log_loss']:.4f}, "
            f"recall empate radar={final['radar_recall_draw']:.2%}"
        )
        return {
            "promote": False, "status": "REJECTED", "reason": reason,
            "chosen_config": chosen_name, "tune": tune, "final": final,
            "baseline": baseline["final"], "all_candidates": evaluated,
            "shadow_predictions": shadow_predictions,
        }

    champion_comparison = None
    challenger_comparison = None
    if champion_artifact is not None:
        if comparison_mask is None or len(comparison_mask) != len(X):
            return {
                "promote": False, "status": "REJECTED",
                "reason": "Campeão preservado: máscara de partidas inéditas ausente/desalinhada",
                "chosen_config": chosen_name, "tune": tune, "final": final,
                "baseline": baseline["final"], "all_candidates": evaluated,
                "shadow_predictions": shadow_predictions,
            }
        comparison_final = (
            np.asarray(comparison_mask, dtype=bool)[tune_end:] & radar_final
        )
        comparison_samples = int(np.sum(comparison_final))
        if comparison_samples < MIN_CHAMPION_COMPARISON:
            return {
                "promote": False, "status": "REJECTED",
                "reason": (
                    "Campeão preservado: somente "
                    f"{comparison_samples}/{MIN_CHAMPION_COMPARISON} partidas inéditas "
                    "do radar disponíveis para comparação direta"
                ),
                "chosen_config": chosen_name, "tune": tune, "final": final,
                "baseline": baseline["final"], "all_candidates": evaluated,
                "shadow_predictions": shadow_predictions,
            }
        try:
            champion_proba = _champion_probabilities(
                X[tune_end:], feature_order, champion_artifact
            )
            champion_proba = _apply_probability_overlay(
                champion_proba,
                context_rows[tune_end:] if context_rows is not None else None,
                probability_overlay,
            )
        except Exception as exc:
            return {
                "promote": False, "status": "REJECTED",
                "reason": f"Campeão preservado: não foi possível avaliá-lo ({exc})",
                "chosen_config": chosen_name, "tune": tune, "final": final,
                "baseline": baseline["final"], "all_candidates": evaluated,
                "shadow_predictions": shadow_predictions,
            }
        champion_comparison = _selection_metrics(
            y[tune_end:], champion_proba, comparison_final, tune["threshold"]
        )
        challenger_comparison = _selection_metrics(
            y[tune_end:], chosen_final_proba, comparison_final, tune["threshold"]
        )
        champion_ticket_comparison = _ticket_metrics(
            y[tune_end:], champion_proba, comparison_final,
            final_ticket_groups, final_radar_groups,
        )
        challenger_ticket_comparison = _ticket_metrics(
            y[tune_end:], chosen_final_proba, comparison_final,
            final_ticket_groups, final_radar_groups,
        )
        shadow_predictions.extend(_shadow_prediction_rows(
            y[tune_end:], {"__active_champion__": champion_proba}, radar_final,
            final_match_ids, final_ticket_groups, final_radar_groups,
            comparison_final,
        ))
        head_ok, head_reason = _head_to_head_gate(
            champion_comparison, challenger_comparison
        )
        if not head_ok:
            return {
                "promote": False, "status": "REJECTED",
                "reason": f"Campeão preservado: {head_reason}",
                "chosen_config": chosen_name, "tune": tune, "final": final,
                "baseline": baseline["final"], "all_candidates": evaluated,
                "champion_comparison": champion_comparison,
                "challenger_comparison": challenger_comparison,
                "champion_ticket_comparison": champion_ticket_comparison,
                "challenger_ticket_comparison": challenger_ticket_comparison,
                "shadow_predictions": shadow_predictions,
            }
        ticket_ok, ticket_reason = _ticket_head_to_head_gate(
            champion_ticket_comparison, challenger_ticket_comparison
        )
        if not ticket_ok:
            return {
                "promote": False, "status": "REJECTED",
                "reason": f"Campeão preservado: {ticket_reason}",
                "chosen_config": chosen_name, "tune": tune, "final": final,
                "baseline": baseline["final"], "all_candidates": evaluated,
                "champion_comparison": champion_comparison,
                "challenger_comparison": challenger_comparison,
                "champion_ticket_comparison": champion_ticket_comparison,
                "challenger_ticket_comparison": challenger_ticket_comparison,
                "shadow_predictions": shadow_predictions,
            }

    # Somente após passar pelo holdout é treinado o artefato de produção em tudo.
    scaler_final = StandardScaler()
    X_full = scaler_final.fit_transform(X)
    if chosen_name in BLEND_CONFIGS:
        blend = BLEND_CONFIGS[chosen_name]
        component_models = [
            _fit_calibrated(
                X_full, y, weights, MODEL_CONFIGS[component], feature_order
            )
            for component in blend["models"]
        ]
        final_model = ProbabilityBlendClassifier(
            component_models, blend["weights"]
        )
    else:
        final_model = _fit_calibrated(
            X_full, y, weights, MODEL_CONFIGS[chosen_name], feature_order
        )
    buffer = BytesIO()
    joblib.dump(final_model, buffer, compress=True)
    return {
        "promote": True, "status": "PROMOTED",
        "reason": (
            "Aprovado no holdout cronológico e venceu o campeão nas partidas "
            "inéditas" if champion_artifact is not None else
            "Aprovado na população real do radar em tune e holdout cronológico"
        ),
        "chosen_config": chosen_name, "tune": tune, "final": final,
        "baseline": baseline["final"], "all_candidates": evaluated,
        "champion_comparison": champion_comparison,
        "challenger_comparison": challenger_comparison,
        "champion_ticket_comparison": (
            champion_ticket_comparison if champion_artifact is not None else None
        ),
        "challenger_ticket_comparison": (
            challenger_ticket_comparison if champion_artifact is not None else None
        ),
        "shadow_predictions": shadow_predictions,
        "model_blob": buffer.getvalue(),
        "scaler_params": {
            "mean": scaler_final.mean_.tolist(),
            "scale": scaler_final.scale_.tolist(),
        },
    }


def persist_evolution_result(db_name, result, total_samples, new_samples,
                             feature_order, team_ratings=None, started_at=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started_at = started_at or now
    tune = result.get("tune", {})
    final = result.get("final", {})
    baseline = result.get("baseline", {})
    with closing(sqlite3.connect(db_name, timeout=60)) as conn:
        ensure_evolution_tables(conn)
        cursor = conn.execute("""INSERT INTO ml_evolution_runs
            (started_at, finished_at, total_samples, new_samples, status,
             chosen_config, reason, min_confidence, tune_accuracy, tune_selected,
             final_accuracy, final_selected, final_coverage, global_accuracy,
             log_loss, brier_score, baseline_accuracy, baseline_log_loss,
             promoted, details_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            started_at, now, int(total_samples), int(new_samples), result.get("status"),
            result.get("chosen_config"), result.get("reason"),
            float(final.get("threshold", 0.50)) * 100,
            tune.get("selected_accuracy"), tune.get("selected"),
            final.get("selected_accuracy"), final.get("selected"), final.get("coverage"),
            final.get("radar_accuracy"), final.get("radar_log_loss"), final.get("radar_brier"),
            baseline.get("radar_accuracy"), baseline.get("radar_log_loss"), int(bool(result.get("promote"))),
            json.dumps({
                "champion_comparison": result.get("champion_comparison"),
                "challenger_comparison": result.get("challenger_comparison"),
                "champion_ticket_comparison": result.get("champion_ticket_comparison"),
                "challenger_ticket_comparison": result.get("challenger_ticket_comparison"),
                "shadow_prediction_count": len(result.get("shadow_predictions") or []),
            }, ensure_ascii=False),
        ))
        evolution_run_id = int(cursor.lastrowid)
        shadow_rows = result.get("shadow_predictions") or []
        if shadow_rows:
            conn.executemany("""INSERT OR REPLACE INTO ml_shadow_predictions
                (evolution_run_id, candidate_config, match_id, radar_run_id,
                 ticket_id, actual_outcome, predicted_outcome,
                 probabilities_json, is_champion_comparison, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", [
                (
                    evolution_run_id, row["candidate_config"], row["match_id"],
                    row.get("radar_run_id"), row.get("ticket_id"),
                    int(row["actual_outcome"]), int(row["predicted_outcome"]),
                    json.dumps(row["probabilities"]),
                    int(row.get("is_champion_comparison", 0)), now,
                )
                for row in shadow_rows
            ])
        if result.get("promote"):
            policy_validated = bool(
                tune.get("validated")
                and final.get("selected_accuracy", 0) >= TARGET_INDIVIDUAL_ACCURACY
                and final.get("selected", 0) >= tune.get("minimum_selected", 0)
            )
            policy_status = "VALIDATED" if policy_validated else "OBSERVATIONAL"
            conn.execute("""INSERT INTO ml_model_backups
                (backed_up_at, liga, data_treinamento, num_amostras,
                 modelo_blob, scaler_params, feature_order, acuracia,
                 log_loss, roc_auc, model_version)
                SELECT ?, liga, data_treinamento, num_amostras, modelo_blob,
                       scaler_params, feature_order, acuracia, log_loss,
                       roc_auc, model_version
                FROM modelos_ml WHERE liga='GLOBAL'""", (now,))
            conn.execute("""DELETE FROM ml_model_backups
                WHERE id NOT IN (
                    SELECT id FROM ml_model_backups
                    ORDER BY id DESC LIMIT 5
                )""")
            conn.execute("""INSERT OR REPLACE INTO modelos_ml
                (liga, data_treinamento, num_amostras, modelo_blob, scaler_params,
                 feature_order, acuracia, log_loss, roc_auc, model_version)
                VALUES ('GLOBAL',?,?,?,?,?,?,?,?,?)""", (
                now, int(total_samples), result["model_blob"],
                json.dumps(result["scaler_params"]), json.dumps(feature_order),
                final.get("radar_accuracy"), final.get("radar_log_loss"), 0.0, MODEL_VERSION,
            ))
            conn.execute("""INSERT OR REPLACE INTO ml_selection_policy
                (scope, min_confidence, target_accuracy, validated_accuracy,
                 selected_samples, coverage, status, source, updated_at,
                 model_version, reason)
                VALUES ('GLOBAL',?,?,?,?,?,?,'holdout_cronologico',?,?,?)""", (
                float(final["threshold"]) * 100, TARGET_INDIVIDUAL_ACCURACY,
                final["selected_accuracy"], final["selected"], final["coverage"],
                policy_status, now, MODEL_VERSION, result["reason"],
            ))
            conn.execute("UPDATE training_data SET usado_treinamento=1 WHERE usado_treinamento=0")
            if team_ratings:
                conn.executemany(
                    "INSERT OR REPLACE INTO ml_team_ratings (team_name, elo, updated_at) VALUES (?,?,?)",
                    [(team, float(elo), now) for team, elo in team_ratings.items()],
                )
        conn.commit()


def record_skipped_evolution(db_name, total_samples, new_samples, reason):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(sqlite3.connect(db_name, timeout=30)) as conn:
        ensure_evolution_tables(conn)
        conn.execute("""INSERT INTO ml_evolution_runs
            (started_at, finished_at, total_samples, new_samples, status, reason, promoted)
            VALUES (?,?,?,?, 'SKIPPED', ?, 0)""",
                     (now, now, int(total_samples), int(new_samples), str(reason)))
        conn.commit()


def monitor_live_predictions(db_name, floor_percent=50, target_accuracy=0.50,
                             min_samples=50, limit=300):
    """Confere resultados reais e só eleva o corte quando há evidência recente."""
    with closing(sqlite3.connect(db_name, timeout=30)) as conn:
        ensure_evolution_tables(conn, floor_percent)
        current = conn.execute(
            "SELECT min_confidence FROM ml_selection_policy WHERE scope='GLOBAL'"
        ).fetchone()
        current_threshold = max(float(floor_percent), float(current[0]) if current else float(floor_percent))
        model_row = conn.execute("""SELECT COALESCE(model_version, 1), data_treinamento
            FROM modelos_ml WHERE liga='GLOBAL'""").fetchone()
        active_model_id = str(model_row[1]) if model_row else "SEM_MODELO"
        rows = conn.execute("""SELECT confianca, status_resultado
            FROM previsoes
            WHERE anulado=0 AND ml_model_id=?
              AND (status_resultado='RED ❌' OR status_resultado LIKE 'GREEN ✅%')
            ORDER BY timestamp DESC LIMIT ?""", (active_model_id, int(limit))).fetchall()
        samples = [(float(conf), str(status).startswith("GREEN"))
                   for conf, status in rows if conf is not None]
        active = [green for conf, green in samples if conf >= current_threshold]
        accuracy = (sum(active) / len(active)) if active else None
        status = "INSUFFICIENT"
        recommended = current_threshold
        recommended_accuracy = accuracy
        recommended_samples = len(active)
        if len(active) >= min_samples:
            status = "HEALTHY" if accuracy >= target_accuracy else "DEGRADED"
            if status == "DEGRADED":
                for threshold in np.arange(current_threshold + 2, 76, 2):
                    subset = [green for conf, green in samples if conf >= threshold]
                    if len(subset) >= min_samples and sum(subset) / len(subset) >= target_accuracy:
                        recommended = float(threshold)
                        recommended_accuracy = sum(subset) / len(subset)
                        recommended_samples = len(subset)
                        status = "ADJUSTED"
                        break
        details = json.dumps({"available": len(samples), "active_samples": len(active),
                              "model_id": active_model_id})
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""INSERT INTO ml_live_monitoring
            (evaluated_at, samples, min_confidence, accuracy,
             recommended_confidence, status, details) VALUES (?,?,?,?,?,?,?)""",
                     (now, len(active), current_threshold, accuracy, recommended, status, details))
        if status == "ADJUSTED":
            conn.execute("""UPDATE ml_selection_policy SET min_confidence=?,
                validated_accuracy=?, selected_samples=?, status='LIVE_ADJUSTED',
                source='resultados_reais', updated_at=?,
                reason='Corte elevado apos degradacao confirmada nos resultados reais'
                WHERE scope='GLOBAL'""",
                         (recommended, recommended_accuracy, recommended_samples, now))
        conn.commit()
    return {
        "status": status, "samples": len(active), "accuracy": accuracy,
        "current_confidence": current_threshold,
        "recommended_confidence": recommended,
        "recommended_accuracy": recommended_accuracy,
        "model_id": active_model_id,
    }
