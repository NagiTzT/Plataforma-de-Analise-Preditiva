"""Coleta histórica e teste cego cronológico do contexto SofaScore.

Uso:
    python sofascore_historical_learning.py status
    python sofascore_historical_learning.py run --limit 1000
    python sofascore_historical_learning.py run
    python sofascore_historical_learning.py blind-test --promote
    python sofascore_historical_learning.py all --promote

O backfill pré-jogo usa apenas ``event/{id}/pregame-form``. O backfill detalhado
grava estatísticas finais em perfis temporais separados: a partida só pode
influenciar previsões cujo horário seja posterior ao seu término.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

from ml_evolution import MODEL_CONFIGS, _fit_calibrated
from sofascore_intelligence import (
    backfill_historical_details,
    backfill_historical_pregame,
    historical_detail_status,
    historical_backfill_status,
    get_pregame_features,
    seed_historical_score_profiles,
)


DEFAULT_DB = Path(__file__).with_name("ia_sports_v5.db")
MODEL_VERSION = 7
SOFA_PREFIXES = ("sofa_pre_", "sofa_roll_")
SAFE_DAILY_CAP = max(
    1, min(4000, int(os.getenv("SOFASCORE_SAFE_DAILY_CAP", "4000")))
)
SAFE_COOLDOWN_SECONDS = 6 * 3600


def _json_print(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def refresh_historical_rollups(db_path: str) -> dict:
    """Materializa forma detalhada anterior a cada snapshot, sem chamadas HTTP."""
    with closing(sqlite3.connect(db_path, timeout=120)) as conn:
        rows = conn.execute("""
            SELECT match_id,start_timestamp,home_name,away_name
            FROM sofascore_pregame_context
            WHERE start_timestamp IS NOT NULL AND start_timestamp>0
            ORDER BY start_timestamp,match_id
        """).fetchall()
    updates = []
    both = detailed = 0
    for index, (match_id, start_timestamp, home_name, away_name) in enumerate(rows, 1):
        features = get_pregame_features(
            db_path, match_id, str(home_name or ""), str(away_name or ""),
            int(start_timestamp),
        )
        home_games = float(features.get("sofa_roll_home_games", 0) or 0)
        away_games = float(features.get("sofa_roll_away_games", 0) or 0)
        both += int(home_games > 0 and away_games > 0)
        detailed += int(
            abs(float(features.get("sofa_roll_home_xg_for_avg", 0) or 0)) > 0
            or abs(float(features.get("sofa_roll_away_xg_for_avg", 0) or 0)) > 0
            or abs(float(features.get("sofa_roll_home_shots_on_target_for_avg", 0) or 0)) > 0
            or abs(float(features.get("sofa_roll_away_shots_on_target_for_avg", 0) or 0)) > 0
        )
        updates.append((
            json.dumps(features, ensure_ascii=False), "historical-pregame-rollup-v2",
            str(match_id),
        ))
        if len(updates) >= 250 or index == len(rows):
            with closing(sqlite3.connect(db_path, timeout=120)) as conn:
                conn.executemany("""
                    UPDATE sofascore_pregame_context
                    SET features_json=?,source_version=? WHERE match_id=?
                """, updates)
                conn.commit()
            updates.clear()
        if index % 500 == 0 or index == len(rows):
            print(f"rollups {index}/{len(rows)}", flush=True)
    return {
        "processed": len(rows), "both_teams_with_history": both,
        "both_coverage": both / len(rows) if rows else 0.0,
        "with_detailed_performance": detailed,
        "detailed_coverage": detailed / len(rows) if rows else 0.0,
    }


def run_backfill(db_path: str, limit: int = 0) -> dict:
    last_print = 0

    def progress(item: dict) -> None:
        nonlocal last_print
        processed = int(item.get("processed", 0))
        total = int(item.get("total_batch", 0))
        if processed != total and processed - last_print < 250:
            return
        last_print = processed
        eta = item.get("eta_seconds")
        eta_text = f"{eta / 60:.1f} min" if eta is not None else "n/d"
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"{processed}/{total} | formas={item.get('available', 0)} | "
            f"sem cobertura={item.get('no_coverage', 0)} | "
            f"erros={item.get('error', 0)} | "
            f"{item.get('rate_per_second', 0):.2f} jogos/s | "
            f"ETA={eta_text}",
            flush=True,
        )

    result = backfill_historical_pregame(
        db_path, limit=max(0, int(limit or 0)), progress=progress
    )
    result["database_status"] = historical_backfill_status(db_path)
    _json_print(result)
    return result


def _safe_collection_state(db_path: str) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    with closing(sqlite3.connect(db_path, timeout=60)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS sofascore_safe_backfill_control (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            blocked_until INTEGER NOT NULL DEFAULT 0,
            last_http_status INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT 0
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sofascore_safe_backfill_usage (
            usage_date TEXT PRIMARY KEY,
            http_requests INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL
        )""")
        conn.execute("""INSERT OR IGNORE INTO sofascore_safe_backfill_control
            (singleton, blocked_until, last_http_status, updated_at)
            VALUES (1,0,0,0)""")
        control = conn.execute("""SELECT blocked_until,last_http_status
            FROM sofascore_safe_backfill_control WHERE singleton=1""").fetchone()
        usage = conn.execute("""SELECT http_requests FROM sofascore_safe_backfill_usage
            WHERE usage_date=?""", (today,)).fetchone()
        conn.commit()
    return {
        "usage_date": today,
        "http_requests_today": int(usage[0] or 0) if usage else 0,
        "blocked_until": int(control[0] or 0),
        "last_http_status": int(control[1] or 0),
    }


def _reserve_safe_budget(db_path: str, usage_date: str, requested: int,
                         daily_cap: int) -> int:
    """Reserva o lote antes da coleta; concorrência/crash não excedem o teto."""
    requested = max(0, int(requested))
    daily_cap = max(1, min(SAFE_DAILY_CAP, int(daily_cap)))
    with closing(sqlite3.connect(db_path, timeout=60)) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT http_requests FROM sofascore_safe_backfill_usage
               WHERE usage_date=?""", (usage_date,),
        ).fetchone()
        used = int(row[0] or 0) if row else 0
        granted = min(requested, max(0, daily_cap - used))
        if granted:
            conn.execute("""INSERT INTO sofascore_safe_backfill_usage
                (usage_date,http_requests,updated_at) VALUES (?,?,?)
                ON CONFLICT(usage_date) DO UPDATE SET
                    http_requests=http_requests+excluded.http_requests,
                    updated_at=excluded.updated_at""",
                (usage_date, granted, int(time.time())))
        conn.commit()
    return granted


def _refund_safe_budget(db_path: str, usage_date: str, amount: int) -> None:
    amount = max(0, int(amount))
    if not amount:
        return
    with closing(sqlite3.connect(db_path, timeout=60)) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("""UPDATE sofascore_safe_backfill_usage SET
            http_requests=MAX(0,http_requests-?),updated_at=? WHERE usage_date=?""",
            (amount, int(time.time()), usage_date))
        conn.commit()


def safe_run(db_path: str, requested_limit: int = 0,
             daily_cap: int = SAFE_DAILY_CAP) -> dict:
    """Executa lote cortês, com teto diário e cooldown persistente."""
    state = _safe_collection_state(db_path)
    now = int(time.time())
    if state["blocked_until"] > now:
        result = {
            "status": "COOLDOWN",
            "retry_after_seconds": state["blocked_until"] - now,
            **state,
            "database_status": historical_backfill_status(db_path),
        }
        _json_print(result)
        return result
    daily_cap = max(1, min(SAFE_DAILY_CAP, int(daily_cap)))
    remaining_today = max(0, daily_cap - state["http_requests_today"])
    if remaining_today <= 0:
        result = {
            "status": "DAILY_CAP_REACHED", **state,
            "daily_cap": int(daily_cap),
            "database_status": historical_backfill_status(db_path),
        }
        _json_print(result)
        return result
    requested_batch = min(
        remaining_today,
        max(1, int(requested_limit)) if int(requested_limit or 0) > 0 else remaining_today,
    )
    batch_limit = _reserve_safe_budget(
        db_path, state["usage_date"], requested_batch, daily_cap
    )
    if batch_limit <= 0:
        state = _safe_collection_state(db_path)
        result = {
            "status": "DAILY_CAP_REACHED", **state,
            "daily_cap": daily_cap,
            "database_status": historical_backfill_status(db_path),
        }
        _json_print(result)
        return result
    try:
        result = run_backfill(db_path, batch_limit)
    except BaseException:
        # Reserva permanece em caso de interrupção abrupta: é a opção segura.
        raise
    used = int(result.get("http_requests", 0) or 0)
    _refund_safe_budget(db_path, state["usage_date"], max(0, batch_limit - used))
    blocked = bool(result.get("circuit_open"))
    blocked_until = now + SAFE_COOLDOWN_SECONDS if blocked else 0
    last_status = int(result.get("blocked_http_status", 0) or 0)
    with closing(sqlite3.connect(db_path, timeout=60)) as conn:
        conn.execute("""UPDATE sofascore_safe_backfill_control SET
            blocked_until=?,last_http_status=?,updated_at=? WHERE singleton=1""",
            (blocked_until, last_status, int(time.time())))
        conn.commit()
    result["safe_status"] = "COOLDOWN" if blocked else "BATCH_COMPLETE"
    result["safe_daily_cap"] = daily_cap
    result["safe_requests_today"] = _safe_collection_state(db_path)["http_requests_today"]
    if blocked:
        result["safe_retry_after_seconds"] = SAFE_COOLDOWN_SECONDS
    _json_print(result)
    return result


def _brier(y: np.ndarray, proba: np.ndarray) -> float:
    expected = np.eye(3, dtype=float)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((np.asarray(proba) - expected) ** 2, axis=1)))


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = np.argmax(proba, axis=1)
    report = {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": _brier(y, proba),
        "samples": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }
    for class_id, name in ((0, "home"), (1, "draw"), (2, "away")):
        mask = y == class_id
        report[f"recall_{name}"] = (
            float(np.mean(pred[mask] == class_id)) if np.any(mask) else 0.0
        )
    return report


def _evaluate_fixed(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, train_end: int
) -> tuple[dict, object, StandardScaler]:
    scaler = StandardScaler()
    train = scaler.fit_transform(X[:train_end])
    holdout = scaler.transform(X[train_end:])
    model = _fit_calibrated(
        train, y[:train_end], weights[:train_end], MODEL_CONFIGS["regularizado"]
    )
    return _metrics(y[train_end:], model.predict_proba(holdout)), model, scaler


def _promotion_gate(baseline: dict, enriched: dict) -> tuple[bool, str]:
    non_inferior = (
        enriched["accuracy"] >= baseline["accuracy"] - 0.003
        and enriched["log_loss"] <= baseline["log_loss"] + 0.005
        and enriched["brier"] <= baseline["brier"] + 0.005
    )
    improvement = (
        enriched["accuracy"] >= baseline["accuracy"] + 0.001
        or enriched["log_loss"] <= baseline["log_loss"] - 0.001
        or enriched["brier"] <= baseline["brier"] - 0.001
        or (
            enriched["recall_draw"] >= baseline["recall_draw"] + 0.01
            and enriched["log_loss"] <= baseline["log_loss"] + 0.002
        )
    )
    if not non_inferior:
        return False, "contexto SofaScore piorou materialmente o holdout"
    if not improvement:
        return False, "ganho no holdout não atingiu o mínimo para promoção"
    return True, "ganho confirmado no holdout cronológico intocado"


def _record_evaluation(db_path: str, report: dict) -> None:
    with closing(sqlite3.connect(db_path, timeout=60)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS ml_historical_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at TEXT NOT NULL,
            source_version TEXT NOT NULL,
            train_samples INTEGER NOT NULL,
            holdout_samples INTEGER NOT NULL,
            promoted INTEGER NOT NULL DEFAULT 0,
            report_json TEXT NOT NULL
        )""")
        conn.execute(
            """INSERT INTO ml_historical_evaluations
               (evaluated_at, source_version, train_samples, holdout_samples,
                promoted, report_json) VALUES (?,?,?,?,?,?)""",
            (
                report["evaluated_at"], "historical-pregame-archive-v1",
                report["train_samples"], report["holdout_samples"],
                int(bool(report.get("promoted"))),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        conn.commit()


def _promote(db_path: str, X: np.ndarray, y: np.ndarray, weights: np.ndarray,
             feature_order: list[str], report: dict, team_ratings: dict) -> None:
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    model = _fit_calibrated(
        X_full, y, weights, MODEL_CONFIGS["regularizado"]
    )
    buffer = BytesIO()
    joblib.dump(model, buffer, compress=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(sqlite3.connect(db_path, timeout=120)) as conn:
        # Cópia recuperável do campeão anterior antes da substituição.
        conn.execute("""CREATE TABLE IF NOT EXISTS ml_model_backups (
            id INTEGER PRIMARY KEY AUTOINCREMENT, backed_up_at TEXT NOT NULL,
            liga TEXT, data_treinamento TEXT, num_amostras INTEGER,
            modelo_blob BLOB, scaler_params TEXT, feature_order TEXT,
            acuracia REAL, log_loss REAL, roc_auc REAL, model_version INTEGER)""")
        conn.execute("""INSERT INTO ml_model_backups
            (backed_up_at, liga, data_treinamento, num_amostras, modelo_blob,
             scaler_params, feature_order, acuracia, log_loss, roc_auc, model_version)
            SELECT ?, liga, data_treinamento, num_amostras, modelo_blob,
                   scaler_params, feature_order, acuracia, log_loss, roc_auc,
                   model_version FROM modelos_ml WHERE liga='GLOBAL'""", (now,))
        scaler_params = json.dumps({
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()
        })
        enriched = report["enriched"]
        conn.execute("""INSERT OR REPLACE INTO modelos_ml
            (liga, data_treinamento, num_amostras, modelo_blob, scaler_params,
             feature_order, acuracia, log_loss, roc_auc, model_version)
            VALUES ('GLOBAL',?,?,?,?,?,?,?,?,?)""", (
                now, len(X), buffer.getvalue(), scaler_params,
                json.dumps(feature_order), enriched["accuracy"],
                enriched["log_loss"], 0.0, MODEL_VERSION,
            ))
        conn.execute("UPDATE training_data SET usado_treinamento=1 WHERE usado_treinamento=0")
        if team_ratings:
            conn.executemany(
                """INSERT OR REPLACE INTO ml_team_ratings
                   (team_name, elo, updated_at) VALUES (?,?,?)""",
                [(name, float(elo), now) for name, elo in team_ratings.items()],
            )
        conn.commit()


def blind_test(db_path: str, promote: bool = False, allow_partial: bool = False) -> dict:
    status = historical_backfill_status(db_path)
    if status["remaining"] and not allow_partial:
        raise RuntimeError(
            f"backfill incompleto: faltam {status['remaining']} partidas; "
            "execute 'run' antes do teste cego"
        )

    # Import tardio evita carregar o robô durante coleta/status.
    import robo_auto as robo

    robo.DB_NAME = db_path
    X, y, feature_order, _, weights = robo.preparar_dados_treinamento(None)
    if X is None or len(X) < 500:
        raise RuntimeError("amostras insuficientes para teste cronológico")
    sofa_indices = [
        index for index, name in enumerate(feature_order)
        if name.startswith(SOFA_PREFIXES)
    ]
    if not sofa_indices:
        raise RuntimeError("nenhuma feature histórica SofaScore passou pelo piso de cobertura")
    baseline_indices = [
        index for index, name in enumerate(feature_order)
        if not name.startswith(SOFA_PREFIXES)
    ]
    train_end = int(len(X) * 0.80)
    baseline, _, _ = _evaluate_fixed(X[:, baseline_indices], y, weights, train_end)
    enriched, _, _ = _evaluate_fixed(X, y, weights, train_end)
    gate, reason = _promotion_gate(baseline, enriched)
    available_idx = feature_order.index("sofa_pre_available")
    report = {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "holdout cronológico final de 20%; configuração regularizado fixada antes do teste",
        "total_samples": int(len(X)),
        "train_samples": int(train_end),
        "holdout_samples": int(len(X) - train_end),
        "holdout_sofa_coverage": float(np.mean(X[train_end:, available_idx] > 0)),
        "sofa_feature_count": len(sofa_indices),
        "baseline_feature_count": len(baseline_indices),
        "baseline": baseline,
        "enriched": enriched,
        "accuracy_delta": enriched["accuracy"] - baseline["accuracy"],
        "log_loss_delta": enriched["log_loss"] - baseline["log_loss"],
        "brier_delta": enriched["brier"] - baseline["brier"],
        "promotion_gate": gate,
        "promotion_reason": reason,
        "promoted": bool(promote and gate),
    }
    if promote and gate:
        _promote(
            db_path, X, y, weights, feature_order, report,
            getattr(robo, "_ultimos_elos_treino", {}),
        )
    _record_evaluation(db_path, report)
    _json_print(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "status", "run", "safe-run", "seed-profiles", "detail-run",
        "refresh-rollups", "blind-test", "all",
    ))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--daily-cap", type=int, default=SAFE_DAILY_CAP)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    db_path = str(Path(args.db).resolve())
    started = time.monotonic()
    if args.command == "status":
        _json_print({
            "pregame": historical_backfill_status(db_path),
            "details": historical_detail_status(db_path),
        })
    elif args.command == "run":
        run_backfill(db_path, args.limit)
    elif args.command == "safe-run":
        result = safe_run(db_path, args.limit, args.daily_cap)
        if (result.get("database_status") or {}).get("remaining") == 0 and args.promote:
            blind_test(db_path, promote=True)
    elif args.command == "seed-profiles":
        _json_print(seed_historical_score_profiles(db_path))
    elif args.command == "detail-run":
        _json_print(backfill_historical_details(
            db_path, limit=args.limit or 250,
            progress=lambda item: print(
                f"detalhes {item.get('processed', 0)}/{item.get('total_batch', 0)}",
                flush=True,
            ),
        ))
    elif args.command == "refresh-rollups":
        _json_print(refresh_historical_rollups(db_path))
    elif args.command == "blind-test":
        blind_test(db_path, promote=args.promote, allow_partial=args.allow_partial)
    else:
        result = run_backfill(db_path, args.limit)
        status = result.get("database_status", {})
        if status.get("remaining") and not args.allow_partial:
            print(
                "Coleta ainda incompleta; o teste cego não será executado. "
                "A fila foi preservada para retomada.",
                flush=True,
            )
            return 2
        blind_test(db_path, promote=args.promote, allow_partial=args.allow_partial)
    print(f"Tempo total: {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
