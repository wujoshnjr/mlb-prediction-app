from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.model_training_common import classification_metrics, safe_float, write_json

PREDICTIONS_PATH = Path("data/walk_forward_predictions.csv")
REPORT_PATH = Path("report/shadow_ensemble_stack_report.json")
ARTIFACT_PATH = Path("data/model_lab/shadow_ensemble_stack.pkl")

MIN_STACK_ROWS = 80
MIN_PROMOTION_SAMPLES = 300
MAX_ECE = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_report(reason: str) -> Dict[str, Any]:
    report = {
        "generated_at": _utc_now(),
        "status": "partial",
        "trained": False,
        "skipped": True,
        "skip_reason": reason,
        "sample_count": 0,
        "train_count": 0,
        "calibration_count": 0,
        "validation_count": 0,
        "ensemble_methods": {},
        "recommended_shadow_ensemble": None,
        "promotion_eligible": False,
        "promotion_blockers": [reason],
        "warnings": [],
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    return report


def _read_predictions(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), "file_missing"

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    required = {"game_id", "model_name", "predicted_prob", "actual_home_win"}
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame(), f"missing columns: {sorted(missing)}"

    return frame, None


def _pivot_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["predicted_prob"] = pd.to_numeric(work["predicted_prob"], errors="coerce")
    work["actual_home_win"] = pd.to_numeric(work["actual_home_win"], errors="coerce")
    work = work.dropna(subset=["predicted_prob", "actual_home_win"])

    pivot = work.pivot_table(
        index="game_id",
        columns="model_name",
        values="predicted_prob",
        aggfunc="last",
    )

    target = work.groupby("game_id")["actual_home_win"].last()
    result = pivot.join(target.rename("actual_home_win"), how="inner").reset_index()
    result = result.dropna(subset=["actual_home_win"])
    result["actual_home_win"] = result["actual_home_win"].astype(int)
    return result


def _split(frame: pd.DataFrame) -> Dict[str, Any]:
    n = len(frame)
    train_end = int(n * 0.70)
    calibration_end = int(n * 0.85)

    return {
        "train": frame.iloc[:train_end].copy(),
        "calibration": frame.iloc[train_end:calibration_end].copy(),
        "validation": frame.iloc[calibration_end:].copy(),
    }


def _metrics_for_method(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    return classification_metrics(y_true, np.clip(prob, 0.01, 0.99))


def _weighted_average_weights(calibration: pd.DataFrame, model_columns: List[str]) -> Dict[str, float]:
    scores = {}
    y = calibration["actual_home_win"].to_numpy(dtype=int)

    for column in model_columns:
        pred = pd.to_numeric(calibration[column], errors="coerce").to_numpy(dtype=float)
        metrics = classification_metrics(y, pred)
        brier = safe_float(metrics.get("brier"))
        if brier is None or brier <= 0:
            continue
        scores[column] = 1.0 / brier

    total = sum(scores.values())
    if total <= 0:
        return {column: 1.0 / len(model_columns) for column in model_columns}

    return {column: value / total for column, value in scores.items()}


def build_report(
    *,
    predictions_path: Path = PREDICTIONS_PATH,
    report_path: Path = REPORT_PATH,
    artifact_path: Path = ARTIFACT_PATH,
) -> Dict[str, Any]:
    frame, error = _read_predictions(predictions_path)
    if error:
        report = _empty_report(f"walk-forward predictions unavailable: {error}")
        write_json(report_path, report)
        return report

    pivot = _pivot_predictions(frame)
    model_columns = [
        column
        for column in pivot.columns
        if column not in {"game_id", "actual_home_win", "market_no_vig_baseline"}
    ]

    if len(pivot) < MIN_STACK_ROWS:
        report = _empty_report(f"stack sample_count below threshold: {len(pivot)} < {MIN_STACK_ROWS}")
        report["sample_count"] = int(len(pivot))
        write_json(report_path, report)
        return report

    if len(model_columns) < 2:
        report = _empty_report("not enough model probability columns for ensemble")
        report["sample_count"] = int(len(pivot))
        write_json(report_path, report)
        return report

    pivot = pivot.dropna(subset=model_columns, how="all").reset_index(drop=True)
    splits = _split(pivot)

    train = splits["train"]
    calibration = splits["calibration"]
    validation = splits["validation"]

    if len(validation) < 5:
        report = _empty_report(f"validation split too small: {len(validation)} < 5")
        report["sample_count"] = int(len(pivot))
        write_json(report_path, report)
        return report

    methods: Dict[str, Any] = {}
    y_validation = validation["actual_home_win"].to_numpy(dtype=int)

    simple_prob = validation[model_columns].mean(axis=1).to_numpy(dtype=float)
    methods["simple_average"] = _metrics_for_method(y_validation, simple_prob)

    weights = _weighted_average_weights(calibration, model_columns)
    weighted_prob = np.zeros(len(validation), dtype=float)
    for column in model_columns:
        weighted_prob += validation[column].fillna(validation[model_columns].mean(axis=1)).to_numpy(dtype=float) * weights.get(column, 0.0)
    methods["weighted_average"] = {
        **_metrics_for_method(y_validation, weighted_prob),
        "weights": weights,
    }

    stacker_payload = None
    try:
        stack_train = pd.concat([train, calibration], axis=0).reset_index(drop=True)
        X_train = stack_train[model_columns].to_numpy(dtype=float)
        y_train = stack_train["actual_home_win"].to_numpy(dtype=int)
        X_validation = validation[model_columns].to_numpy(dtype=float)

        if len(np.unique(y_train)) >= 2:
            stacker = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
                ]
            )
            stacker.fit(X_train, y_train)
            stacking_prob = stacker.predict_proba(X_validation)[:, 1]
            methods["logistic_stacking"] = _metrics_for_method(y_validation, stacking_prob)
            stacker_payload = stacker
        else:
            methods["logistic_stacking"] = {
                "ok": False,
                "errors": ["stack train target has one class"],
            }
    except Exception as exc:
        methods["logistic_stacking"] = {
            "ok": False,
            "errors": [str(exc)],
        }

    ranked = []
    for method, metrics in methods.items():
        brier = safe_float(metrics.get("brier"))
        logloss = safe_float(metrics.get("logloss"))
        if brier is None:
            continue
        ranked.append((brier, logloss if logloss is not None else 999.0, method))
    ranked.sort()

    recommended = ranked[0][2] if ranked else None
    best_metrics = methods.get(recommended, {}) if recommended else {}

    blockers = []
    if len(pivot) < MIN_PROMOTION_SAMPLES:
        blockers.append(f"sample_count below promotion threshold: {len(pivot)} < {MIN_PROMOTION_SAMPLES}")

    ece = safe_float(best_metrics.get("ece"))
    if ece is None:
        blockers.append("best ensemble ECE unavailable")
    elif ece > MAX_ECE:
        blockers.append(f"best ensemble ECE above threshold: {ece:.4f} > {MAX_ECE}")

    promotion_eligible = len(blockers) == 0

    if stacker_payload is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": stacker_payload,
                "model_columns": model_columns,
                "shadow_only": True,
                "trained_at": _utc_now(),
                "sample_count": int(len(pivot)),
            },
            artifact_path,
        )

    report = {
        "generated_at": _utc_now(),
        "status": "ok",
        "trained": stacker_payload is not None,
        "skipped": False,
        "skip_reason": "",
        "sample_count": int(len(pivot)),
        "train_count": int(len(train)),
        "calibration_count": int(len(calibration)),
        "validation_count": int(len(validation)),
        "model_columns": model_columns,
        "ensemble_methods": methods,
        "recommended_shadow_ensemble": recommended,
        "promotion_eligible": promotion_eligible,
        "promotion_blockers": blockers,
        "warnings": [],
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    write_json(report_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
