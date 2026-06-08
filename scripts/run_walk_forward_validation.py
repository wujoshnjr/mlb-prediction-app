from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.feature_schema import MODEL_FEATURES
from scripts.model_training_common import (
    PIPELINE_VERSION,
    build_feature_matrix,
    build_training_frame,
    classification_metrics,
    find_market_probability_column,
    safe_float,
    write_json,
)

REPORT_PATH = Path("report/walk_forward_validation_report.json")
PREDICTIONS_PATH = Path("data/walk_forward_predictions.csv")

MIN_TRAIN_SAMPLES = 80
MIN_REQUIRED_OOS = 300
VALIDATION_WINDOW_SIZE = 10
STEP_SIZE = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_report(reason: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    report = {
        "generated_at": _utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "walkforward_ready": False,
        "skipped": True,
        "skip_reason": reason,
        "total_oos_predictions": 0,
        "minimum_train_samples": MIN_TRAIN_SAMPLES,
        "minimum_required_oos_predictions": MIN_REQUIRED_OOS,
        "rolling_window_size": None,
        "validation_window_size": VALIDATION_WINDOW_SIZE,
        "model_metrics": {},
        "market_metrics": {},
        "model_vs_market": {},
        "month_by_month": {},
        "favorite_underdog_split": {},
        "home_away_split": {},
        "confidence_bucket_split": {},
        "data_quality_split": {},
        "blockers": [reason],
        "warnings": warnings or [],
        "next_actions": [
            "Accumulate more finalized-joined samples.",
            "Keep all walk-forward outputs in shadow research mode.",
        ],
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    return report


def _model_names() -> List[str]:
    return [
        "logistic_baseline",
        "lightgbm_classifier",
        "xgboost_classifier",
        "market_no_vig_baseline",
        "market_residual_model",
    ]


def _fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray) -> Tuple[Optional[np.ndarray], List[str]]:
    warnings: List[str] = []

    if len(np.unique(y_train)) < 2:
        return None, ["train target contains one class"]

    try:
        if model_name == "logistic_baseline":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
                ]
            )

        elif model_name == "lightgbm_classifier":
            try:
                from lightgbm import LGBMClassifier
            except Exception as exc:
                return None, [f"lightgbm import failed: {exc}"]

            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=60,
                            learning_rate=0.035,
                            max_depth=2,
                            num_leaves=7,
                            min_child_samples=20,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0.25,
                            reg_lambda=1.5,
                            random_state=42,
                            verbose=-1,
                        ),
                    ),
                ]
            )

        elif model_name == "xgboost_classifier":
            try:
                from xgboost import XGBClassifier
            except Exception as exc:
                return None, [f"xgboost import failed: {exc}"]

            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=70,
                            max_depth=2,
                            learning_rate=0.035,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            min_child_weight=8,
                            reg_alpha=0.25,
                            reg_lambda=1.5,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            )

        else:
            return None, [f"unsupported model: {model_name}"]

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_valid)[:, 1]
        return np.clip(np.asarray(prob, dtype=float), 0.01, 0.99), warnings
    except Exception as exc:
        return None, [str(exc)]


def _market_residual_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    market_train: np.ndarray,
    X_valid: np.ndarray,
    market_valid: np.ndarray,
) -> Tuple[Optional[np.ndarray], List[str]]:
    warnings: List[str] = []

    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        return None, [f"lightgbm import failed: {exc}"]

    train_valid = np.isfinite(market_train) & np.isfinite(y_train)
    valid_valid = np.isfinite(market_valid)

    if int(train_valid.sum()) < 20:
        return None, [f"not enough residual train rows after market filtering: {int(train_valid.sum())} < 20"]

    if int(valid_valid.sum()) < 1:
        return None, ["no residual validation rows after market filtering"]

    dropped_train = int((~train_valid).sum())
    dropped_valid = int((~valid_valid).sum())

    if dropped_train or dropped_valid:
        warnings.append(f"dropped_invalid_market_prob_rows: train={dropped_train}, validation={dropped_valid}")

    try:
        train_market = np.clip(market_train[train_valid], 0.01, 0.99)
        valid_market = np.clip(market_valid, 0.01, 0.99)

        residual_target = y_train[train_valid].astype(float) - train_market

        X_train_res = np.column_stack([X_train[train_valid], train_market])
        X_valid_res = np.column_stack([X_valid, valid_market])

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=60,
                        learning_rate=0.025,
                        max_depth=2,
                        num_leaves=7,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.25,
                        reg_lambda=1.5,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train_res, residual_target)
        delta = model.predict(X_valid_res)
        prob = np.clip(valid_market + delta, 0.01, 0.99)
        return prob, warnings
    except Exception as exc:
        return None, [*warnings, str(exc)]


def _segment_metrics(predictions: pd.DataFrame, column: str) -> Dict[str, Any]:
    if predictions.empty or column not in predictions.columns:
        return {}

    result: Dict[str, Any] = {}
    for key, group in predictions.groupby(column):
        if group.empty:
            continue
        result[str(key)] = classification_metrics(group["actual_home_win"], group["predicted_prob"])
    return result


def _confidence_bucket(prob: float) -> str:
    edge = abs(float(prob) - 0.5)
    if edge < 0.05:
        return "low_45_55"
    if edge < 0.15:
        return "medium_55_65"
    return "high_65_plus"


def _build_model_vs_market(model_metrics: Dict[str, Any], market_metrics: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    market_brier = safe_float(market_metrics.get("brier"))
    market_logloss = safe_float(market_metrics.get("logloss"))
    market_accuracy = safe_float(market_metrics.get("accuracy"))

    for model_name, metrics in model_metrics.items():
        if model_name == "market_no_vig_baseline":
            continue

        brier = safe_float(metrics.get("brier"))
        logloss = safe_float(metrics.get("logloss"))
        accuracy = safe_float(metrics.get("accuracy"))

        result[model_name] = {
            "beats_market_brier": brier is not None and market_brier is not None and brier < market_brier,
            "beats_market_logloss": logloss is not None and market_logloss is not None and logloss < market_logloss,
            "beats_market_accuracy": accuracy is not None and market_accuracy is not None and accuracy > market_accuracy,
            "delta_brier": None if brier is None or market_brier is None else brier - market_brier,
            "delta_logloss": None if logloss is None or market_logloss is None else logloss - market_logloss,
            "delta_accuracy": None if accuracy is None or market_accuracy is None else accuracy - market_accuracy,
        }

    return result


def build_report(
    *,
    snapshot_path: Path = Path("data/prediction_snapshots.csv"),
    finalized_path: Path = Path("data/finalized_games.csv"),
    report_path: Path = REPORT_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    minimum_train_samples: int = MIN_TRAIN_SAMPLES,
    validation_window_size: int = VALIDATION_WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> Dict[str, Any]:
    warnings: List[str] = []
    blockers: List[str] = []

    training = build_training_frame(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        pipeline_version=PIPELINE_VERSION,
    )
    warnings.extend(training.get("warnings") or [])

    if not training.get("ok"):
        report = _empty_report(training.get("skip_reason") or "training frame unavailable", warnings)
        write_json(report_path, report)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(predictions_path, index=False)
        return report

    frame = training["frame"].copy()
    feature_result = build_feature_matrix(frame, base_features=MODEL_FEATURES)

    if not feature_result.get("ok"):
        report = _empty_report(feature_result.get("skip_reason") or "feature matrix unavailable", warnings)
        write_json(report_path, report)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(predictions_path, index=False)
        return report

    X = np.asarray(feature_result["X"], dtype=float)
    y = np.asarray(feature_result["y"], dtype=int)
    work = feature_result["frame"].reset_index(drop=True).copy()
    sample_count = int(len(y))

    if sample_count < minimum_train_samples + 5:
        report = _empty_report(
            f"sample_count below walk-forward threshold: {sample_count} < {minimum_train_samples + 5}",
            warnings,
        )
        write_json(report_path, report)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(predictions_path, index=False)
        return report

    market_column = find_market_probability_column(work)
    prediction_rows: List[Dict[str, Any]] = []
    model_warning_map: Dict[str, List[str]] = {name: [] for name in _model_names()}

    for start in range(minimum_train_samples, sample_count, step_size):
        end = min(start + validation_window_size, sample_count)
        if end <= start:
            continue

        X_train = X[:start]
        y_train = y[:start]
        X_valid = X[start:end]
        y_valid = y[start:end]
        frame_valid = work.iloc[start:end].copy()

        for model_name in ("logistic_baseline", "lightgbm_classifier", "xgboost_classifier"):
            prob, model_warnings = _fit_predict(model_name, X_train, y_train, X_valid)
            model_warning_map[model_name].extend(model_warnings)

            if prob is None:
                continue

            for row_index, (_, row) in enumerate(frame_valid.iterrows()):
                prediction_rows.append(
                    {
                        "game_id": row.get("game_id"),
                        "snapshot_time": str(row.get("_training_sort_time", "")),
                        "model_name": model_name,
                        "predicted_prob": float(prob[row_index]),
                        "actual_home_win": int(y_valid[row_index]),
                        "market_prob": None,
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "data_quality_grade": row.get("data_quality_grade", ""),
                        "confidence_bucket": _confidence_bucket(float(prob[row_index])),
                        "prediction_side": "home" if float(prob[row_index]) >= 0.5 else "away",
                    }
                )

        if market_column is not None:
            market_prob = pd.to_numeric(frame_valid[market_column], errors="coerce").to_numpy(dtype=float)
            for row_index, (_, row) in enumerate(frame_valid.iterrows()):
                if not np.isfinite(market_prob[row_index]):
                    continue
                clipped_market = float(np.clip(market_prob[row_index], 0.01, 0.99))
                prediction_rows.append(
                    {
                        "game_id": row.get("game_id"),
                        "snapshot_time": str(row.get("_training_sort_time", "")),
                        "model_name": "market_no_vig_baseline",
                        "predicted_prob": clipped_market,
                        "actual_home_win": int(y_valid[row_index]),
                        "market_prob": clipped_market,
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "data_quality_grade": row.get("data_quality_grade", ""),
                        "confidence_bucket": _confidence_bucket(clipped_market),
                        "prediction_side": "home" if clipped_market >= 0.5 else "away",
                    }
                )

            residual_prob, residual_warnings = _market_residual_predict(
                X_train,
                y_train.astype(float),
                pd.to_numeric(work.iloc[:start][market_column], errors="coerce").to_numpy(dtype=float),
                X_valid,
                pd.to_numeric(frame_valid[market_column], errors="coerce").to_numpy(dtype=float),
            )
            model_warning_map["market_residual_model"].extend(residual_warnings)

            if residual_prob is not None:
                for row_index, (_, row) in enumerate(frame_valid.iterrows()):
                    prediction_rows.append(
                        {
                            "game_id": row.get("game_id"),
                            "snapshot_time": str(row.get("_training_sort_time", "")),
                            "model_name": "market_residual_model",
                            "predicted_prob": float(residual_prob[row_index]),
                            "actual_home_win": int(y_valid[row_index]),
                            "market_prob": None,
                            "home_team": row.get("home_team", ""),
                            "away_team": row.get("away_team", ""),
                            "data_quality_grade": row.get("data_quality_grade", ""),
                            "confidence_bucket": _confidence_bucket(float(residual_prob[row_index])),
                            "prediction_side": "home" if float(residual_prob[row_index]) >= 0.5 else "away",
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    model_metrics: Dict[str, Any] = {}
    if not predictions.empty:
        for model_name, group in predictions.groupby("model_name"):
            model_metrics[str(model_name)] = classification_metrics(
                group["actual_home_win"],
                group["predicted_prob"],
            )
            model_metrics[str(model_name)]["warnings"] = sorted(
                set((model_metrics[str(model_name)].get("warnings") or []) + model_warning_map.get(str(model_name), []))
            )

    market_metrics = model_metrics.get("market_no_vig_baseline", {})
    model_vs_market = _build_model_vs_market(model_metrics, market_metrics) if market_metrics else {}

    total_oos = int(len(predictions[predictions["model_name"] != "market_no_vig_baseline"])) if not predictions.empty else 0
    walkforward_ready = total_oos >= MIN_REQUIRED_OOS

    if total_oos < MIN_REQUIRED_OOS:
        blockers.append(f"rolling OOS predictions below threshold: {total_oos} < {MIN_REQUIRED_OOS}")

    if market_metrics:
        any_beats_market = any(
            item.get("beats_market_brier") or item.get("beats_market_logloss")
            for item in model_vs_market.values()
        )
        if not any_beats_market:
            blockers.append("no shadow model beats market brier or logloss")
    else:
        warnings.append("market baseline unavailable for walk-forward comparison")

    report = {
        "generated_at": _utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "walkforward_ready": walkforward_ready,
        "skipped": False,
        "skip_reason": "",
        "total_oos_predictions": total_oos,
        "minimum_train_samples": minimum_train_samples,
        "minimum_required_oos_predictions": MIN_REQUIRED_OOS,
        "rolling_window_size": None,
        "validation_window_size": validation_window_size,
        "model_metrics": model_metrics,
        "market_metrics": market_metrics,
        "model_vs_market": model_vs_market,
        "month_by_month": {},
        "favorite_underdog_split": _segment_metrics(
            predictions[predictions["model_name"] != "market_no_vig_baseline"],
            "prediction_side",
        ) if not predictions.empty else {},
        "home_away_split": {},
        "confidence_bucket_split": _segment_metrics(
            predictions[predictions["model_name"] != "market_no_vig_baseline"],
            "confidence_bucket",
        ) if not predictions.empty else {},
        "data_quality_split": _segment_metrics(
            predictions[predictions["model_name"] != "market_no_vig_baseline"],
            "data_quality_grade",
        ) if not predictions.empty else {},
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "next_actions": [
            "Continue accumulating forward OOS samples.",
            "Do not promote any model until at least 300 OOS predictions exist.",
            "Use brier/logloss and calibration over accuracy-only conclusions.",
        ],
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
