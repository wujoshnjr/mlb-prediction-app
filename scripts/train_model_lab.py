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

try:
    from scripts.feature_schema import MODEL_FEATURES
except Exception:
    MODEL_FEATURES = []

from scripts.model_training_common import (
    MARKET_PROBABILITY_COLUMNS,
    PIPELINE_VERSION,
    build_feature_matrix,
    build_training_frame,
    classification_metrics,
    find_market_probability_column,
    safe_float,
    time_ordered_split,
    write_json,
)


REPORT_PATH = Path("report/model_lab_report.json")
ARTIFACT_DIR = Path("data/model_lab")

MIN_SHADOW_TRAIN_SAMPLES = 80
MIN_PROMOTION_SAMPLES = 300
MIN_VALIDATION_SAMPLES = 10
MAX_ECE_FOR_CANDIDATE = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_model_result(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "trained": False,
        "skipped": True,
        "skip_reason": "",
        "train_count": 0,
        "calibration_count": 0,
        "validation_count": 0,
        "feature_count": 0,
        "features_used": [],
        "experimental_features_used": [],
        "brier": None,
        "logloss": None,
        "accuracy": None,
        "auc": None,
        "ece": None,
        "beats_market_brier": None,
        "beats_market_logloss": None,
        "beats_market_accuracy": None,
        "promotion_eligible": False,
        "promotion_blockers": [],
        "warnings": [],
        "errors": [],
    }


def _probability_from_estimator(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)

    prediction = model.predict(X)
    return np.asarray(prediction, dtype=float)


def _copy_metrics(target: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    target["brier"] = metrics.get("brier")
    target["logloss"] = metrics.get("logloss")
    target["accuracy"] = metrics.get("accuracy")
    target["auc"] = metrics.get("auc")
    target["ece"] = metrics.get("ece")
    target["warnings"].extend(metrics.get("warnings") or [])
    target["errors"].extend(metrics.get("errors") or [])


def _beats_market(model_result: Dict[str, Any], market_metrics: Optional[Dict[str, Any]]) -> None:
    if not market_metrics:
        return

    model_brier = safe_float(model_result.get("brier"))
    market_brier = safe_float(market_metrics.get("brier"))
    model_logloss = safe_float(model_result.get("logloss"))
    market_logloss = safe_float(market_metrics.get("logloss"))
    model_accuracy = safe_float(model_result.get("accuracy"))
    market_accuracy = safe_float(market_metrics.get("accuracy"))

    model_result["beats_market_brier"] = (
        model_brier is not None and market_brier is not None and model_brier < market_brier
    )
    model_result["beats_market_logloss"] = (
        model_logloss is not None and market_logloss is not None and model_logloss < market_logloss
    )
    model_result["beats_market_accuracy"] = (
        model_accuracy is not None and market_accuracy is not None and model_accuracy > market_accuracy
    )


def _apply_promotion_gate(
    model_result: Dict[str, Any],
    *,
    sample_count: int,
    validation_y: np.ndarray,
    market_baseline_available: bool,
) -> None:
    blockers: List[str] = []

    if sample_count < MIN_PROMOTION_SAMPLES:
        blockers.append(f"insufficient_samples_for_promotion: {sample_count} < {MIN_PROMOTION_SAMPLES}")

    if len(np.unique(validation_y)) < 2:
        blockers.append("validation set does not contain both classes")

    if safe_float(model_result.get("brier")) is None:
        blockers.append("brier unavailable")

    if safe_float(model_result.get("logloss")) is None:
        blockers.append("logloss unavailable")

    ece = safe_float(model_result.get("ece"))
    if ece is None:
        blockers.append("ece unavailable")
    elif ece > MAX_ECE_FOR_CANDIDATE:
        blockers.append(f"ece above candidate threshold: {ece:.4f} > {MAX_ECE_FOR_CANDIDATE}")

    critical_errors = [err for err in model_result.get("errors", []) if err]
    if critical_errors:
        blockers.append("critical model errors present")

    leakage_warnings = [
        warning for warning in model_result.get("warnings", [])
        if "leakage" in str(warning).lower()
    ]
    if leakage_warnings:
        blockers.append("leakage warning present")

    if market_baseline_available:
        beats_market = bool(model_result.get("beats_market_brier")) or bool(model_result.get("beats_market_logloss"))
        if not beats_market:
            blockers.append("does not beat market brier or logloss")

    model_result["promotion_blockers"] = blockers
    model_result["promotion_eligible"] = len(blockers) == 0


def _fit_logistic(
    split: Dict[str, Any],
    features_used: List[str],
    experimental_features_used: List[str],
    sample_count: int,
    market_metrics: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Any]]:
    result = _base_model_result("logistic_baseline")
    result.update(
        {
            "skipped": False,
            "train_count": int(split["train_count"]),
            "calibration_count": int(split["calibration_count"]),
            "validation_count": int(split["validation_count"]),
            "feature_count": len(features_used),
            "features_used": features_used,
            "experimental_features_used": experimental_features_used,
        }
    )

    if len(np.unique(split["y_train"])) < 2:
        result["skipped"] = True
        result["skip_reason"] = "train set contains only one target class"
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    try:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(split["X_train"], split["y_train"])
        prob = _probability_from_estimator(model, split["X_validation"])
        metrics = classification_metrics(split["y_validation"], prob)
        _copy_metrics(result, metrics)
        result["trained"] = True
        _beats_market(result, market_metrics)
        _apply_promotion_gate(
            result,
            sample_count=sample_count,
            validation_y=split["y_validation"],
            market_baseline_available=market_metrics is not None,
        )
        return result, model
    except Exception as exc:
        result["skipped"] = True
        result["trained"] = False
        result["skip_reason"] = str(exc)
        result["errors"].append(str(exc))
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None


def _fit_lightgbm_classifier(
    split: Dict[str, Any],
    features_used: List[str],
    experimental_features_used: List[str],
    sample_count: int,
    market_metrics: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Any]]:
    result = _base_model_result("lightgbm_classifier")
    result.update(
        {
            "skipped": False,
            "train_count": int(split["train_count"]),
            "calibration_count": int(split["calibration_count"]),
            "validation_count": int(split["validation_count"]),
            "feature_count": len(features_used),
            "features_used": features_used,
            "experimental_features_used": experimental_features_used,
        }
    )

    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = f"lightgbm import failed: {exc}"
        result["warnings"].append(result["skip_reason"])
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    if len(np.unique(split["y_train"])) < 2:
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = "train set contains only one target class"
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    try:
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
        model.fit(split["X_train"], split["y_train"])
        prob = _probability_from_estimator(model, split["X_validation"])
        metrics = classification_metrics(split["y_validation"], prob)
        _copy_metrics(result, metrics)
        result["trained"] = True
        result["skipped"] = False
        _beats_market(result, market_metrics)
        _apply_promotion_gate(
            result,
            sample_count=sample_count,
            validation_y=split["y_validation"],
            market_baseline_available=market_metrics is not None,
        )
        return result, model
    except Exception as exc:
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = str(exc)
        result["errors"].append(str(exc))
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None


def _fit_xgboost_classifier(
    split: Dict[str, Any],
    features_used: List[str],
    experimental_features_used: List[str],
    sample_count: int,
    market_metrics: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Any]]:
    result = _base_model_result("xgboost_classifier")
    result.update(
        {
            "skipped": False,
            "train_count": int(split["train_count"]),
            "calibration_count": int(split["calibration_count"]),
            "validation_count": int(split["validation_count"]),
            "feature_count": len(features_used),
            "features_used": features_used,
            "experimental_features_used": experimental_features_used,
        }
    )

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = f"xgboost import failed: {exc}"
        result["warnings"].append(result["skip_reason"])
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None
        
    if len(np.unique(split["y_train"])) < 2:
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = "train set contains only one target class"
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    try:
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
        model.fit(split["X_train"], split["y_train"])
        prob = _probability_from_estimator(model, split["X_validation"])
        metrics = classification_metrics(split["y_validation"], prob)
        _copy_metrics(result, metrics)
        result["trained"] = True
        result["skipped"] = False
        _beats_market(result, market_metrics)
        _apply_promotion_gate(
            result,
            sample_count=sample_count,
            validation_y=split["y_validation"],
            market_baseline_available=market_metrics is not None,
        )
        return result, model
    except Exception as exc:
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = str(exc)
        result["errors"].append(str(exc))
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None


def _evaluate_market_baseline(frame_validation: pd.DataFrame, y_validation: np.ndarray) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
    result = _base_model_result("market_no_vig_baseline")
    result["skipped"] = True
    result["train_count"] = 0
    result["calibration_count"] = 0
    result["validation_count"] = int(len(y_validation))

    column = find_market_probability_column(frame_validation)
    if column is None:
        result["skip_reason"] = "market no-vig home probability column missing"
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None, None

    market_prob = pd.to_numeric(frame_validation[column], errors="coerce").to_numpy(dtype=float)
    metrics = classification_metrics(y_validation, market_prob)
    _copy_metrics(result, metrics)
    result["trained"] = False
    result["skipped"] = False
    result["skip_reason"] = ""
    result["feature_count"] = 1
    result["features_used"] = [column]
    result["promotion_eligible"] = False
    result["promotion_blockers"] = ["market baseline is not a candidate model"]
    return result, metrics if metrics.get("ok") else None, column


def _fit_lightgbm_market_residual(
    split: Dict[str, Any],
    features_used: List[str],
    experimental_features_used: List[str],
    sample_count: int,
    market_metrics: Optional[Dict[str, Any]],
    market_column: Optional[str],
) -> Tuple[Dict[str, Any], Optional[Any]]:
    result = _base_model_result("lightgbm_market_residual")
    result.update(
        {
            "train_count": int(split["train_count"]),
            "calibration_count": int(split["calibration_count"]),
            "validation_count": int(split["validation_count"]),
            "feature_count": len(features_used) + (1 if market_column else 0),
            "features_used": features_used + ([market_column] if market_column else []),
            "experimental_features_used": experimental_features_used,
        }
    )

    if market_column is None:
        result["skip_reason"] = "market probability missing"
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        result["skip_reason"] = f"lightgbm import failed: {exc}"
        result["warnings"].append(result["skip_reason"])
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    try:
        frame_train = split["frame_train"].copy()
        frame_validation = split["frame_validation"].copy()

        train_market = pd.to_numeric(
            frame_train[market_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        validation_market = pd.to_numeric(
            frame_validation[market_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        y_train = np.asarray(split["y_train"], dtype=float)
        y_validation = np.asarray(split["y_validation"], dtype=int)

        train_valid = np.isfinite(train_market) & np.isfinite(y_train)
        validation_valid = np.isfinite(validation_market) & np.isfinite(y_validation)

        dropped_train = int((~train_valid).sum())
        dropped_validation = int((~validation_valid).sum())

        if dropped_train or dropped_validation:
            result["warnings"].append(
                "dropped_invalid_market_prob_rows: "
                f"train={dropped_train}, validation={dropped_validation}"
            )

        if int(train_valid.sum()) < 20:
            result["trained"] = False
            result["skipped"] = True
            result["skip_reason"] = (
                "not enough residual train rows after market probability filtering: "
                f"{int(train_valid.sum())} < 20"
            )
            result["promotion_blockers"] = [result["skip_reason"]]
            return result, None

        if int(validation_valid.sum()) < 5:
            result["trained"] = False
            result["skipped"] = True
            result["skip_reason"] = (
                "not enough residual validation rows after market probability filtering: "
                f"{int(validation_valid.sum())} < 5"
            )
            result["promotion_blockers"] = [result["skip_reason"]]
            return result, None

        train_market = np.clip(train_market[train_valid], 0.01, 0.99)
        validation_market = np.clip(validation_market[validation_valid], 0.01, 0.99)

        y_train = y_train[train_valid]
        y_validation = y_validation[validation_valid]

        X_train = np.asarray(split["X_train"])[train_valid]
        X_validation = np.asarray(split["X_validation"])[validation_valid]

        residual_target = y_train - train_market

        X_train = np.column_stack([X_train, train_market])
        X_validation = np.column_stack([X_validation, validation_market])

        result["residual_train_count"] = int(len(y_train))
        result["residual_validation_count"] = int(len(y_validation))
        result["dropped_invalid_market_prob_rows"] = {
            "train": dropped_train,
            "validation": dropped_validation,
        }

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

        model.fit(X_train, residual_target)
        predicted_delta = model.predict(X_validation)
        final_prob = np.clip(validation_market + predicted_delta, 0.01, 0.99)

        metrics = classification_metrics(y_validation, final_prob)
        _copy_metrics(result, metrics)
        result["trained"] = True
        result["skipped"] = False
        result["skip_reason"] = ""
        _beats_market(result, market_metrics)
        _apply_promotion_gate(
            result,
            sample_count=sample_count,
            validation_y=split["y_validation"],
            market_baseline_available=market_metrics is not None,
        )
        return result, model
    except Exception as exc:
        result["skip_reason"] = str(exc)
        result["errors"].append(str(exc))
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None


def _best_model(models: List[Dict[str, Any]], metric: str, lower_is_better: bool = True) -> Optional[str]:
    candidates = []
    for model in models:
        if model.get("model_name") == "market_no_vig_baseline":
            continue
        value = safe_float(model.get(metric))
        if value is None:
            continue
        candidates.append((value, model.get("model_name")))

    if not candidates:
        return None

    candidates.sort(reverse=not lower_is_better)
    return str(candidates[0][1])


def build_report(
    *,
    snapshot_path: Path = Path("data/prediction_snapshots.csv"),
    finalized_path: Path = Path("data/finalized_games.csv"),
    report_path: Path = REPORT_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Dict[str, Any]:
    warnings: List[str] = []
    global_blockers: List[str] = []
    next_actions: List[str] = []

    training_frame_result = build_training_frame(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        pipeline_version=PIPELINE_VERSION,
    )
    warnings.extend(training_frame_result.get("warnings") or [])

    if not training_frame_result.get("ok"):
        global_blockers.append(training_frame_result.get("skip_reason") or "training frame unavailable")
        report = {
            "generated_at": _utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "sample_count": 0,
            "train_count": 0,
            "calibration_count": 0,
            "validation_count": 0,
            "models": [],
            "market_baseline_available": False,
            "best_by_brier": None,
            "best_by_logloss": None,
            "best_by_ece": None,
            "champion_candidate": None,
            "global_blockers": global_blockers,
            "warnings": warnings,
            "next_recommended_actions": [
                "Fix training frame availability before shadow model lab can run.",
            ],
        }
        write_json(report_path, report)
        return report

    frame = training_frame_result["frame"]
    sample_count = int(len(frame))

    feature_result = build_feature_matrix(frame, base_features=MODEL_FEATURES)
    warnings.extend(feature_result.get("warnings") or [])

    if not feature_result.get("ok"):
        global_blockers.append(feature_result.get("skip_reason") or "feature matrix unavailable")
        report = {
            "generated_at": _utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "sample_count": sample_count,
            "train_count": 0,
            "calibration_count": 0,
            "validation_count": 0,
            "models": [],
            "market_baseline_available": False,
            "best_by_brier": None,
            "best_by_logloss": None,
            "best_by_ece": None,
            "champion_candidate": None,
            "global_blockers": global_blockers,
            "warnings": warnings,
            "next_recommended_actions": [
                "Fix feature matrix availability.",
            ],
        }
        write_json(report_path, report)
        return report

    X = feature_result["X"]
    y = feature_result["y"]
    feature_frame = feature_result["frame"]
    features_used = list(feature_result["features_used"])
    experimental_features_used = list(feature_result["experimental_features_used"])

    split = time_ordered_split(
        X,
        y,
        feature_frame,
        min_train_samples=20,
        min_calibration_samples=5,
        min_validation_samples=MIN_VALIDATION_SAMPLES,
    )

    if not split.get("ok"):
        global_blockers.append(split.get("skip_reason") or "time split unavailable")
        train_count = int(split.get("train_count") or 0)
        calibration_count = int(split.get("calibration_count") or 0)
        validation_count = int(split.get("validation_count") or 0)
    else:
        train_count = int(split["train_count"])
        calibration_count = int(split["calibration_count"])
        validation_count = int(split["validation_count"])

    models: List[Dict[str, Any]] = []
    market_metrics: Optional[Dict[str, Any]] = None
    market_column: Optional[str] = None

    if split.get("ok"):
        market_model, market_metrics, market_column = _evaluate_market_baseline(
            split["frame_validation"],
            split["y_validation"],
        )
        models.append(market_model)

    if sample_count < MIN_SHADOW_TRAIN_SAMPLES:
        global_blockers.append(f"sample_count below shadow training threshold: {sample_count} < {MIN_SHADOW_TRAIN_SAMPLES}")
        for name in (
            "logistic_baseline",
            "lightgbm_classifier",
            "xgboost_classifier",
            "lightgbm_market_residual",
        ):
            skipped = _base_model_result(name)
            skipped["skip_reason"] = f"sample_count below shadow training threshold: {sample_count} < {MIN_SHADOW_TRAIN_SAMPLES}"
            skipped["promotion_blockers"] = [skipped["skip_reason"]]
            models.append(skipped)
    elif not split.get("ok"):
        for name in (
            "logistic_baseline",
            "lightgbm_classifier",
            "xgboost_classifier",
            "lightgbm_market_residual",
        ):
            skipped = _base_model_result(name)
            skipped["skip_reason"] = split.get("skip_reason") or "time split unavailable"
            skipped["promotion_blockers"] = [skipped["skip_reason"]]
            models.append(skipped)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        logistic_result, logistic_model = _fit_logistic(
            split,
            features_used,
            experimental_features_used,
            sample_count,
            market_metrics,
        )
        models.append(logistic_result)
        if logistic_model is not None:
            joblib.dump(
                {
                    "model": logistic_model,
                    "model_name": "logistic_baseline",
                    "features": features_used,
                    "pipeline_version": PIPELINE_VERSION,
                    "shadow_only": True,
                    "trained_at": _utc_now(),
                    "training_sample_count": sample_count,
                },
                artifact_dir / "logistic_baseline.pkl",
            )

        lgb_result, lgb_model = _fit_lightgbm_classifier(
            split,
            features_used,
            experimental_features_used,
            sample_count,
            market_metrics,
        )
        models.append(lgb_result)
        if lgb_model is not None:
            joblib.dump(
                {
                    "model": lgb_model,
                    "model_name": "lightgbm_classifier",
                    "features": features_used,
                    "pipeline_version": PIPELINE_VERSION,
                    "shadow_only": True,
                    "trained_at": _utc_now(),
                    "training_sample_count": sample_count,
                },
                artifact_dir / "lightgbm_classifier.pkl",
            )

        xgb_result, xgb_model = _fit_xgboost_classifier(
            split,
            features_used,
            experimental_features_used,
            sample_count,
            market_metrics,
        )
        models.append(xgb_result)
        if xgb_model is not None:
            joblib.dump(
                {
                    "model": xgb_model,
                    "model_name": "xgboost_classifier",
                    "features": features_used,
                    "pipeline_version": PIPELINE_VERSION,
                    "shadow_only": True,
                    "trained_at": _utc_now(),
                    "training_sample_count": sample_count,
                },
                artifact_dir / "xgboost_classifier.pkl",
            )

        residual_result, residual_model = _fit_lightgbm_market_residual(
            split,
            features_used,
            experimental_features_used,
            sample_count,
            market_metrics,
            market_column,
        )
        models.append(residual_result)
        if residual_model is not None:
            joblib.dump(
                {
                    "model": residual_model,
                    "model_name": "lightgbm_market_residual",
                    "features": features_used + ([market_column] if market_column else []),
                    "market_probability_column": market_column,
                    "pipeline_version": PIPELINE_VERSION,
                    "shadow_only": True,
                    "trained_at": _utc_now(),
                    "training_sample_count": sample_count,
                },
                artifact_dir / "lightgbm_market_residual.pkl",
            )

    best_by_brier = _best_model(models, "brier", lower_is_better=True)
    best_by_logloss = _best_model(models, "logloss", lower_is_better=True)
    best_by_ece = _best_model(models, "ece", lower_is_better=True)

    eligible = [model for model in models if model.get("promotion_eligible")]
    champion_candidate = None
    if eligible:
        eligible_sorted = sorted(
            eligible,
            key=lambda item: (
                safe_float(item.get("brier")) if safe_float(item.get("brier")) is not None else 999,
                safe_float(item.get("logloss")) if safe_float(item.get("logloss")) is not None else 999,
            ),
        )
        champion_candidate = eligible_sorted[0].get("model_name")

    if sample_count < MIN_PROMOTION_SAMPLES:
        global_blockers.append(f"insufficient_samples_for_promotion: {sample_count} < {MIN_PROMOTION_SAMPLES}")

    if market_metrics is None:
        warnings.append("market baseline unavailable or invalid")

    next_actions.extend(
        [
            "Keep all model lab outputs in shadow research mode.",
            "Do not replace production prediction with shadow challengers.",
            "Accumulate at least 300 finalized-joined samples before considering champion candidates.",
            "Accumulate at least 300 rolling OOS predictions before promotion review.",
        ]
    )

    report = {
        "generated_at": _utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "sample_count": sample_count,
        "train_count": train_count,
        "calibration_count": calibration_count,
        "validation_count": validation_count,
        "models": models,
        "market_baseline_available": market_metrics is not None,
        "market_probability_column": market_column,
        "best_by_brier": best_by_brier,
        "best_by_logloss": best_by_logloss,
        "best_by_ece": best_by_ece,
        "champion_candidate": champion_candidate,
        "global_blockers": sorted(set(global_blockers)),
        "warnings": sorted(set(warnings)),
        "next_recommended_actions": next_actions,
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
