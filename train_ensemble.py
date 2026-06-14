# train_ensemble.py
"""Train and calibrate the MLB baseline model from canonical training samples.

Canonical source:
- data/training_samples.csv

This file intentionally refuses to re-join prediction_snapshots.csv and
finalized_games.csv. Outcome linkage is owned by the evidence chain.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.feature_schema import (
    CORE_MODEL_FEATURES,
    MODEL_FEATURE_VERSION,
    get_model_feature_schema_hash,
)


try:
    import config
except ImportError:

    class config:  # type: ignore[no-redef]
        PIPELINE_VERSION = "baseline_v2_clean"
        ALLOW_LEGACY_TRAINING_DATA = False
        MIN_CLEAN_TRAIN_SAMPLES = 300


PIPELINE_VERSION = str(getattr(config, "PIPELINE_VERSION", "baseline_v2_clean"))
ALLOW_LEGACY_TRAINING_DATA = bool(getattr(config, "ALLOW_LEGACY_TRAINING_DATA", False))

TRAINING_SAMPLES_FILE = Path("data/training_samples.csv")
TRAINING_STATUS_FILE = Path("data/training_status.json")
MODEL_ARTIFACT_STATUS_FILE = Path("data/model_artifact_status.json")
CALIBRATOR_FILE = Path("data/calibrator.pkl")
TRAINING_LOG_FILE = Path("data/training_log.csv")
FEATURE_IMPORTANCE_LOG = Path("data/feature_importance.csv")
TRAIN_ENSEMBLE_REPORT_FILE = Path("report/train_ensemble_report.json")

MIN_TRAIN_SAMPLES = int(getattr(config, "MIN_CLEAN_TRAIN_SAMPLES", 300))
MIN_PROMOTION_SAMPLES = 500

TRAINING_SOURCE = "data/training_samples.csv"
MODEL_TYPE = "calibrated_logistic_regression_with_imputer"
MODEL_FAMILY = "calibrated_logistic_baseline"

LABEL_COLUMNS = [
    "home_win",
    "y_true",
    "label",
    "result_home_win",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    try:
        if hasattr(value, "item"):
            return _json_safe(value.item())
    except Exception:
        pass

    return str(value)


def safe_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(data)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            safe,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def _base_train_report() -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "report_type": "train_ensemble_v2",
        "status": "skipped",
        "training_source": TRAINING_SOURCE,
        "sample_count": 0,
        "minimum_train_samples": MIN_TRAIN_SAMPLES,
        "minimum_promotion_samples": MIN_PROMOTION_SAMPLES,
        "label_column": "",
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "missing_core_features": [],
        "trained": False,
        "artifact_written": False,
        "artifact_path": str(CALIBRATOR_FILE),
        "brier": None,
        "logloss": None,
        "accuracy": None,
        "roc_auc": None,
        "calibration_method": "sigmoid",
        "production_model_replacement_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }


def write_train_report(report: dict[str, Any]) -> None:
    safe_json_dump(report, TRAIN_ENSEMBLE_REPORT_FILE)


def write_status(
    *,
    trained: bool,
    skipped: bool,
    sample_count: int,
    reason: str | None = None,
    brier: float | None = None,
    logloss: float | None = None,
    accuracy: float | None = None,
    roc_auc: float | None = None,
    used_feature_count: int | None = None,
    transformed_feature_count: int | None = None,
    removed_features: list[str] | None = None,
    train_sample_count: int | None = None,
    calibration_sample_count: int | None = None,
    test_sample_count: int | None = None,
) -> None:
    status = {
        "generated_at": _utc_now(),
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "training_source": TRAINING_SOURCE,
        "allow_legacy_training_data": ALLOW_LEGACY_TRAINING_DATA,
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "minimum_clean_train_samples": MIN_TRAIN_SAMPLES,
        "minimum_promotion_samples": MIN_PROMOTION_SAMPLES,
        "trained": bool(trained),
        "skipped": bool(skipped),
        "sample_count": int(sample_count),
        "reason": reason,
        "brier": brier,
        "logloss": logloss,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "model_feature_count": len(CORE_MODEL_FEATURES),
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "used_feature_count": used_feature_count,
        "transformed_feature_count": transformed_feature_count,
        "removed_low_variance_features": removed_features or [],
        "train_sample_count": train_sample_count,
        "calibration_sample_count": calibration_sample_count,
        "test_sample_count": test_sample_count,
        "active_model_allowed": False,
        "training_allowed_for_production": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_allowed": False,
        "production_model_replacement_allowed": False,
    }

    safe_json_dump(status, TRAINING_STATUS_FILE)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame([row])

    if not path.exists():
        new_frame.to_csv(path, index=False, encoding="utf-8")
        return

    try:
        existing = pd.read_csv(path)
    except Exception:
        backup_path = path.with_suffix(path.suffix + ".bak")
        path.replace(backup_path)
        new_frame.to_csv(path, index=False, encoding="utf-8")
        return

    columns = list(existing.columns)

    for column in new_frame.columns:
        if column not in columns:
            columns.append(column)

    for column in columns:
        if column not in existing.columns:
            existing[column] = np.nan
        if column not in new_frame.columns:
            new_frame[column] = np.nan

    combined = pd.concat(
        [
            existing[columns],
            new_frame[columns],
        ],
        ignore_index=True,
    )
    combined.to_csv(path, index=False, encoding="utf-8")


def _find_label_column(frame: pd.DataFrame) -> str | None:
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            return column

    return None


def _skip_training(
    report: dict[str, Any],
    *,
    reason: str,
    sample_count: int,
    label_column: str = "",
    missing_core_features: list[str] | None = None,
) -> None:
    report["status"] = "skipped"
    report["sample_count"] = int(sample_count)
    report["label_column"] = label_column
    report["missing_core_features"] = missing_core_features or []
    report["trained"] = False
    report["artifact_written"] = False
    report["recommendations"].append("Do not use a trained artifact from this run.")
    report["errors"].append(reason)

    write_status(
        trained=False,
        skipped=True,
        sample_count=sample_count,
        reason=reason,
    )
    write_train_report(report)


def _load_training_samples(report: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    if ALLOW_LEGACY_TRAINING_DATA:
        _skip_training(
            report,
            reason="legacy_training_data_not_allowed_for_baseline_v2_clean",
            sample_count=0,
        )
        return pd.DataFrame(), None

    if not TRAINING_SAMPLES_FILE.exists():
        _skip_training(
            report,
            reason="training_samples_missing",
            sample_count=0,
        )
        return pd.DataFrame(), None

    try:
        frame = pd.read_csv(TRAINING_SAMPLES_FILE)
    except Exception as exc:
        _skip_training(
            report,
            reason=f"unable_to_read_training_samples:{exc}",
            sample_count=0,
        )
        return pd.DataFrame(), None

    if frame.empty:
        _skip_training(
            report,
            reason="training_samples_empty",
            sample_count=0,
        )
        return pd.DataFrame(), None

    label_column = _find_label_column(frame)

    if label_column is None:
        _skip_training(
            report,
            reason="training_samples_missing_label_column",
            sample_count=int(len(frame)),
        )
        return pd.DataFrame(), None

    frame = frame.copy()
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame[frame[label_column].isin([0, 1])].copy()
    frame[label_column] = frame[label_column].astype(int)

    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str).str.strip()
        frame = frame[frame["game_id"] != ""].copy()
        frame = frame.drop_duplicates("game_id", keep="first")

    if "pipeline_version" in frame.columns:
        frame = frame[frame["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()

    if frame.empty:
        _skip_training(
            report,
            reason="no_clean_training_samples_after_contract_filters",
            sample_count=0,
            label_column=label_column,
        )
        return pd.DataFrame(), None

    return frame.reset_index(drop=True), label_column


def prepare_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str], int, str]
    | None
):
    report = _base_train_report()
    frame, label_column = _load_training_samples(report)

    if frame.empty or label_column is None:
        return None

    sample_count = int(len(frame))
    report["sample_count"] = sample_count
    report["label_column"] = label_column

    if sample_count < MIN_TRAIN_SAMPLES:
        _skip_training(
            report,
            reason=f"insufficient_samples:{sample_count}<{MIN_TRAIN_SAMPLES}",
            sample_count=sample_count,
            label_column=label_column,
        )
        return None

    missing_core_features = [
        feature for feature in CORE_MODEL_FEATURES if feature not in frame.columns
    ]

    if missing_core_features:
        _skip_training(
            report,
            reason="missing_core_features",
            sample_count=sample_count,
            label_column=label_column,
            missing_core_features=missing_core_features,
        )
        return None

    if "snapshot_created_at" in frame.columns:
        frame["snapshot_created_at"] = pd.to_datetime(
            frame["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values("snapshot_created_at").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    for column in CORE_MODEL_FEATURES:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    matrix = frame[CORE_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(
        dtype=float
    )
    target = frame[label_column].to_numpy(dtype=int)

    if "snapshot_created_at" in frame.columns and frame["snapshot_created_at"].notna().any():
        latest_snapshot = frame["snapshot_created_at"].max()
        frame["days_ago"] = (
            latest_snapshot - frame["snapshot_created_at"]
        ).dt.days.fillna(0)
    else:
        frame["days_ago"] = 0

    frame["sample_weight"] = np.exp(
        -frame["days_ago"] / 365 * np.log(2)
    ).clip(lower=0.1)
    weights = frame["sample_weight"].to_numpy(dtype=float)

    variance = np.nanvar(matrix, axis=0)
    keep = variance > 1e-8

    if not np.any(keep):
        _skip_training(
            report,
            reason="all_candidate_features_have_zero_variance",
            sample_count=sample_count,
            label_column=label_column,
        )
        return None

    removed_features = [
        feature
        for feature, retained in zip(CORE_MODEL_FEATURES, keep)
        if not retained
    ]

    used_features = [
        feature
        for feature, retained in zip(CORE_MODEL_FEATURES, keep)
        if retained
    ]

    return (
        matrix[:, keep],
        target,
        weights,
        frame,
        used_features,
        removed_features,
        sample_count,
        label_column,
    )


def make_calibrator(estimator: Any) -> CalibratedClassifierCV:
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            FrozenEstimator(estimator),
            method="sigmoid",
        )
    except ImportError:
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv="prefit",
        )


def _safe_metric(value: Any) -> float | None:
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def train() -> None:
    prepared = prepare_data()

    if prepared is None:
        return

    (
        matrix,
        target,
        weights,
        all_rows,
        used_features,
        removed_features,
        sample_count,
        label_column,
    ) = prepared

    report = _base_train_report()
    report["sample_count"] = sample_count
    report["label_column"] = label_column
    report["missing_core_features"] = []

    row_count = len(matrix)
    train_end = int(row_count * 0.70)
    calibration_end = int(row_count * 0.85)

    if (
        train_end < 20
        or calibration_end <= train_end
        or calibration_end >= row_count
    ):
        _skip_training(
            report,
            reason="not_enough_samples_after_train_calibration_test_split",
            sample_count=sample_count,
            label_column=label_column,
        )
        return

    x_train = matrix[:train_end]
    y_train = target[:train_end]
    w_train = weights[:train_end]

    x_calib = matrix[train_end:calibration_end]
    y_calib = target[train_end:calibration_end]
    w_calib = weights[train_end:calibration_end]

    x_test = matrix[calibration_end:]
    y_test = target[calibration_end:]

    for name, subset in (
        ("train", y_train),
        ("calibration", y_calib),
        ("test", y_test),
    ):
        if len(np.unique(subset)) < 2:
            _skip_training(
                report,
                reason=f"{name}_set_contains_only_one_target_class",
                sample_count=sample_count,
                label_column=label_column,
            )
            return

    print(
        f"Training samples: {len(x_train)}, "
        f"calibration samples: {len(x_calib)}, "
        f"test samples: {len(x_test)}"
    )

    base_model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
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

    base_model.fit(
        x_train,
        y_train,
        model__sample_weight=w_train,
    )

    calibrated = make_calibrator(base_model)
    calibrated.fit(x_calib, y_calib, sample_weight=w_calib)

    probabilities = calibrated.predict_proba(x_test)[:, 1]
    clipped_probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predicted_labels = (probabilities >= 0.5).astype(int)

    test_brier = _safe_metric(brier_score_loss(y_test, probabilities))
    test_logloss = _safe_metric(log_loss(y_test, clipped_probabilities))
    test_accuracy = _safe_metric(accuracy_score(y_test, predicted_labels))

    try:
        test_roc_auc = _safe_metric(roc_auc_score(y_test, probabilities))
    except Exception:
        test_roc_auc = None

    try:
        transformed_features = list(
            base_model.named_steps["imputer"].get_feature_names_out(used_features)
        )
    except Exception:
        transformed_features = list(used_features)

    metadata = {
        "artifact_version": "v2",
        "pipeline_version": PIPELINE_VERSION,
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "training_source": TRAINING_SOURCE,
        "training_sample_count": sample_count,
        "training_status_sample_count": sample_count,
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "core_model_features": list(CORE_MODEL_FEATURES),
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "used_features": used_features,
        "created_at": _utc_now(),
        "minimum_train_samples": MIN_TRAIN_SAMPLES,
        "minimum_promotion_samples": MIN_PROMOTION_SAMPLES,
        "brier": test_brier,
        "logloss": test_logloss,
        "accuracy": test_accuracy,
        "roc_auc": test_roc_auc,
        "calibration_method": "sigmoid",
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    artifact = {
        "model": calibrated,
        "features": used_features,
        "feature_names": used_features,
        "transformed_features": transformed_features,
        "schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "pipeline_version": PIPELINE_VERSION,
        "training_source": TRAINING_SOURCE,
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "trained_at": _utc_now(),
        "training_sample_count": sample_count,
        "sample_count": sample_count,
        "test_brier": test_brier,
        "test_logloss": test_logloss,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_roc_auc,
        "metadata": metadata,
    }

    CALIBRATOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, CALIBRATOR_FILE)

    write_status(
        trained=True,
        skipped=False,
        sample_count=sample_count,
        brier=test_brier,
        logloss=test_logloss,
        accuracy=test_accuracy,
        roc_auc=test_roc_auc,
        used_feature_count=len(used_features),
        transformed_feature_count=len(transformed_features),
        removed_features=removed_features,
        train_sample_count=len(x_train),
        calibration_sample_count=len(x_calib),
        test_sample_count=len(x_test),
    )

    report["status"] = "trained"
    report["trained"] = True
    report["artifact_written"] = True
    report["brier"] = test_brier
    report["logloss"] = test_logloss
    report["accuracy"] = test_accuracy
    report["roc_auc"] = test_roc_auc
    report["warnings"] = []
    report["recommendations"] = [
        "Artifact was rebuilt from canonical training samples.",
        "Production model replacement remains disabled by policy.",
    ]
    write_train_report(report)

    append_csv(
        TRAINING_LOG_FILE,
        {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "feature_schema_version": MODEL_FEATURE_VERSION,
            "feature_schema_hash": get_model_feature_schema_hash(),
            "training_source": TRAINING_SOURCE,
            "model_type": MODEL_TYPE,
            "num_samples": len(all_rows),
            "core_feature_count": len(CORE_MODEL_FEATURES),
            "used_feature_count": len(used_features),
            "transformed_feature_count": len(transformed_features),
            "removed_feature_count": len(removed_features),
            "brier": test_brier,
            "logloss": test_logloss,
            "accuracy": test_accuracy,
            "roc_auc": test_roc_auc,
        },
    )

    coefficients = base_model.named_steps["model"].coef_[0]
    absolute_coefficients = np.abs(coefficients)

    importance_feature_names = (
        transformed_features
        if len(transformed_features) == len(absolute_coefficients)
        else used_features
    )

    importance_row = {
        feature: float(value)
        for feature, value in zip(importance_feature_names, absolute_coefficients)
    }
    importance_row["timestamp"] = datetime.now().isoformat()
    importance_row["pipeline_version"] = PIPELINE_VERSION
    importance_row["feature_schema_version"] = MODEL_FEATURE_VERSION
    importance_row["feature_schema_hash"] = get_model_feature_schema_hash()
    importance_row["model_type"] = MODEL_TYPE
    importance_row["used_feature_count"] = len(used_features)
    importance_row["transformed_feature_count"] = len(transformed_features)
    append_csv(FEATURE_IMPORTANCE_LOG, importance_row)

    sorted_indices = np.argsort(absolute_coefficients)
    print("Five lowest absolute clean-model coefficients:")

    for index in sorted_indices[:5]:
        if index >= len(importance_feature_names):
            continue

        print(
            f"  {importance_feature_names[index]}: "
            f"{absolute_coefficients[index]:.6f}"
        )


if __name__ == "__main__":
    train()
