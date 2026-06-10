# train_ensemble.py
"""Train and calibrate the MLB baseline model from canonical training samples.

Canonical source:
- data/training_samples.csv

This file intentionally refuses to re-join prediction_snapshots.csv and
finalized_games.csv. Outcome linkage is owned by the evidence chain.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.feature_schema import (
    MODEL_FEATURES,
    AVAILABILITY_FLAG_FEATURES,
    TRACKING_ONLY_FEATURES,
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
MIN_TRAIN_SAMPLES = int(getattr(config, "MIN_CLEAN_TRAIN_SAMPLES", 300))

TRAINING_SAMPLES_FILE = Path("data/training_samples.csv")
MODEL_OUTPUT = Path("data/calibrator.pkl")
STATUS_FILE = Path("data/training_status.json")
TRAINING_LOG = Path("data/training_log.csv")
FEATURE_IMPORTANCE_LOG = Path("data/feature_importance.csv")

MODEL_SCHEMA_VERSION = "v3-feature-governed"
MODEL_TYPE = "calibrated_logistic_regression_with_imputer"
TRAINING_SOURCE = "data/training_samples.csv"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(
    trained: bool,
    skipped: bool,
    sample_count: int,
    reason: str | None = None,
    brier: float | None = None,
    logloss: float | None = None,
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
        "schema_version": MODEL_SCHEMA_VERSION,
        "training_source": TRAINING_SOURCE,
        "allow_legacy_training_data": ALLOW_LEGACY_TRAINING_DATA,
        "model_type": MODEL_TYPE,
        "model_family": "calibrated_logistic_baseline",
        "minimum_clean_train_samples": MIN_TRAIN_SAMPLES,
        "trained": bool(trained),
        "skipped": bool(skipped),
        "sample_count": int(sample_count),
        "reason": reason,
        "brier": brier,
        "logloss": logloss,
        "model_feature_count": len(MODEL_FEATURES),
        "availability_flag_feature_count": len(AVAILABILITY_FLAG_FEATURES),
        "tracking_only_feature_count": len(TRACKING_ONLY_FEATURES),
        "used_feature_count": used_feature_count,
        "transformed_feature_count": transformed_feature_count,
        "removed_low_variance_features": removed_features or [],
        "train_sample_count": train_sample_count,
        "calibration_sample_count": calibration_sample_count,
        "test_sample_count": test_sample_count,
        "training_allowed_for_production": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_allowed": False,
        "production_model_replacement_allowed": False,
    }

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(status, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


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


def _load_training_samples() -> pd.DataFrame:
    if not TRAINING_SAMPLES_FILE.exists():
        write_status(
            trained=False,
            skipped=True,
            sample_count=0,
            reason="training_samples_missing",
        )
        return pd.DataFrame()

    try:
        frame = pd.read_csv(TRAINING_SAMPLES_FILE)
    except Exception as exc:
        write_status(
            trained=False,
            skipped=True,
            sample_count=0,
            reason=f"unable_to_read_training_samples:{exc}",
        )
        return pd.DataFrame()

    if frame.empty:
        write_status(
            trained=False,
            skipped=True,
            sample_count=0,
            reason="training_samples_empty",
        )
        return pd.DataFrame()

    if "home_win" not in frame.columns:
        write_status(
            trained=False,
            skipped=True,
            sample_count=int(len(frame)),
            reason="training_samples_missing_home_win",
        )
        return pd.DataFrame()

    frame = frame.copy()
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    frame["home_win"] = frame["home_win"].astype(int)

    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str).str.strip()
        frame = frame[frame["game_id"] != ""].copy()

    if "pipeline_version" in frame.columns:
        frame = frame[frame["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()

    if frame.empty:
        write_status(
            trained=False,
            skipped=True,
            sample_count=0,
            reason="no_clean_training_samples_after_contract_filters",
        )
        return pd.DataFrame()

    frame = frame.drop_duplicates("game_id", keep="first") if "game_id" in frame.columns else frame

    return frame.reset_index(drop=True)


def prepare_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str], int]
    | None
):
    if ALLOW_LEGACY_TRAINING_DATA:
        write_status(
            trained=False,
            skipped=True,
            sample_count=0,
            reason=(
                "Legacy training data is enabled, but baseline_v2_clean "
                "training refuses legacy rows to prevent leakage."
            ),
        )
        return None

    frame = _load_training_samples()
    if frame.empty:
        return None

    sample_count = int(len(frame))

    if sample_count < MIN_TRAIN_SAMPLES:
        write_status(
            trained=False,
            skipped=True,
            sample_count=sample_count,
            reason=f"insufficient_clean_training_samples:{sample_count}<{MIN_TRAIN_SAMPLES}",
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

    for column in MODEL_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if column in AVAILABILITY_FLAG_FEATURES:
            frame[column] = frame[column].fillna(0.0).clip(lower=0.0, upper=1.0)

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

    matrix = frame[MODEL_FEATURES].to_numpy(dtype=float)
    target = frame["home_win"].to_numpy(dtype=int)
    weights = frame["sample_weight"].to_numpy(dtype=float)

    variance = np.nanvar(matrix, axis=0)
    keep = variance > 1e-8

    if not np.any(keep):
        write_status(
            trained=False,
            skipped=True,
            sample_count=sample_count,
            reason="all_candidate_features_have_zero_variance",
        )
        return None

    removed_features = [
        feature
        for feature, retained in zip(MODEL_FEATURES, keep)
        if not retained
    ]

    used_features = [
        feature
        for feature, retained in zip(MODEL_FEATURES, keep)
        if retained
    ]

    return matrix[:, keep], target, weights, frame, used_features, removed_features, sample_count


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


def train() -> None:
    prepared = prepare_data()
    if prepared is None:
        return

    matrix, target, weights, all_rows, used_features, removed_features, sample_count = prepared

    row_count = len(matrix)
    train_end = int(row_count * 0.70)
    calibration_end = int(row_count * 0.85)

    if (
        train_end < 20
        or calibration_end <= train_end
        or calibration_end >= row_count
    ):
        write_status(
            trained=False,
            skipped=True,
            sample_count=sample_count,
            reason="not_enough_samples_after_train_calibration_test_split",
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
            write_status(
                trained=False,
                skipped=True,
                sample_count=sample_count,
                reason=f"{name}_set_contains_only_one_target_class",
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
    test_brier = float(brier_score_loss(y_test, probabilities))
    test_logloss = float(log_loss(y_test, probabilities))

    try:
        transformed_features = list(
            base_model.named_steps["imputer"].get_feature_names_out(used_features)
        )
    except Exception:
        transformed_features = list(used_features)

    artifact = {
        "model": calibrated,
        "features": used_features,
        "feature_names": used_features,
        "transformed_features": transformed_features,
        "schema_version": MODEL_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "training_source": TRAINING_SOURCE,
        "model_type": MODEL_TYPE,
        "model_family": "calibrated_logistic_baseline",
        "trained_at": datetime.now().isoformat(),
        "training_sample_count": sample_count,
        "sample_count": sample_count,
        "test_brier": round(test_brier, 4),
        "test_logloss": round(test_logloss, 4),
        "metadata": {
            "artifact_version": "v1",
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "training_source": TRAINING_SOURCE,
            "training_sample_count": sample_count,
            "created_at": _utc_now(),
            "feature_names": used_features,
            "brier": round(test_brier, 4),
            "logloss": round(test_logloss, 4),
        },
    }

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT)

    write_status(
        trained=True,
        skipped=False,
        sample_count=sample_count,
        brier=round(test_brier, 4),
        logloss=round(test_logloss, 4),
        used_feature_count=len(used_features),
        transformed_feature_count=len(transformed_features),
        removed_features=removed_features,
        train_sample_count=len(x_train),
        calibration_sample_count=len(x_calib),
        test_sample_count=len(x_test),
    )

    append_csv(
        TRAINING_LOG,
        {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": MODEL_SCHEMA_VERSION,
            "training_source": TRAINING_SOURCE,
            "model_type": MODEL_TYPE,
            "num_samples": len(all_rows),
            "model_feature_count": len(MODEL_FEATURES),
            "used_feature_count": len(used_features),
            "transformed_feature_count": len(transformed_features),
            "removed_feature_count": len(removed_features),
            "brier": round(test_brier, 4),
            "logloss": round(test_logloss, 4),
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
    importance_row["schema_version"] = MODEL_SCHEMA_VERSION
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

    print(
        "Clean training completed. "
        f"Pipeline={PIPELINE_VERSION}, "
        f"Test Brier={test_brier:.4f}, "
        f"Test LogLoss={test_logloss:.4f}"
    )


if __name__ == "__main__":
    train()
