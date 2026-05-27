# train_ensemble.py
"""Train and calibrate the MLB ensemble model using finalized historical games."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.feature_schema import EXPECTED_FEATURES
from scripts.snapshot_store import read_snapshot_rows
try:
    import config
except ImportError:
    class config:  # type: ignore[no-redef]
        PIPELINE_VERSION = "baseline_v2_clean"
        SNAPSHOT_STORE_FILE = "data/prediction_snapshots.csv"
        ALLOW_LEGACY_TRAINING_DATA = False
        MIN_CLEAN_TRAIN_SAMPLES = 300


PIPELINE_VERSION = str(
    getattr(config, "PIPELINE_VERSION", "baseline_v2_clean")
)
SNAPSHOT_FILE = Path(
    str(
        getattr(
            config,
            "SNAPSHOT_STORE_FILE",
            "data/prediction_snapshots.csv",
        )
    )
)
ALLOW_LEGACY_TRAINING_DATA = bool(
    getattr(config, "ALLOW_LEGACY_TRAINING_DATA", False)
)
MIN_TRAIN_SAMPLES = int(
    getattr(config, "MIN_CLEAN_TRAIN_SAMPLES", 300)
)

MODEL_OUTPUT = Path("data/calibrator.pkl")
STATUS_FILE = Path("data/training_status.json")
TRAINING_LOG = Path("data/training_log.csv")
FEATURE_IMPORTANCE_LOG = Path("data/feature_importance.csv")

def write_status(
    trained: bool,
    skipped: bool,
    sample_count: int,
    reason: str | None = None,
    brier: float | None = None,
    logloss: float | None = None,
) -> None:
    status = {
        "pipeline_version": PIPELINE_VERSION,
        "training_source": "clean_prediction_snapshots",
        "allow_legacy_training_data": ALLOW_LEGACY_TRAINING_DATA,
        "model_type": "calibrated_logistic_regression",
        "minimum_clean_train_samples": MIN_TRAIN_SAMPLES,
        "trained": trained,
        "skipped": skipped,
        "sample_count": sample_count,
        "reason": reason,
        "brier": brier,
        "logloss": logloss,
        "timestamp": datetime.now().isoformat(),
    }

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8",
    )


def prepare_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], int]
    | None
):
    """Load forward-collected, settled clean snapshots only."""
    if ALLOW_LEGACY_TRAINING_DATA:
        write_status(
            False,
            True,
            0,
            reason=(
                "Legacy training data is enabled, but baseline_v2_clean "
                "training refuses legacy rows to prevent leakage."
            ),
        )
        return None

    if not SNAPSHOT_FILE.exists():
        write_status(
            False,
            True,
            0,
            reason="Clean prediction snapshot file does not exist.",
        )
        return None

    try:
        frame = read_snapshot_rows(
            pipeline_version=PIPELINE_VERSION,
            valid_only=True,
            settled_only=True,
            path=SNAPSHOT_FILE,
        )
    except Exception as exc:
        write_status(
            False,
            True,
            0,
            reason=f"Unable to read clean prediction snapshots: {exc}",
        )
        return None

    required_columns = {
        "pipeline_version",
        "snapshot_valid",
        "snapshot_created_at",
        "home_win",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        write_status(
            False,
            True,
            0,
            reason=f"Clean snapshot file is missing columns: {missing_columns}",
        )
        return None

    frame = frame.copy()
    frame["snapshot_created_at"] = pd.to_datetime(
        frame["snapshot_created_at"],
        errors="coerce",
        utc=True,
    )
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")

    frame = frame.dropna(subset=["snapshot_created_at", "home_win"])
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    frame["home_win"] = frame["home_win"].astype(int)
    frame = frame.sort_values("snapshot_created_at").reset_index(drop=True)

    sample_count = len(frame)
    if sample_count < MIN_TRAIN_SAMPLES:
        write_status(
            False,
            True,
            sample_count,
            reason=(
                "Not enough settled baseline_v2_clean snapshots "
                f"({sample_count} < {MIN_TRAIN_SAMPLES})."
            ),
        )
        return None

    for column in EXPECTED_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0.0)

    latest_snapshot = frame["snapshot_created_at"].max()
    frame["days_ago"] = (
        latest_snapshot - frame["snapshot_created_at"]
    ).dt.days.fillna(0)
    frame["sample_weight"] = np.exp(
        -frame["days_ago"] / 365 * np.log(2)
    ).clip(lower=0.1)

    matrix = frame[EXPECTED_FEATURES].to_numpy(dtype=float)
    target = frame["home_win"].to_numpy(dtype=int)
    weights = frame["sample_weight"].to_numpy(dtype=float)

    variance = np.var(matrix, axis=0)
    keep = variance > 1e-8

    if not np.any(keep):
        write_status(
            False,
            True,
            sample_count,
            reason="All clean candidate features have zero variance.",
        )
        return None

    removed_features = [
        feature
        for feature, retained in zip(EXPECTED_FEATURES, keep)
        if not retained
    ]
    if removed_features:
        print(f"Removed low-variance clean features: {removed_features}")

    used_features = [
        feature
        for feature, retained in zip(EXPECTED_FEATURES, keep)
        if retained
    ]

    return matrix[:, keep], target, weights, frame, used_features, sample_count

def make_calibrator(stacking: StackingClassifier) -> CalibratedClassifierCV:
    """Support newer and older scikit-learn calibration APIs."""
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            FrozenEstimator(stacking),
            method="sigmoid",
        )
    except ImportError:
        return CalibratedClassifierCV(
            estimator=stacking,
            method="sigmoid",
            cv="prefit",
        )


def train() -> None:
    prepared = prepare_data()
    if prepared is None:
        return

    matrix, target, weights, all_rows, used_features, sample_count = prepared

    row_count = len(matrix)
    train_end = int(row_count * 0.70)
    calibration_end = int(row_count * 0.85)

    if (
        train_end < 20
        or calibration_end <= train_end
        or calibration_end >= row_count
    ):
        write_status(
            False,
            True,
            sample_count,
            reason="Not enough samples after train/calibration/test split.",
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
                False,
                True,
                sample_count,
                reason=f"{name} set contains only one target class.",
            )
            return

    print(
        f"Training samples: {len(x_train)}, "
        f"calibration samples: {len(x_calib)}, "
        f"test samples: {len(x_test)}"
    )

      base_model = Pipeline(
        [
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

    artifact = {
        "model": calibrated,
        "features": used_features,
        "schema_version": "v2-shared-schema",
        "pipeline_version": PIPELINE_VERSION,
        "training_source": "clean_prediction_snapshots",
        "model_type": "calibrated_logistic_regression",
        "trained_at": datetime.now().isoformat(),
        "training_sample_count": sample_count,
        "test_brier": round(test_brier, 4),
        "test_logloss": round(test_logloss, 4),
    }

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT)

    write_status(
        True,
        False,
        sample_count,
        brier=round(test_brier, 4),
        logloss=round(test_logloss, 4),
    )

    append_csv(
        TRAINING_LOG,
        {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "training_source": "clean_prediction_snapshots",
            "model_type": "calibrated_logistic_regression",
            "num_samples": len(all_rows),
            "used_feature_count": len(used_features),
            "brier": round(test_brier, 4),
            "logloss": round(test_logloss, 4),
        },
    )

    coefficients = base_model.named_steps["model"].coef_[0]
    absolute_coefficients = np.abs(coefficients)

    importance_row = {
        feature: float(value)
        for feature, value in zip(used_features, absolute_coefficients)
    }
    importance_row["timestamp"] = datetime.now().isoformat()
    importance_row["pipeline_version"] = PIPELINE_VERSION
    importance_row["model_type"] = "calibrated_logistic_regression"
    append_csv(FEATURE_IMPORTANCE_LOG, importance_row)

    sorted_indices = np.argsort(absolute_coefficients)
    print("Five lowest absolute clean-model coefficients:")
    for index in sorted_indices[:5]:
        print(f"  {used_features[index]}: {absolute_coefficients[index]:.6f}")

    print(
        "Clean training completed. "
        f"Pipeline={PIPELINE_VERSION}, "
        f"Test Brier={test_brier:.4f}, "
        f"Test LogLoss={test_logloss:.4f}"
    )


if __name__ == "__main__":
    train()
