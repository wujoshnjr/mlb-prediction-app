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
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from scripts.feature_schema import EXPECTED_FEATURES

try:
    import config
except ImportError:
    class config:  # type: ignore[no-redef]
        MODEL_USE_MLP = False
        MODEL_META = "lr"


HISTORY_FILE = Path("data/historical_predictions.csv")
MODEL_OUTPUT = Path("data/calibrator.pkl")
STATUS_FILE = Path("data/training_status.json")
TRAINING_LOG = Path("data/training_log.csv")
FEATURE_IMPORTANCE_LOG = Path("data/feature_importance.csv")

MIN_TRAIN_SAMPLES = 100


def write_status(
    trained: bool,
    skipped: bool,
    sample_count: int,
    reason: str | None = None,
    brier: float | None = None,
    logloss: float | None = None,
) -> None:
    status = {
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
    if not HISTORY_FILE.exists():
        write_status(
            False,
            True,
            0,
            reason="Historical prediction file does not exist.",
        )
        return None

    try:
        frame = pd.read_csv(HISTORY_FILE)
    except Exception as exc:
        write_status(
            False,
            True,
            0,
            reason=f"Unable to read historical prediction file: {exc}",
        )
        return None

    if "home_win" not in frame.columns:
        write_status(
            False,
            True,
            0,
            reason="Historical prediction file is missing home_win.",
        )
        return None

    date_column = None
    if "game_date" in frame.columns:
        date_column = "game_date"
    elif "date" in frame.columns:
        date_column = "date"

    if date_column is not None:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.sort_values(date_column).reset_index(drop=True)

    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame = frame.dropna(subset=["home_win"])
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    frame["home_win"] = frame["home_win"].astype(int)

    sample_count = len(frame)
    if sample_count < MIN_TRAIN_SAMPLES:
        write_status(
            False,
            True,
            sample_count,
            reason=(
                f"Not enough finalized training samples "
                f"({sample_count} < {MIN_TRAIN_SAMPLES})."
            ),
        )
        return None

    for column in EXPECTED_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    if date_column is not None and frame[date_column].notna().any():
        max_date = frame[date_column].max()
        frame["days_ago"] = (max_date - frame[date_column]).dt.days.fillna(0)
        frame["sample_weight"] = np.exp(
            -frame["days_ago"] / 365 * np.log(2)
        ).clip(lower=0.1)
    else:
        frame["sample_weight"] = 1.0

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
            reason="All candidate features have zero variance.",
        )
        return None

    removed_features = [
        feature
        for feature, retained in zip(EXPECTED_FEATURES, keep)
        if not retained
    ]
    if removed_features:
        print(f"Removed low-variance features: {removed_features}")

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

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        importance_type="gain",
        random_state=42,
        eval_metric="logloss",
    )
    lgb = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        random_state=42,
        verbose=-1,
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
    )

    estimators: list[tuple[str, Any]] = [
        ("xgb", xgb),
        ("lgb", lgb),
        ("rf", forest),
    ]

    if getattr(config, "MODEL_USE_MLP", False):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.001,
            early_stopping=True,
            random_state=42,
        )
        estimators.append(
            (
                "mlp",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("mlp", mlp),
                    ]
                ),
            )
        )

    if getattr(config, "MODEL_META", "lr") == "elasticnet":
        from sklearn.linear_model import SGDClassifier

        meta_model: Any = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.5,
            alpha=0.0001,
            random_state=42,
            max_iter=2000,
            tol=1e-3,
        )
    else:
        meta_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
        )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
    )

    stacking.fit(x_train, y_train, sample_weight=w_train)

    calibrated = make_calibrator(stacking)
    calibrated.fit(x_calib, y_calib, sample_weight=w_calib)

    probabilities = calibrated.predict_proba(x_test)[:, 1]
    test_brier = float(brier_score_loss(y_test, probabilities))
    test_logloss = float(log_loss(y_test, probabilities))

    artifact = {
        "model": calibrated,
        "features": used_features,
        "schema_version": "v2-shared-schema",
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
            "num_samples": len(all_rows),
            "used_feature_count": len(used_features),
            "brier": round(test_brier, 4),
            "logloss": round(test_logloss, 4),
        },
    )

    importance_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        importance_type="gain",
        random_state=42,
        eval_metric="logloss",
    )
    importance_model.fit(x_train, y_train, sample_weight=w_train)
    importances = importance_model.feature_importances_

    importance_row = {
        feature: float(value)
        for feature, value in zip(used_features, importances)
    }
    importance_row["timestamp"] = datetime.now().isoformat()
    append_csv(FEATURE_IMPORTANCE_LOG, importance_row)

    sorted_indices = np.argsort(importances)
    print("Five lowest-importance features:")
    for index in sorted_indices[:5]:
        print(f"  {used_features[index]}: {importances[index]:.6f}")

    print(
        f"Training completed. Test Brier={test_brier:.4f}, "
        f"Test LogLoss={test_logloss:.4f}"
    )


if __name__ == "__main__":
    train()
