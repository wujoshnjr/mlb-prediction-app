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
FINALIZED_FILE = Path("data/finalized_games.csv")
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
MODEL_SCHEMA_VERSION = "v3-feature-governed"
MODEL_TYPE = "calibrated_logistic_regression_with_imputer"

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
) -> None:
    status = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": MODEL_SCHEMA_VERSION,
        "training_source": "finalized_joined_prediction_snapshots",
        "allow_legacy_training_data": ALLOW_LEGACY_TRAINING_DATA,
        "model_type": MODEL_TYPE,
        "minimum_clean_train_samples": MIN_TRAIN_SAMPLES,
        "trained": trained,
        "skipped": skipped,
        "sample_count": sample_count,
        "reason": reason,
        "brier": brier,
        "logloss": logloss,
        "model_feature_count": len(MODEL_FEATURES),
        "availability_flag_feature_count": len(AVAILABILITY_FLAG_FEATURES),
        "tracking_only_feature_count": len(TRACKING_ONLY_FEATURES),
        "used_feature_count": used_feature_count,
        "transformed_feature_count": transformed_feature_count,
        "removed_low_variance_features": removed_features or [],
        "training_allowed_for_production": (
            trained is True and sample_count >= MIN_TRAIN_SAMPLES
        ),
        "timestamp": datetime.now().isoformat(),
    }

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one row while preserving schema when columns evolve."""
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


def _normalize_game_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = float(text)
        if np.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "valid", "ok"})


def _prepare_snapshot_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if "game_id" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["game_id"] = frame["game_id"].apply(_normalize_game_id)
    frame = frame[frame["game_id"] != ""].copy()

    if "pipeline_version" in frame.columns:
        frame = frame[frame["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()

    if "snapshot_valid" in frame.columns:
        frame = frame[_bool_series(frame["snapshot_valid"])].copy()

    for leakage_column in ("home_win", "home_score", "away_score"):
        if leakage_column in frame.columns:
            frame = frame.drop(columns=[leakage_column])

    if "snapshot_created_at" not in frame.columns:
        return pd.DataFrame()

    frame["snapshot_created_at"] = pd.to_datetime(
        frame["snapshot_created_at"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["snapshot_created_at"])

    if frame.empty:
        return frame

    frame = (
        frame.sort_values("snapshot_created_at")
        .groupby("game_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return frame


def _prepare_finalized_outcomes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if "game_id" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["game_id"] = frame["game_id"].apply(_normalize_game_id)
    frame = frame[frame["game_id"] != ""].copy()

    if "home_win" not in frame.columns:
        if {"home_score", "away_score"}.issubset(frame.columns):
            home_score = pd.to_numeric(frame["home_score"], errors="coerce")
            away_score = pd.to_numeric(frame["away_score"], errors="coerce")
            frame["home_win"] = (home_score > away_score).astype("Int64")
        else:
            return pd.DataFrame()

    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    frame["home_win"] = frame["home_win"].astype(int)

    return frame[["game_id", "home_win"]].drop_duplicates("game_id", keep="last")


def prepare_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str], int]
    | None
):
    """Load clean pregame snapshots and join outcomes only from finalized_games.csv."""
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

    if not FINALIZED_FILE.exists():
        write_status(
            False,
            True,
            0,
            reason="finalized_games.csv does not exist; trusted outcomes are unavailable.",
        )
        return None

    try:
        snapshots = _prepare_snapshot_features(SNAPSHOT_FILE)
        finalized = _prepare_finalized_outcomes(FINALIZED_FILE)
    except Exception as exc:
        write_status(
            False,
            True,
            0,
            reason=f"Unable to prepare finalized-joined training rows: {exc}",
        )
        return None

    if snapshots.empty:
        write_status(
            False,
            True,
            0,
            reason="No valid clean pregame prediction snapshots are available.",
        )
        return None

    if finalized.empty:
        write_status(
            False,
            True,
            0,
            reason="No trusted finalized outcomes are available in finalized_games.csv.",
        )
        return None

    frame = snapshots.merge(
        finalized,
        on="game_id",
        how="inner",
    )

    if frame.empty:
        write_status(
            False,
            True,
            0,
            reason="No prediction snapshots join to finalized_games.csv by game_id.",
        )
        return None

    frame = frame.sort_values("snapshot_created_at").reset_index(drop=True)

    sample_count = len(frame)
    if sample_count < MIN_TRAIN_SAMPLES:
        write_status(
            False,
            True,
            sample_count,
            reason=(
                "Not enough finalized-joined clean snapshots "
                f"({sample_count} < {MIN_TRAIN_SAMPLES})."
            ),
        )
        return None

    for column in MODEL_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if column in AVAILABILITY_FLAG_FEATURES:
            frame[column] = frame[column].fillna(0.0).clip(lower=0.0, upper=1.0)

    latest_snapshot = frame["snapshot_created_at"].max()
    frame["days_ago"] = (
        latest_snapshot - frame["snapshot_created_at"]
    ).dt.days.fillna(0)
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
            False,
            True,
            sample_count,
            reason="All finalized-joined candidate features have zero variance.",
        )
        return None

    removed_features = [
        feature
        for feature, retained in zip(MODEL_FEATURES, keep)
        if not retained
    ]
    if removed_features:
        print(f"Removed low-variance clean features: {removed_features}")

    used_features = [
        feature
        for feature, retained in zip(MODEL_FEATURES, keep)
        if retained
    ]

    return matrix[:, keep], target, weights, frame, used_features, removed_features, sample_count
    

def make_calibrator(estimator: Any) -> CalibratedClassifierCV:
    """Create a sigmoid calibrator around an already fitted estimator."""
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
        "transformed_features": transformed_features,
        "schema_version": MODEL_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "training_source": "finalized_joined_prediction_snapshots",
        "model_type": MODEL_TYPE,
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
        used_feature_count=len(used_features),
        transformed_feature_count=len(transformed_features),
        removed_features=removed_features,
    )

    append_csv(
        TRAINING_LOG,
        {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": MODEL_SCHEMA_VERSION,
            "training_source": "finalized_joined_prediction_snapshots",
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
