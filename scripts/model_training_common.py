from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


try:
    import config
except ImportError:
    class config:  # type: ignore[no-redef]
        PIPELINE_VERSION = "baseline_v2_clean"
        SNAPSHOT_STORE_FILE = "data/prediction_snapshots.csv"


try:
    from scripts.feature_schema import MODEL_FEATURES, TRACKING_ONLY_FEATURES
except Exception:
    MODEL_FEATURES = []
    TRACKING_ONLY_FEATURES = []


SNAPSHOT_PATH = Path(str(getattr(config, "SNAPSHOT_STORE_FILE", "data/prediction_snapshots.csv")))
FINALIZED_PATH = Path("data/finalized_games.csv")
PIPELINE_VERSION = str(getattr(config, "PIPELINE_VERSION", "baseline_v2_clean"))

LEAKAGE_COLUMNS = {
    "home_win",
    "home_score",
    "away_score",
    "final_score",
    "home_final_score",
    "away_final_score",
    "final_home_score",
    "final_away_score",
    "settled_result",
    "settled_at",
    "actual_winner",
    "actual_result",
    "postgame_win_probability",
    "winner",
    "result",
    "is_final",
}

MARKET_PROBABILITY_COLUMNS = [
    "market_no_vig_home_prob",
    "no_vig_market_home_prob",
    "no_vig_home_prob",
    "market_home_prob",
    "market_probability",
    "market_prob",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalize_game_id(value: Any) -> str:
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
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def safe_int(value: Any, default: int = 0) -> int:
    parsed = safe_float(value)
    return int(parsed) if parsed is not None else default


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "valid", "ok"})


def read_csv_safe(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "error": "",
    }

    if not path.exists():
        status["error"] = "file_missing"
        return pd.DataFrame(), status

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return pd.DataFrame(), status

    status["rows"] = int(len(frame))
    return frame, status


def detect_leakage_columns(columns: Iterable[str]) -> List[str]:
    result: List[str] = []
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in LEAKAGE_COLUMNS:
            result.append(str(column))
            continue
        if "postgame" in normalized:
            result.append(str(column))
            continue
        if normalized.startswith("final_") or normalized.endswith("_final"):
            result.append(str(column))
            continue
        if normalized in {"home_win_x", "home_win_y"}:
            result.append(str(column))
    return sorted(set(result))


def remove_leakage_columns(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    leakage = detect_leakage_columns(frame.columns)
    if not leakage:
        return frame.copy(), []
    return frame.drop(columns=[col for col in leakage if col in frame.columns]).copy(), leakage


def prepare_snapshots(
    snapshots: pd.DataFrame,
    *,
    pipeline_version: Optional[str] = PIPELINE_VERSION,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if snapshots.empty:
        warnings.append("prediction_snapshots.csv is empty")
        return pd.DataFrame(), warnings, errors

    if "game_id" not in snapshots.columns:
        errors.append("prediction_snapshots.csv missing game_id")
        return pd.DataFrame(), warnings, errors

    frame = snapshots.copy()
    frame["game_id"] = frame["game_id"].apply(normalize_game_id)
    frame = frame[frame["game_id"] != ""].copy()

    if frame.empty:
        warnings.append("no snapshots with valid game_id")
        return pd.DataFrame(), warnings, errors

    if pipeline_version and "pipeline_version" in frame.columns:
        filtered = frame[frame["pipeline_version"].astype(str) == str(pipeline_version)].copy()
        if filtered.empty:
            warnings.append(f"no snapshots matched pipeline_version={pipeline_version}; using all versions")
        else:
            frame = filtered

    if "snapshot_valid" in frame.columns:
        frame = frame[bool_series(frame["snapshot_valid"])].copy()

    if frame.empty:
        warnings.append("no valid snapshots after snapshot_valid filtering")
        return pd.DataFrame(), warnings, errors

    frame, leakage = remove_leakage_columns(frame)
    if leakage:
        warnings.append("removed leakage columns from snapshots: " + ", ".join(leakage))

    time_column = None
    for candidate in ("snapshot_created_at", "prediction_created_at", "generated_at", "game_date"):
        if candidate in frame.columns:
            time_column = candidate
            break

    if time_column:
        frame["_training_sort_time"] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
        frame["_training_sort_missing"] = frame["_training_sort_time"].isna().astype(int)
        frame = frame.sort_values(
            ["game_id", "_training_sort_missing", "_training_sort_time"],
            kind="mergesort",
        )
        frame = frame.groupby("game_id", as_index=False).tail(1)
        frame = frame.sort_values(
            ["_training_sort_missing", "_training_sort_time", "game_id"],
            kind="mergesort",
        )
    else:
        warnings.append("no timestamp column found; using stable row order fallback")
        frame["_training_sort_time"] = pd.NaT
        frame = frame.drop_duplicates("game_id", keep="last").sort_values("game_id", kind="mergesort")

    return frame.reset_index(drop=True), warnings, errors


def prepare_finalized(finalized: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if finalized.empty:
        warnings.append("finalized_games.csv is empty")
        return pd.DataFrame(), warnings, errors

    if "game_id" not in finalized.columns:
        errors.append("finalized_games.csv missing game_id")
        return pd.DataFrame(), warnings, errors

    frame = finalized.copy()
    frame["game_id"] = frame["game_id"].apply(normalize_game_id)
    frame = frame[frame["game_id"] != ""].copy()

    if "home_win" not in frame.columns:
        if {"home_score", "away_score"}.issubset(frame.columns):
            home_score = pd.to_numeric(frame["home_score"], errors="coerce")
            away_score = pd.to_numeric(frame["away_score"], errors="coerce")
            frame["home_win"] = (home_score > away_score).astype("Int64")
            warnings.append("derived home_win from finalized_games home_score/away_score")
        else:
            errors.append("finalized_games.csv missing home_win and score columns")
            return pd.DataFrame(), warnings, errors

    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame = frame[frame["home_win"].isin([0, 1])].copy()

    if frame.empty:
        warnings.append("no finalized rows with valid home_win target")
        return pd.DataFrame(), warnings, errors

    frame["home_win"] = frame["home_win"].astype(int)

    return frame[["game_id", "home_win"]].drop_duplicates("game_id", keep="last"), warnings, errors


def find_market_probability_column(frame: pd.DataFrame) -> Optional[str]:
    for column in MARKET_PROBABILITY_COLUMNS:
        if column in frame.columns:
            return column
    return None


def build_training_frame(
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    finalized_path: Path = FINALIZED_PATH,
    pipeline_version: Optional[str] = PIPELINE_VERSION,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []

    snapshots_raw, snapshot_status = read_csv_safe(snapshot_path)
    finalized_raw, finalized_status = read_csv_safe(finalized_path)

    if snapshot_status["error"]:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": f"prediction_snapshots unavailable: {snapshot_status['error']}",
            "warnings": warnings,
            "errors": [snapshot_status["error"]],
            "input_files": {"snapshots": snapshot_status, "finalized": finalized_status},
            "frame": pd.DataFrame(),
        }

    if finalized_status["error"]:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": f"finalized_games unavailable: {finalized_status['error']}",
            "warnings": warnings,
            "errors": [finalized_status["error"]],
            "input_files": {"snapshots": snapshot_status, "finalized": finalized_status},
            "frame": pd.DataFrame(),
        }

    snapshots, snapshot_warnings, snapshot_errors = prepare_snapshots(
        snapshots_raw,
        pipeline_version=pipeline_version,
    )
    finalized, finalized_warnings, finalized_errors = prepare_finalized(finalized_raw)

    warnings.extend(snapshot_warnings)
    warnings.extend(finalized_warnings)
    errors.extend(snapshot_errors)
    errors.extend(finalized_errors)

    if errors:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "; ".join(errors),
            "warnings": warnings,
            "errors": errors,
            "input_files": {"snapshots": snapshot_status, "finalized": finalized_status},
            "frame": pd.DataFrame(),
        }

    if snapshots.empty or finalized.empty:
        reason = "no clean snapshots or finalized outcomes available"
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": reason,
            "warnings": warnings,
            "errors": errors,
            "input_files": {"snapshots": snapshot_status, "finalized": finalized_status},
            "frame": pd.DataFrame(),
        }

    frame = snapshots.merge(finalized, on="game_id", how="inner")

    if frame.empty:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "no prediction snapshots join to finalized_games by game_id",
            "warnings": warnings,
            "errors": errors,
            "input_files": {"snapshots": snapshot_status, "finalized": finalized_status},
            "frame": pd.DataFrame(),
        }

    frame = frame.reset_index(drop=True)

    if "_training_sort_time" in frame.columns:
        frame = frame.sort_values(["_training_sort_time", "game_id"], kind="mergesort").reset_index(drop=True)
    else:
        frame = frame.sort_values("game_id", kind="mergesort").reset_index(drop=True)

    return {
        "ok": True,
        "skipped": False,
        "skip_reason": "",
        "warnings": warnings,
        "errors": errors,
        "input_files": {"snapshots": snapshot_status, "finalized": finalized_status},
        "frame": frame,
        "sample_count": int(len(frame)),
    }


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    base_features: Optional[Sequence[str]] = None,
    candidate_features: Optional[Sequence[str]] = None,
    allow_tracking_only: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []

    if frame.empty:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "empty training frame",
            "warnings": warnings,
            "errors": errors,
            "X": np.empty((0, 0)),
            "y": np.array([], dtype=int),
            "features_used": [],
            "experimental_features_used": [],
            "frame": frame,
        }

    base = list(base_features or MODEL_FEATURES)
    candidate = list(candidate_features or [])

    blocked_tracking = [
        feature for feature in candidate
        if feature in TRACKING_ONLY_FEATURES and not allow_tracking_only
    ]
    if blocked_tracking:
        warnings.append(
            "tracking-only features blocked from feature matrix: "
            + ", ".join(blocked_tracking)
        )

    allowed_candidate = [
        feature for feature in candidate
        if allow_tracking_only or feature not in TRACKING_ONLY_FEATURES
    ]

    features = []
    for feature in [*base, *allowed_candidate]:
        if feature not in features:
            features.append(feature)

    leakage = detect_leakage_columns(features)
    if leakage:
        warnings.append("removed requested leakage features: " + ", ".join(leakage))
        features = [feature for feature in features if feature not in leakage]

    work = frame.copy()
    for feature in features:
        if feature not in work.columns:
            work[feature] = np.nan
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    if "home_win" not in work.columns:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "training frame missing target home_win",
            "warnings": warnings,
            "errors": ["missing home_win"],
            "X": np.empty((0, 0)),
            "y": np.array([], dtype=int),
            "features_used": features,
            "experimental_features_used": allowed_candidate,
            "frame": work,
        }

    y = pd.to_numeric(work["home_win"], errors="coerce")
    valid_target = y.isin([0, 1])
    work = work[valid_target].copy()
    y = y[valid_target].astype(int)

    if work.empty:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "no rows with valid target",
            "warnings": warnings,
            "errors": errors,
            "X": np.empty((0, len(features))),
            "y": np.array([], dtype=int),
            "features_used": features,
            "experimental_features_used": allowed_candidate,
            "frame": work,
        }

    X = work[features].to_numpy(dtype=float)

    return {
        "ok": True,
        "skipped": False,
        "skip_reason": "",
        "warnings": warnings,
        "errors": errors,
        "X": X,
        "y": y.to_numpy(dtype=int),
        "features_used": features,
        "experimental_features_used": allowed_candidate,
        "frame": work.reset_index(drop=True),
    }


def time_ordered_split(
    X: np.ndarray,
    y: np.ndarray,
    frame: Optional[pd.DataFrame] = None,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.15,
    validation_ratio: float = 0.15,
    min_train_samples: int = 20,
    min_calibration_samples: int = 5,
    min_validation_samples: int = 5,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []

    n = int(len(y))
    if n == 0 or len(X) != n:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "empty data or X/y length mismatch",
            "warnings": warnings,
            "errors": ["X/y length mismatch" if len(X) != n else "empty data"],
        }

    total_ratio = train_ratio + calibration_ratio + validation_ratio
    if total_ratio <= 0:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": "invalid split ratios",
            "warnings": warnings,
            "errors": ["invalid split ratios"],
        }

    train_ratio = train_ratio / total_ratio
    calibration_ratio = calibration_ratio / total_ratio

    train_end = int(n * train_ratio)
    calibration_end = int(n * (train_ratio + calibration_ratio))

    train_count = train_end
    calibration_count = calibration_end - train_end
    validation_count = n - calibration_end

    if train_count < min_train_samples:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": f"train split too small: {train_count} < {min_train_samples}",
            "warnings": warnings,
            "errors": errors,
            "train_count": train_count,
            "calibration_count": calibration_count,
            "validation_count": validation_count,
        }

    if calibration_count < min_calibration_samples:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": f"calibration split too small: {calibration_count} < {min_calibration_samples}",
            "warnings": warnings,
            "errors": errors,
            "train_count": train_count,
            "calibration_count": calibration_count,
            "validation_count": validation_count,
        }

    if validation_count < min_validation_samples:
        return {
            "ok": False,
            "skipped": True,
            "skip_reason": f"validation split too small: {validation_count} < {min_validation_samples}",
            "warnings": warnings,
            "errors": errors,
            "train_count": train_count,
            "calibration_count": calibration_count,
            "validation_count": validation_count,
        }

    result = {
        "ok": True,
        "skipped": False,
        "skip_reason": "",
        "warnings": warnings,
        "errors": errors,
        "train_count": train_count,
        "calibration_count": calibration_count,
        "validation_count": validation_count,
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_calibration": X[train_end:calibration_end],
        "y_calibration": y[train_end:calibration_end],
        "X_validation": X[calibration_end:],
        "y_validation": y[calibration_end:],
    }

    if frame is not None and not frame.empty:
        result["frame_train"] = frame.iloc[:train_end].copy()
        result["frame_calibration"] = frame.iloc[train_end:calibration_end].copy()
        result["frame_validation"] = frame.iloc[calibration_end:].copy()

    return result


def expected_calibration_error(
    y_true: Sequence[int],
    y_pred: Sequence[float],
    *,
    n_bins: int = 10,
) -> Optional[float]:
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_pred, dtype=float)

    if len(y_arr) == 0 or len(y_arr) != len(p_arr):
        return None

    valid = np.isfinite(p_arr)
    y_arr = y_arr[valid]
    p_arr = p_arr[valid]

    if len(y_arr) == 0:
        return None

    p_arr = np.clip(p_arr, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        if i == n_bins - 1:
            mask = (p_arr >= lower) & (p_arr <= upper)
        else:
            mask = (p_arr >= lower) & (p_arr < upper)

        if not np.any(mask):
            continue

        bin_confidence = float(np.mean(p_arr[mask]))
        bin_accuracy = float(np.mean(y_arr[mask]))
        bin_weight = float(np.mean(mask))
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[float],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []

    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_pred, dtype=float)

    if len(y_arr) == 0:
        return {
            "ok": False,
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "brier": None,
            "logloss": None,
            "accuracy": None,
            "auc": None,
            "ece": None,
            "warnings": warnings,
            "errors": ["empty y_true"],
        }

    if len(y_arr) != len(p_arr):
        return {
            "ok": False,
            "sample_count": int(len(y_arr)),
            "positive_count": 0,
            "negative_count": 0,
            "brier": None,
            "logloss": None,
            "accuracy": None,
            "auc": None,
            "ece": None,
            "warnings": warnings,
            "errors": ["y_true/y_pred length mismatch"],
        }

    valid = np.isfinite(p_arr)
    if not np.all(valid):
        warnings.append(f"dropped {int((~valid).sum())} rows with invalid probability")
        y_arr = y_arr[valid]
        p_arr = p_arr[valid]

    if len(y_arr) == 0:
        return {
            "ok": False,
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "brier": None,
            "logloss": None,
            "accuracy": None,
            "auc": None,
            "ece": None,
            "warnings": warnings,
            "errors": ["no valid probabilities"],
        }

    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        warnings.append("probabilities outside [0, 1] were clipped")

    p_arr = np.clip(p_arr, 0.0, 1.0)
    y_arr = y_arr.astype(int)

    positive_count = int(np.sum(y_arr == 1))
    negative_count = int(np.sum(y_arr == 0))
    hard_pred = (p_arr >= 0.5).astype(int)

    result = {
        "ok": True,
        "sample_count": int(len(y_arr)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "brier": None,
        "logloss": None,
        "accuracy": None,
        "auc": None,
        "ece": expected_calibration_error(y_arr, p_arr, n_bins=n_bins),
        "warnings": warnings,
        "errors": errors,
    }

    try:
        result["brier"] = float(brier_score_loss(y_arr, p_arr))
    except Exception as exc:
        warnings.append(f"brier unavailable: {exc}")

    try:
        result["logloss"] = float(log_loss(y_arr, np.clip(p_arr, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception as exc:
        warnings.append(f"logloss unavailable: {exc}")

    try:
        result["accuracy"] = float(accuracy_score(y_arr, hard_pred))
    except Exception as exc:
        warnings.append(f"accuracy unavailable: {exc}")

    if len(np.unique(y_arr)) >= 2:
        try:
            result["auc"] = float(roc_auc_score(y_arr, p_arr))
        except Exception as exc:
            warnings.append(f"auc unavailable: {exc}")
    else:
        warnings.append("auc unavailable: validation target has one class")

    return result
