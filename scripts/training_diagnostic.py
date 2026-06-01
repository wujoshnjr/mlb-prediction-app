from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read a JSON file, return (data, error_message)."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return None, f"File {path} does not contain a JSON object"

        return data, ""

    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read a CSV file into a DataFrame, return (df, error_message)."""
    try:
        frame = pd.read_csv(path)
        return frame, ""

    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


def _column_exists(frame: pd.DataFrame, column: str) -> bool:
    """Check if a column exists in a DataFrame."""
    return isinstance(frame, pd.DataFrame) and column in frame.columns


def _safe_count(frame: Optional[pd.DataFrame]) -> int:
    """Return number of rows in a DataFrame, 0 if None."""
    if frame is None:
        return 0
    return int(len(frame))


def _clean_float(value: Any) -> Optional[float]:
    """Convert to float, return None for NaN/inf or unparseable."""
    try:
        if value is None:
            return None

        numeric_value = float(value)

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None

        return numeric_value

    except (ValueError, TypeError):
        return None


def _clean_int(value: Any) -> int:
    """Convert to int safely. Invalid values become 0."""
    try:
        if value is None:
            return 0

        numeric_value = float(value)

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return 0

        return int(numeric_value)

    except (ValueError, TypeError):
        return 0


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/NaN values into JSON-safe primitive values."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value

    if isinstance(value, (str, int, bool)):
        return value

    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                return ""
            return item
        except Exception:
            pass

    return str(value)


def _string_key(value: Any) -> str:
    """Return a clean lowercase string key."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""

    return text


# ---------------------------------------------------------------------------
# Diagnostic sections
# ---------------------------------------------------------------------------

def _process_training_status(path: str) -> Dict[str, Any]:
    """Process training_status.json."""
    status: Dict[str, Any] = {
        "exists": False,
        "error": "",
        "trained": False,
        "skipped": False,
        "sample_count": 0,
        "minimum_clean_train_samples": 0,
        "brier": "",
        "logloss": "",
        "reason": "",
        "model_type": "",
        "training_source": "",
        "pipeline_version": "",
        "timestamp": "",
    }

    data, error = _safe_read_json(Path(path))

    if data is None:
        status["error"] = error
        return status

    status["exists"] = True
    status["trained"] = bool(data.get("trained", False))
    status["skipped"] = bool(data.get("skipped", False))
    status["sample_count"] = _clean_int(data.get("sample_count"))
    status["minimum_clean_train_samples"] = _clean_int(
        data.get("minimum_clean_train_samples")
    )
    status["brier"] = _clean_float(data.get("brier"))
    status["logloss"] = _clean_float(data.get("logloss"))
    status["reason"] = _json_safe(data.get("reason"))
    status["model_type"] = _json_safe(data.get("model_type"))
    status["training_source"] = _json_safe(data.get("training_source"))
    status["pipeline_version"] = _json_safe(data.get("pipeline_version"))
    status["timestamp"] = _json_safe(data.get("timestamp"))

    if status["brier"] is None:
        status["brier"] = ""
    if status["logloss"] is None:
        status["logloss"] = ""

    return status


def _process_training_log(path: str) -> Dict[str, Any]:
    """Process training_log.csv."""
    log: Dict[str, Any] = {
        "exists": False,
        "error": "",
        "row_count": 0,
        "latest_num_samples": 0,
        "latest_used_feature_count": 0,
        "latest_brier": "",
        "latest_logloss": "",
        "latest_timestamp": "",
        "trend_available": False,
        "recent_training_runs": [],
    }

    frame, error = _safe_read_csv(Path(path))

    if frame is None:
        log["error"] = error
        return log

    log["exists"] = True
    log["row_count"] = int(len(frame))

    if frame.empty:
        log["error"] = "CSV file is empty"
        return log

    frame = frame.copy()

    if _column_exists(frame, "timestamp"):
        frame["timestamp_dt"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
            utc=True,
        )
        if frame["timestamp_dt"].notna().any():
            frame = frame.sort_values("timestamp_dt")

    latest = frame.iloc[-1]

    latest_brier = _clean_float(latest.get("brier"))
    latest_logloss = _clean_float(latest.get("logloss"))

    log["latest_num_samples"] = _clean_int(
        latest.get("num_samples", latest.get("sample_count", 0))
    )
    log["latest_used_feature_count"] = _clean_int(
        latest.get("used_feature_count", latest.get("feature_count", 0))
    )
    log["latest_brier"] = latest_brier if latest_brier is not None else ""
    log["latest_logloss"] = latest_logloss if latest_logloss is not None else ""
    log["latest_timestamp"] = _json_safe(latest.get("timestamp", ""))
    log["trend_available"] = bool(len(frame) >= 2)

    recent_runs: List[Dict[str, Any]] = []
    for _, row in frame.tail(5).iterrows():
        brier = _clean_float(row.get("brier"))
        logloss = _clean_float(row.get("logloss"))

        recent_runs.append(
            {
                "timestamp": _json_safe(row.get("timestamp", "")),
                "num_samples": _clean_int(
                    row.get("num_samples", row.get("sample_count", 0))
                ),
                "used_feature_count": _clean_int(
                    row.get("used_feature_count", row.get("feature_count", 0))
                ),
                "brier": brier if brier is not None else "",
                "logloss": logloss if logloss is not None else "",
            }
        )

    log["recent_training_runs"] = recent_runs

    return log


def _process_feature_importance(path: str) -> Dict[str, Any]:
    """Process feature_importance.csv, using the latest row."""
    feature_importance: Dict[str, Any] = {
        "exists": False,
        "error": "",
        "used_feature_count": 0,
        "top_features": [],
        "low_importance_features": [],
        "zero_or_missing_importance_features": [],
    }

    frame, error = _safe_read_csv(Path(path))

    if frame is None:
        feature_importance["error"] = error
        return feature_importance

    feature_importance["exists"] = True

    if frame.empty:
        feature_importance["error"] = "CSV file is empty"
        return feature_importance

    frame = frame.copy()

    if _column_exists(frame, "timestamp") and len(frame) > 1:
        frame["timestamp_dt"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
            utc=True,
        )
        if frame["timestamp_dt"].notna().any():
            frame = frame.sort_values("timestamp_dt")

    latest_row = frame.iloc[-1]

    metadata_columns = {
        "timestamp",
        "pipeline_version",
        "model_type",
        "timestamp_dt",
        "Unnamed: 0",
        "",
    }

    feature_columns = [
        column
        for column in frame.columns
        if column not in metadata_columns and not str(column).startswith("Unnamed")
    ]

    features: List[Tuple[str, Optional[float]]] = []

    for column in feature_columns:
        importance = _clean_float(latest_row.get(column))
        features.append((str(column), importance))

    positive_features = [
        (name, importance)
        for name, importance in features
        if importance is not None and importance > 0
    ]

    feature_importance["used_feature_count"] = int(len(positive_features))

    sorted_features = sorted(
        features,
        key=lambda item: item[1] if item[1] is not None else -1.0,
        reverse=True,
    )

    feature_importance["top_features"] = [
        {
            "feature": name,
            "importance": float(importance) if importance is not None else 0.0,
        }
        for name, importance in sorted_features[:15]
    ]

    feature_importance["low_importance_features"] = [
        {
            "feature": name,
            "importance": float(importance),
        }
        for name, importance in sorted_features
        if importance is not None and 0 < importance <= 0.02
    ]

    feature_importance["zero_or_missing_importance_features"] = [
        name
        for name, importance in sorted_features
        if importance is None or importance == 0
    ]

    return feature_importance


def _process_snapshots(path: str) -> Dict[str, Any]:
    """Process prediction_snapshots.csv."""
    snapshots: Dict[str, Any] = {
        "exists": False,
        "error": "",
        "total_rows": 0,
        "clean_pipeline_rows": 0,
        "valid_rows": 0,
        "settled_rows": 0,
        "clean_settled_rows": 0,
        "target_home_win_counts": {
            "home_win_0": 0,
            "home_win_1": 0,
        },
        "date_range": {
            "min": "",
            "max": "",
        },
        "recent_rows_sample_count": 0,
    }

    frame, error = _safe_read_csv(Path(path))

    if frame is None:
        snapshots["error"] = error
        return snapshots

    snapshots["exists"] = True
    snapshots["total_rows"] = int(len(frame))

    if frame.empty:
        snapshots["error"] = "CSV file is empty"
        return snapshots

    frame = frame.copy()

    clean_mask = pd.Series(True, index=frame.index)
    valid_mask = pd.Series(True, index=frame.index)
    settled_mask = pd.Series(False, index=frame.index)

    if _column_exists(frame, "pipeline_version"):
        clean_mask = frame["pipeline_version"].astype(str).eq("baseline_v2_clean")
        snapshots["clean_pipeline_rows"] = int(clean_mask.sum())

    if _column_exists(frame, "snapshot_valid"):
        valid_text = frame["snapshot_valid"].astype(str).str.lower()
        valid_mask = valid_text.isin(["true", "1", "yes"])
        snapshots["valid_rows"] = int(valid_mask.sum())

    if _column_exists(frame, "home_win"):
        home_win = pd.to_numeric(frame["home_win"], errors="coerce")
        settled_mask = home_win.isin([0, 1])

        snapshots["settled_rows"] = int(settled_mask.sum())
        snapshots["target_home_win_counts"]["home_win_0"] = int((home_win == 0).sum())
        snapshots["target_home_win_counts"]["home_win_1"] = int((home_win == 1).sum())

    snapshots["clean_settled_rows"] = int(
        (clean_mask & valid_mask & settled_mask).sum()
    )

    date_source = ""
    if _column_exists(frame, "snapshot_created_at"):
        date_source = "snapshot_created_at"
    elif _column_exists(frame, "game_date"):
        date_source = "game_date"

    if date_source:
        dates = pd.to_datetime(
            frame[date_source],
            errors="coerce",
            utc=True,
        )
        valid_dates = dates.dropna()

        if not valid_dates.empty:
            snapshots["date_range"]["min"] = valid_dates.min().isoformat()
            snapshots["date_range"]["max"] = valid_dates.max().isoformat()

            recent_cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=30)
            snapshots["recent_rows_sample_count"] = int(
                (valid_dates >= recent_cutoff).sum()
            )

    return snapshots


def _extract_prediction_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract prediction rows from supported prediction report shapes."""
    for key in (
        "today_predictions",
        "predictions",
        "games",
        "recommendations",
        "paper_bets",
    ):
        rows = report.get(key, [])
        if isinstance(rows, list) and rows:
            return [item for item in rows if isinstance(item, dict)]

    return []


def _process_prediction_usage(path: str) -> Dict[str, Any]:
    """Process report/prediction.json for model usage indicators."""
    usage: Dict[str, Any] = {
        "exists": False,
        "error": "",
        "prediction_count": 0,
        "model_source_counts": {},
        "model_feature_count_values": [],
        "model_training_sample_count_values": [],
        "model_load_error_count": 0,
        "sample_model_load_errors": [],
        "using_ml_count": 0,
        "using_manual_count": 0,
    }

    data, error = _safe_read_json(Path(path))

    if data is None:
        usage["error"] = error
        return usage

    usage["exists"] = True

    predictions = _extract_prediction_rows(data)
    usage["prediction_count"] = int(len(predictions))

    model_source_counts: Dict[str, int] = {}

    for prediction in predictions:
        model_source = _string_key(prediction.get("model_source"))
        if not model_source:
            model_source = "missing"

        model_source_counts[model_source] = model_source_counts.get(model_source, 0) + 1

        feature_count = prediction.get("model_feature_count")
        if feature_count is None:
            feature_count = prediction.get("used_feature_count")

        training_sample_count = prediction.get("model_training_sample_count")
        if training_sample_count is None:
            training_sample_count = prediction.get("training_sample_count")

        usage["model_feature_count_values"].append(_clean_int(feature_count))
        usage["model_training_sample_count_values"].append(
            _clean_int(training_sample_count)
        )

        load_error = _json_safe(
            prediction.get("model_load_error") or prediction.get("load_error")
        )
        if load_error:
            usage["model_load_error_count"] += 1
            if len(usage["sample_model_load_errors"]) < 5:
                usage["sample_model_load_errors"].append(load_error)

        if model_source in {"ml", "ensemble", "model"}:
            usage["using_ml_count"] += 1
        else:
            usage["using_manual_count"] += 1

    usage["model_source_counts"] = model_source_counts

    return usage


# ---------------------------------------------------------------------------
# Main diagnostic function
# ---------------------------------------------------------------------------

def build_training_diagnostic(
    training_status_path: str = "data/training_status.json",
    training_log_path: str = "data/training_log.csv",
    feature_importance_path: str = "data/feature_importance.csv",
    snapshot_path: str = "data/prediction_snapshots.csv",
    prediction_path: str = "report/prediction.json",
    output_path: str = "report/training_diagnostic.json",
) -> Dict[str, Any]:
    """Build a comprehensive training diagnostic report."""
    generated_at = datetime.now(timezone.utc).isoformat()

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "training_status": _process_training_status(training_status_path),
        "training_log": _process_training_log(training_log_path),
        "feature_importance": _process_feature_importance(feature_importance_path),
        "snapshots": _process_snapshots(snapshot_path),
        "prediction_usage": _process_prediction_usage(prediction_path),
        "model_stage": "unknown",
        "health_flags": [],
        "recommendations": [],
    }

    training_status = report["training_status"]
    training_log = report["training_log"]
    feature_importance = report["feature_importance"]
    snapshots = report["snapshots"]
    prediction_usage = report["prediction_usage"]

    trained = bool(training_status.get("trained", False))
    sample_count = _clean_int(training_status.get("sample_count"))
    brier = _clean_float(training_status.get("brier"))
    logloss = _clean_float(training_status.get("logloss"))
    using_ml_count = _clean_int(prediction_usage.get("using_ml_count"))
    using_manual_count = _clean_int(prediction_usage.get("using_manual_count"))

    if not trained and using_ml_count == 0:
        report["model_stage"] = "no_model"
    elif trained and sample_count < 150:
        report["model_stage"] = "early_model"
    elif trained and 150 <= sample_count < 300:
        report["model_stage"] = "developing_model"
    elif trained and sample_count >= 300 and brier is not None and logloss is not None:
        report["model_stage"] = "production_candidate"
    else:
        report["model_stage"] = "unknown"

    flags: List[str] = []

    if trained:
        flags.append("model_trained")
    else:
        flags.append("model_not_trained")

    if report["model_stage"] == "early_model":
        flags.append("early_model_small_sample")

    if using_ml_count > 0:
        flags.append("prediction_using_ml")
    else:
        flags.append("prediction_still_manual")

    feature_count = _clean_int(training_log.get("latest_used_feature_count"))
    if feature_count == 0:
        feature_count = _clean_int(feature_importance.get("used_feature_count"))

    if feature_count < 10:
        flags.append("feature_count_low")
    else:
        flags.append("feature_count_ok")

    home_win_0 = _clean_int(
        snapshots.get("target_home_win_counts", {}).get("home_win_0")
    )
    home_win_1 = _clean_int(
        snapshots.get("target_home_win_counts", {}).get("home_win_1")
    )
    total_targets = home_win_0 + home_win_1

    if total_targets > 0:
        min_class_ratio = min(home_win_0, home_win_1) / total_targets
        if min_class_ratio < 0.3:
            flags.append("target_class_imbalance")

    clean_settled_rows = _clean_int(snapshots.get("clean_settled_rows"))
    if clean_settled_rows < 300:
        flags.append("snapshot_count_below_production_threshold")

    if brier is not None and logloss is not None:
        flags.append("training_metrics_available")
    else:
        flags.append("training_metrics_missing")

    if _clean_int(prediction_usage.get("model_load_error_count")) > 0:
        flags.append("model_load_errors_present")

    if bool(feature_importance.get("exists")) and _clean_int(
        feature_importance.get("used_feature_count")
    ) > 0:
        flags.append("feature_importance_available")
    else:
        flags.append("feature_importance_missing")

    report["health_flags"] = flags

    recommendations: List[str] = []

    if report["model_stage"] == "early_model":
        recommendations.append(
            "Early model is for engineering validation only; do not use it as a production betting model."
        )

    if "prediction_still_manual" in flags:
        recommendations.append(
            "Prediction is still using manual fallback. Check data/calibrator.pkl and model loading in prediction.py."
        )

    if "feature_count_low" in flags:
        recommendations.append(
            "Very few features are active. Investigate snapshot feature extraction and zero-variance columns."
        )

    if "target_class_imbalance" in flags:
        recommendations.append(
            "Home win target distribution is imbalanced. Accumulate more settled snapshots before trusting model calibration."
        )

    if "snapshot_count_below_production_threshold" in flags:
        recommendations.append(
            "Continue collecting clean settled snapshots. Restore the 300+ sample production threshold when enough data exists."
        )

    if "model_load_errors_present" in flags:
        recommendations.append(
            "Model load errors detected. Check artifact compatibility, pickle/joblib versions, and feature alignment."
        )

    if "feature_importance_missing" in flags:
        recommendations.append(
            "Feature importance is missing or empty. Verify train_ensemble.py produced data/feature_importance.csv."
        )

    if not recommendations:
        recommendations.append(
            "Model diagnostic looks healthy. Continue monitoring Brier/logloss and collect more clean settled data."
        )

    report["recommendations"] = recommendations

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=True,
            default=str,
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    diagnostic = build_training_diagnostic()

    summary = {
        "generated_at": diagnostic["generated_at"],
        "model_stage": diagnostic["model_stage"],
        "training_status": {
            "trained": diagnostic["training_status"]["trained"],
            "sample_count": diagnostic["training_status"]["sample_count"],
            "brier": diagnostic["training_status"]["brier"],
            "logloss": diagnostic["training_status"]["logloss"],
        },
        "prediction_usage": {
            "using_ml": diagnostic["prediction_usage"]["using_ml_count"],
            "using_manual": diagnostic["prediction_usage"]["using_manual_count"],
        },
        "feature_importance": {
            "used_feature_count": diagnostic["feature_importance"]["used_feature_count"],
            "top_features": diagnostic["feature_importance"]["top_features"][:5],
        },
        "snapshots": {
            "clean_settled_rows": diagnostic["snapshots"]["clean_settled_rows"],
            "target_home_win_counts": diagnostic["snapshots"]["target_home_win_counts"],
        },
        "health_flags": diagnostic["health_flags"],
        "recommendations": diagnostic["recommendations"],
        "full_report_written_to": "report/training_diagnostic.json",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
