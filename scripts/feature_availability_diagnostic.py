from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read JSON file into a dict, return (data, error)."""
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if not isinstance(data, dict):
            return None, f"File {path} is not a JSON object"
        return data, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read CSV file into a DataFrame, return (df, error)."""
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


def _current_utc_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to finite float, else None."""
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    """Return clean string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if value is None or isinstance(value, str):
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return _json_safe(value.item())
    except Exception:
        return str(value)


def _extract_predictions(prediction_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract predictions from common report structures."""
    for key in ("today_predictions", "predictions", "games", "recommendations"):
        value = prediction_json.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _load_latest_feature_importance(
    feature_importance_path: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Load latest feature importance row."""
    frame, error = _safe_read_csv(feature_importance_path)

    summary: Dict[str, Any] = {
        "exists": frame is not None,
        "error": error,
        "row_count": int(len(frame)) if frame is not None else 0,
        "feature_count": 0,
    }

    importance: Dict[str, float] = {}

    if frame is None or frame.empty:
        return importance, summary

    metadata_cols = {
        "timestamp",
        "pipeline_version",
        "model_type",
        "Unnamed: 0",
        "",
    }

    working = frame.copy()

    if "timestamp" in working.columns and len(working) > 1:
        working["timestamp_dt"] = pd.to_datetime(
            working["timestamp"],
            errors="coerce",
            utc=True,
        )
        working = working.sort_values("timestamp_dt")

    latest = working.iloc[-1]

    for column in working.columns:
        if column in metadata_cols or str(column).startswith("Unnamed"):
            continue
        value = _safe_float(latest.get(column))
        if value is not None:
            importance[str(column)] = value

    summary["feature_count"] = int(len(importance))
    return importance, summary


def _feature_status(
    *,
    missing_rate: float,
    zero_rate: float,
    non_zero_rate: float,
) -> str:
    """Classify feature availability health."""
    if missing_rate >= 0.95:
        return "missing"
    if zero_rate >= 0.95:
        return "all_zero"
    if non_zero_rate >= 0.50:
        return "healthy"
    if non_zero_rate > 0:
        return "sparse"
    return "missing"


def _feature_group(feature_name: str) -> str:
    """Group features by likely source/domain."""
    name = feature_name.lower()

    if name.startswith("statcast") or any(
        token in name
        for token in (
            "barrel",
            "hard_hit",
            "hardhit",
            "woba",
            "launch",
            "bat_speed",
            "swing_miss",
        )
    ):
        return "statcast"

    if name.startswith("sp_") or "pitcher" in name or "csw" in name or "k_pct" in name or "bb_pct" in name:
        return "pitching"

    if "bullpen" in name or "closer" in name:
        return "bullpen"

    if "lineup" in name or "top3" in name or "platoon" in name:
        return "lineup"

    if "catcher" in name or name in {"cs_diff"}:
        return "catcher"

    if "weather" in name or "wind" in name or "temp" in name or "precip" in name:
        return "weather"

    if "umpire" in name or "zone" in name:
        return "umpire"

    if "elo" in name or "rating" in name or "winrate" in name or "strength" in name:
        return "team_strength"

    if "park" in name:
        return "park"

    if "timezone" in name or "day_game" in name or "back2back" in name or "rest" in name:
        return "schedule"

    return "other"


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def build_feature_availability_diagnostic(
    prediction_path: str = "report/prediction.json",
    feature_importance_path: str = "data/feature_importance.csv",
    output_path: str = "report/feature_availability_diagnostic.json",
) -> Dict[str, Any]:
    """
    Build feature availability diagnostic from prediction report and feature importance.

    The diagnostic checks:
    - missing / zero / non-zero rate per feature
    - latest feature importance
    - high-importance features that are sparse or all-zero
    - source-level groups such as statcast, pitcher, lineup, weather, umpire
    """
    generated_at = _current_utc_iso()

    prediction_json, prediction_error = _safe_read_json(Path(prediction_path))
    importance_map, importance_summary = _load_latest_feature_importance(
        Path(feature_importance_path)
    )

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "input_files": {
            "prediction": {
                "exists": prediction_json is not None,
                "error": prediction_error,
            },
            "feature_importance": importance_summary,
        },
        "prediction_count": 0,
        "prediction_with_features_count": 0,
        "feature_count": 0,
        "feature_availability": [],
        "group_summary": {},
        "high_risk_features": [],
        "all_zero_features": [],
        "sparse_features": [],
        "missing_features": [],
        "recommendations": [],
    }

    if prediction_json is None:
        report["recommendations"] = [
            f"Prediction file error: {prediction_error}",
        ]
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file_obj:
            json.dump(_json_safe(report), file_obj, indent=2, ensure_ascii=True)
        return _json_safe(report)

    predictions = _extract_predictions(prediction_json)
    report["prediction_count"] = int(len(predictions))

    feature_dicts: List[Dict[str, Any]] = []
    for prediction in predictions:
        features = prediction.get("features")
        if isinstance(features, dict):
            feature_dicts.append(features)

    report["prediction_with_features_count"] = int(len(feature_dicts))

    feature_names = set(importance_map.keys())
    for features in feature_dicts:
        feature_names.update(str(name) for name in features.keys())

    feature_rows: List[Dict[str, Any]] = []

    total_predictions = len(feature_dicts)

    for feature_name in sorted(feature_names):
        missing_count = 0
        zero_count = 0
        non_zero_count = 0

        for features in feature_dicts:
            if feature_name not in features:
                missing_count += 1
                continue

            value = _safe_float(features.get(feature_name))
            if value is None:
                missing_count += 1
            elif value == 0.0:
                zero_count += 1
            else:
                non_zero_count += 1

        denominator = total_predictions if total_predictions > 0 else 1

        missing_rate = missing_count / denominator
        zero_rate = zero_count / denominator
        non_zero_rate = non_zero_count / denominator

        status = _feature_status(
            missing_rate=missing_rate,
            zero_rate=zero_rate,
            non_zero_rate=non_zero_rate,
        )

        latest_importance = importance_map.get(feature_name, 0.0)
        group = _feature_group(feature_name)

        feature_rows.append(
            {
                "feature": feature_name,
                "group": group,
                "missing_count": int(missing_count),
                "zero_count": int(zero_count),
                "non_zero_count": int(non_zero_count),
                "missing_rate": round(missing_rate, 4),
                "zero_rate": round(zero_rate, 4),
                "non_zero_rate": round(non_zero_rate, 4),
                "latest_importance": round(float(latest_importance), 8),
                "status": status,
            }
        )

    feature_rows = sorted(
        feature_rows,
        key=lambda item: (
            -float(item.get("latest_importance") or 0.0),
            str(item.get("feature") or ""),
        ),
    )

    report["feature_count"] = int(len(feature_rows))
    report["feature_availability"] = feature_rows

    high_risk_features = []
    all_zero_features = []
    sparse_features = []
    missing_features = []

    for item in feature_rows:
        feature = str(item["feature"])
        status = str(item["status"])
        importance = float(item.get("latest_importance") or 0.0)

        if status == "all_zero":
            all_zero_features.append(feature)
        elif status == "sparse":
            sparse_features.append(feature)
        elif status == "missing":
            missing_features.append(feature)

        if importance > 0.03 and status in {"all_zero", "sparse", "missing"}:
            high_risk_features.append(feature)

    report["high_risk_features"] = high_risk_features
    report["all_zero_features"] = all_zero_features
    report["sparse_features"] = sparse_features
    report["missing_features"] = missing_features

    group_summary: Dict[str, Dict[str, Any]] = {}

    for item in feature_rows:
        group = str(item["group"])
        group_info = group_summary.setdefault(
            group,
            {
                "feature_count": 0,
                "healthy_count": 0,
                "sparse_count": 0,
                "all_zero_count": 0,
                "missing_count": 0,
                "avg_non_zero_rate": 0.0,
                "features": [],
            },
        )

        status = str(item["status"])
        group_info["feature_count"] += 1
        group_info["features"].append(item["feature"])
        group_info["avg_non_zero_rate"] += float(item["non_zero_rate"])

        if status == "healthy":
            group_info["healthy_count"] += 1
        elif status == "sparse":
            group_info["sparse_count"] += 1
        elif status == "all_zero":
            group_info["all_zero_count"] += 1
        elif status == "missing":
            group_info["missing_count"] += 1

    for group, group_info in group_summary.items():
        feature_count = int(group_info["feature_count"])
        if feature_count > 0:
            group_info["avg_non_zero_rate"] = round(
                float(group_info["avg_non_zero_rate"]) / feature_count,
                4,
            )

    report["group_summary"] = group_summary

    recommendations: List[str] = []

    if report["prediction_with_features_count"] == 0:
        recommendations.append(
            "No prediction rows contain a features dict; check prediction.py feature serialization."
        )

    if high_risk_features:
        recommendations.append(
            "Some high-importance features are sparse, missing, or all-zero: "
            + ", ".join(high_risk_features[:10])
        )

    statcast_group = group_summary.get("statcast", {})
    if statcast_group and int(statcast_group.get("all_zero_count", 0)) > 0:
        recommendations.append(
            "Statcast-related features are all-zero or sparse; check Savant top-3 context and prediction feature mapping."
        )

    pitching_group = group_summary.get("pitching", {})
    if pitching_group and int(pitching_group.get("all_zero_count", 0)) > 0:
        recommendations.append(
            "Some pitching features are all-zero; check probable pitcher enrichment, FIP, CSW, Stuff+, K%, and BB% sources."
        )

    lineup_group = group_summary.get("lineup", {})
    if lineup_group and (
        int(lineup_group.get("all_zero_count", 0)) > 0
        or int(lineup_group.get("missing_count", 0)) > 0
    ):
        recommendations.append(
            "Lineup/platoon/top3 features are incomplete; projected or confirmed lineup integration should be prioritized."
        )

    catcher_group = group_summary.get("catcher", {})
    if catcher_group and (
        int(catcher_group.get("all_zero_count", 0)) > 0
        or int(catcher_group.get("missing_count", 0)) > 0
    ):
        recommendations.append(
            "Catcher-related features are incomplete; verify catcher context source and catcher ERA / caught-stealing mapping."
        )

    weather_group = group_summary.get("weather", {})
    if weather_group and (
        int(weather_group.get("all_zero_count", 0)) > 0
        or int(weather_group.get("missing_count", 0)) > 0
    ):
        recommendations.append(
            "Weather-related features are incomplete; verify Open-Meteo integration and weather-to-feature mapping."
        )

    umpire_group = group_summary.get("umpire", {})
    if umpire_group and (
        int(umpire_group.get("all_zero_count", 0)) > 0
        or int(umpire_group.get("missing_count", 0)) > 0
    ):
        recommendations.append(
            "Umpire-related features are incomplete; verify umpire source and zone-size feature mapping."
        )

    if not recommendations:
        recommendations.append(
            "Feature availability looks healthy; continue monitoring high-importance features."
        )

    report["recommendations"] = recommendations

    safe_report = _json_safe(report)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_obj:
        json.dump(safe_report, file_obj, indent=2, ensure_ascii=True)

    return safe_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    diagnostic = build_feature_availability_diagnostic()

    summary = {
        "generated_at": diagnostic.get("generated_at"),
        "prediction_count": diagnostic.get("prediction_count"),
        "prediction_with_features_count": diagnostic.get(
            "prediction_with_features_count"
        ),
        "feature_count": diagnostic.get("feature_count"),
        "high_risk_features": diagnostic.get("high_risk_features"),
        "all_zero_features": diagnostic.get("all_zero_features"),
        "sparse_features": diagnostic.get("sparse_features"),
        "recommendations": diagnostic.get("recommendations"),
        "report_written_to": "report/feature_availability_diagnostic.json",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
