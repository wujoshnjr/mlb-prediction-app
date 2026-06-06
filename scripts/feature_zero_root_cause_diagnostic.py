from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SOURCE_TO_FEATURES: Dict[str, List[str]] = {
    "weather": ["temp_effect", "wind_effect", "precip_effect"],
    "top3_player": ["top3_woba_diff"],
    "savant_top3": [
        "statcast_woba_diff",
        "statcast_barrel_diff",
        "statcast_hard_hit_diff",
        "statcast_launch_speed_diff",
        "avg_bat_speed_diff",
        "barrel_pa_diff",
        "hardhit_pa_diff",
    ],
    "pitcher_advanced": [
        "sp_fip_diff",
        "sp_csw_diff",
        "sp_stuff_plus_diff",
        "k_pct_diff",
        "bb_pct_diff",
    ],
    "team_form": [
        "lag30_winrate_diff",
        "lag30_runs_diff",
        "rest_diff",
        "back2back_diff",
        "games_last_3d_diff",
        "games_last_7d_diff",
        "rest_pressure_diff",
        "log5_prob",
        "elo_momentum_7d",
        "elo_momentum_30d",
    ],
    "context_bridge": [
        "bullpen_ip_diff",
        "bullpen_availability_diff",
    ],
}

SOURCE_FILES: Dict[str, str] = {
    "weather": "data/weather_context.csv",
    "top3_player": "data/top3_player_context.csv",
    "savant_top3": "data/savant_top3_context.csv",
    "pitcher_advanced": "data/pitcher_advanced_context.csv",
    "team_form": "data/team_form_context.csv",
    "context_bridge": "data/context_feature_bridge.csv",
}

EXPECTED_NEUTRAL_ZERO_FEATURES = {
    "rest_diff",
    "back2back_diff",
}

FEATURE_SIGNAL_ALTERNATIVES = {
    "rest_diff": ["rest_pressure_diff", "games_last_3d_diff", "games_last_7d_diff"],
    "back2back_diff": ["rest_pressure_diff", "games_last_3d_diff", "games_last_7d_diff"],
}

def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> tuple[Optional[Dict[str, Any]], str]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return None, "json_not_object"
        return data, ""
    except FileNotFoundError:
        return None, "file_missing"
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    except Exception as exc:
        return None, f"read_error: {exc}"


def _safe_read_csv(path: Path) -> tuple[Optional[pd.DataFrame], str]:
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, "file_missing"
    except Exception as exc:
        return None, f"read_error: {exc}"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _as_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return float(number)
    except (TypeError, ValueError):
        return None


def _extract_predictions(prediction_json: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not prediction_json:
        return []

    for key in ("today_predictions", "predictions", "games", "recommendations"):
        value = prediction_json.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _source_file_status(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    frame, error = _safe_read_csv(path)

    if frame is None:
        return {
            "path": path_text,
            "exists": False,
            "row_count": 0,
            "column_count": 0,
            "status": "missing",
            "error": error,
        }

    return {
        "path": path_text,
        "exists": True,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "status": "empty" if frame.empty else "present",
        "error": error,
    }


def _likely_root_cause(
    source: str,
    non_zero_count: int,
    source_file_status: Dict[str, Any],
) -> str:
    if not source_file_status.get("exists"):
        return "source_file_missing"

    if int(source_file_status.get("row_count", 0)) == 0:
        return "source_file_empty"

    if non_zero_count == 0:
        return "source_present_but_not_integrated_or_no_signal"

    return "repaired_or_active"


def _has_alternative_signal(
    feature: str,
    predictions: List[Dict[str, Any]],
) -> bool:
    alternatives = FEATURE_SIGNAL_ALTERNATIVES.get(feature, [])
    if not alternatives:
        return False

    for prediction in predictions:
        features = prediction.get("features")
        if not isinstance(features, dict):
            continue

        for alternative in alternatives:
            value = _as_float(features.get(alternative))
            if value is not None and value != 0.0:
                return True

    return False


def build_feature_zero_root_cause_diagnostic(
    prediction_path: str = "report/prediction.json",
    weather_path: str = "data/weather_context.csv",
    top3_player_path: str = "data/top3_player_context.csv",
    savant_top3_path: str = "data/savant_top3_context.csv",
    pitcher_adv_path: str = "data/pitcher_advanced_context.csv",
    team_form_path: str = "data/team_form_context.csv",
    context_bridge_path: str = "data/context_feature_bridge.csv",
    output_path: str = "report/feature_zero_root_cause_diagnostic.json",
) -> Dict[str, Any]:
    generated_at = _current_utc_iso()

    source_files = {
        "weather": weather_path,
        "top3_player": top3_player_path,
        "savant_top3": savant_top3_path,
        "pitcher_advanced": pitcher_adv_path,
        "team_form": team_form_path,
        "context_bridge": context_bridge_path,
    }

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "prediction_count": 0,
        "feature_zero_summary": [],
        "source_file_status": {},
        "source_to_feature_map": SOURCE_TO_FEATURES,
        "still_zero_features": [],
        "repaired_features": [],
        "recommendations": [],
    }

    prediction_json, prediction_error = _safe_read_json(Path(prediction_path))
    predictions = _extract_predictions(prediction_json)
    report["prediction_count"] = int(len(predictions))

    for source, path_text in source_files.items():
        report["source_file_status"][source] = _source_file_status(path_text)

    feature_to_source: Dict[str, str] = {}
    for source, features in SOURCE_TO_FEATURES.items():
        for feature in features:
            feature_to_source[feature] = source

    feature_stats: Dict[str, Dict[str, int]] = {
        feature: {
            "missing_count": 0,
            "zero_count": 0,
            "non_zero_count": 0,
            "total": 0,
        }
        for feature in feature_to_source
    }

    for prediction in predictions:
        features = prediction.get("features")
        if not isinstance(features, dict):
            continue

        for feature in feature_stats:
            feature_stats[feature]["total"] += 1
            raw_value = features.get(feature)

            number = _as_float(raw_value)
            if number is None:
                feature_stats[feature]["missing_count"] += 1
            elif number == 0.0:
                feature_stats[feature]["zero_count"] += 1
            else:
                feature_stats[feature]["non_zero_count"] += 1

    still_zero_features: List[str] = []
    repaired_features: List[str] = []
    feature_zero_summary: List[Dict[str, Any]] = []

    for feature, stats in feature_stats.items():
        total = int(stats["total"])
        source = feature_to_source.get(feature, "unknown")
        source_status = report["source_file_status"].get(source, {})
        non_zero_count = int(stats["non_zero_count"])

        zero_rate = (stats["zero_count"] / total) if total > 0 else 0.0
        non_zero_rate = (non_zero_count / total) if total > 0 else 0.0

        likely_root_cause = _likely_root_cause(
            source,
            non_zero_count,
            source_status,
        )

        item = {
            "feature": feature,
            "missing_count": int(stats["missing_count"]),
            "zero_count": int(stats["zero_count"]),
            "non_zero_count": non_zero_count,
            "zero_rate": round(float(zero_rate), 4),
            "non_zero_rate": round(float(non_zero_rate), 4),
            "source": source,
            "source_status": source_status.get("status", "unknown"),
            "likely_root_cause": likely_root_cause,
        }
        feature_zero_summary.append(item)

        expected_neutral_zero = (
            feature in EXPECTED_NEUTRAL_ZERO_FEATURES
            and _has_alternative_signal(feature, predictions)
        )

        item["expected_neutral_zero"] = bool(expected_neutral_zero)

        if non_zero_count > 0 or expected_neutral_zero:
            repaired_features.append(feature)
        else:
            still_zero_features.append(feature)
            
    report["feature_zero_summary"] = sorted(
        feature_zero_summary,
        key=lambda item: (item["source"], item["feature"]),
    )
    report["still_zero_features"] = sorted(still_zero_features)
    report["repaired_features"] = sorted(repaired_features)

    recommendations: List[str] = []

    if prediction_error:
        recommendations.append(f"Prediction file issue: {prediction_error}")

    for source, features in SOURCE_TO_FEATURES.items():
        status = report["source_file_status"].get(source, {})
        if status.get("status") == "missing":
            recommendations.append(
                f"Source file for {source} is missing: {status.get('path')}"
            )
            continue

        if status.get("status") == "empty":
            recommendations.append(
                f"Source file for {source} exists but is empty: {status.get('path')}"
            )
            continue

        source_still_zero = [
            feature for feature in features if feature in still_zero_features
        ]
        if source_still_zero:
            recommendations.append(
                f"{source} source exists but these features remain all-zero: {source_still_zero}"
            )

    if not recommendations:
        recommendations.append("All tracked feature sources have at least one non-zero feature.")

    report["recommendations"] = recommendations

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    diagnostic = build_feature_zero_root_cause_diagnostic()
    print(
        json.dumps(
            {
                "prediction_count": diagnostic["prediction_count"],
                "still_zero_features": diagnostic["still_zero_features"],
                "repaired_features": diagnostic["repaired_features"],
                "output_path": "report/feature_zero_root_cause_diagnostic.json",
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
