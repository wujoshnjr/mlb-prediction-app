from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_CONTEXT_COLUMNS = [
    "home_starter_status",
    "away_starter_status",
    "home_starter_confidence",
    "away_starter_confidence",
    "home_starter_confidence_score",
    "away_starter_confidence_score",
    "home_closer_available_known",
    "away_closer_available_known",
    "home_closer_status",
    "away_closer_status",
    "home_closer_risk_score",
    "away_closer_risk_score",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_lineup_player_count",
    "away_lineup_player_count",
]


# ---------------------------------------------------------------------------
# Safe helpers
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read CSV into a DataFrame, return (df, error) tuple."""
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read JSON file into a dict, return (data, error)."""
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)

        if not isinstance(data, dict):
            return None, f"File {path} does not contain a JSON object"

        return data, ""

    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _safe_bool(value: Any) -> Optional[bool]:
    """Interpret a value as bool, returning None if ambiguous."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"true", "1", "yes", "y"}:
            return True

        if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
            return False

    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if value == 1:
            return True

        if value == 0:
            return False

    return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert to int, returning None on failure."""
    try:
        if value is None:
            return None

        numeric_value = float(value)

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None

        return int(numeric_value)

    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> str:
    """Return a clean string. Null-like values become empty string."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def _current_utc_iso() -> str:
    """Return current UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _count_values(values: List[Any]) -> Dict[str, int]:
    """Count values as clean strings."""
    counts: Dict[str, int] = {}

    for value in values:
        key = _safe_str(value) or "unknown"
        counts[key] = counts.get(key, 0) + 1

    return counts


def _extract_predictions(prediction_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return prediction rows from common report structures."""
    for key in ("predictions", "today_predictions", "games", "recommendations"):
        value = prediction_json.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _daily_summary(prediction_row: Dict[str, Any]) -> Dict[str, Any]:
    """Return nested daily_context_summary safely."""
    summary = prediction_row.get("daily_context_summary")

    if isinstance(summary, dict):
        return summary

    return {}


def _classify_lineup_status_from_values(
    home_confirmed: Any,
    away_confirmed: Any,
    home_count: Any,
    away_count: Any,
) -> str:
    """Classify lineup readiness from confirmed flags and player counts."""
    home_conf = _safe_bool(home_confirmed)
    away_conf = _safe_bool(away_confirmed)
    home_lineup_count = _safe_int(home_count) or 0
    away_lineup_count = _safe_int(away_count) or 0

    if home_conf is True and away_conf is True:
        return "confirmed"

    if home_conf is True or away_conf is True:
        return "partial_confirmed"

    if home_lineup_count >= 7 and away_lineup_count >= 7:
        return "projected_available"

    return "pending"


def _classify_lineup_status(row: pd.Series) -> str:
    """Determine lineup status for a single context row."""
    return _classify_lineup_status_from_values(
        home_confirmed=row.get("home_lineup_confirmed"),
        away_confirmed=row.get("away_lineup_confirmed"),
        home_count=row.get("home_lineup_player_count"),
        away_count=row.get("away_lineup_player_count"),
    )


def _latest_context_rows(context_frame: pd.DataFrame) -> pd.DataFrame:
    """Return latest row per game_id by captured_at."""
    if (
        context_frame.empty
        or "captured_at" not in context_frame.columns
        or "game_id" not in context_frame.columns
    ):
        return pd.DataFrame()

    frame = context_frame.copy()
    frame["captured_at_dt"] = pd.to_datetime(
        frame["captured_at"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["captured_at_dt", "game_id"])

    if frame.empty:
        return pd.DataFrame()

    latest_indexes = frame.groupby("game_id")["captured_at_dt"].idxmax()
    return frame.loc[latest_indexes].copy()


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, str)) or value is None:
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return value.item()
    except Exception:
        pass

    return str(value)


# ---------------------------------------------------------------------------
# Main diagnostic function
# ---------------------------------------------------------------------------

def build_context_diagnostic(
    context_path: str = "data/daily_game_context.csv",
    prediction_path: str = "report/prediction.json",
    output_path: str = "report/context_diagnostic.json",
) -> Dict[str, Any]:
    """Build a diagnostic report for the pregame context pipeline."""
    generated_at = _current_utc_iso()

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "daily_context": {},
        "prediction_context": {},
        "schema_checks": {},
        "latest_snapshot_checks": {},
        "recommendations": [],
    }

    # ------------------------------------------------------------------
    # 1. Daily context file checks
    # ------------------------------------------------------------------
    context_frame, context_error = _safe_read_csv(Path(context_path))

    daily_context: Dict[str, Any] = {
        "exists": context_frame is not None,
        "error": context_error,
        "row_count": 0,
        "column_count": 0,
        "schema_versions": [],
        "latest_captured_at": "",
        "latest_game_count": 0,
        "columns_present": [],
        "columns_missing": list(REQUIRED_CONTEXT_COLUMNS),
    }

    if context_frame is not None:
        daily_context["row_count"] = int(len(context_frame))
        daily_context["column_count"] = int(len(context_frame.columns))

        if "context_schema_version" in context_frame.columns:
            versions = (
                context_frame["context_schema_version"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            daily_context["schema_versions"] = sorted(versions)

        if "captured_at" in context_frame.columns:
            captured_times = pd.to_datetime(
                context_frame["captured_at"],
                errors="coerce",
                utc=True,
            )
            if captured_times.notna().any():
                latest_captured_at = captured_times.max()
                daily_context["latest_captured_at"] = latest_captured_at.isoformat()

                latest_mask = captured_times == latest_captured_at
                if "game_id" in context_frame.columns:
                    latest_games = (
                        context_frame.loc[latest_mask, "game_id"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    daily_context["latest_game_count"] = int(len(latest_games))

        columns = set(context_frame.columns)
        daily_context["columns_present"] = [
            column for column in REQUIRED_CONTEXT_COLUMNS if column in columns
        ]
        daily_context["columns_missing"] = [
            column for column in REQUIRED_CONTEXT_COLUMNS if column not in columns
        ]

    report["daily_context"] = daily_context

    # ------------------------------------------------------------------
    # 2. Latest context snapshot checks
    # ------------------------------------------------------------------
    latest_check: Dict[str, Any] = {
        "latest_rows": 0,
        "starter_status_counts": {"home": {}, "away": {}},
        "lineup_status_estimate_counts": {
            "confirmed": 0,
            "partial_confirmed": 0,
            "projected_available": 0,
            "pending": 0,
        },
        "closer_known_count": 0,
        "high_confidence_starter_count": 0,
        "projected_lineup_available_count": 0,
        "confirmed_lineup_count": 0,
    }

    latest_context = (
        _latest_context_rows(context_frame)
        if context_frame is not None
        else pd.DataFrame()
    )

    if not latest_context.empty:
        latest_check["latest_rows"] = int(len(latest_context))

        for side in ("home", "away"):
            status_column = f"{side}_starter_status"
            if status_column in latest_context.columns:
                latest_check["starter_status_counts"][side] = _count_values(
                    latest_context[status_column].tolist()
                )
            else:
                latest_check["starter_status_counts"][side] = {
                    "unknown": int(len(latest_context))
                }

        lineup_statuses = latest_context.apply(_classify_lineup_status, axis=1)
        lineup_counts = _count_values(lineup_statuses.tolist())

        for key in latest_check["lineup_status_estimate_counts"]:
            latest_check["lineup_status_estimate_counts"][key] = int(
                lineup_counts.get(key, 0)
            )

        latest_check["projected_lineup_available_count"] = int(
            latest_check["lineup_status_estimate_counts"].get(
                "projected_available",
                0,
            )
        )
        latest_check["confirmed_lineup_count"] = int(
            latest_check["lineup_status_estimate_counts"].get("confirmed", 0)
        )

        if (
            "home_closer_available_known" in latest_context.columns
            and "away_closer_available_known" in latest_context.columns
        ):
            home_known = (
                latest_context["home_closer_available_known"].apply(_safe_bool)
                is not None
            )
            home_known = latest_context["home_closer_available_known"].apply(
                lambda value: _safe_bool(value) is True
            )
            away_known = latest_context["away_closer_available_known"].apply(
                lambda value: _safe_bool(value) is True
            )
            latest_check["closer_known_count"] = int((home_known & away_known).sum())

        home_high = pd.Series(False, index=latest_context.index)
        away_high = pd.Series(False, index=latest_context.index)

        if "home_starter_confidence" in latest_context.columns:
            home_high = latest_context["home_starter_confidence"].apply(
                lambda value: _safe_bool(value) is True
            )

        if "away_starter_confidence" in latest_context.columns:
            away_high = latest_context["away_starter_confidence"].apply(
                lambda value: _safe_bool(value) is True
            )

        latest_check["high_confidence_starter_count"] = int(
            (home_high | away_high).sum()
        )

    report["latest_snapshot_checks"] = latest_check

    # ------------------------------------------------------------------
    # 3. Prediction context checks
    # ------------------------------------------------------------------
    prediction_json, prediction_error = _safe_read_json(Path(prediction_path))
    predictions = _extract_predictions(prediction_json) if prediction_json else []

    prediction_context: Dict[str, Any] = {
        "exists": prediction_json is not None,
        "error": prediction_error,
        "prediction_count": int(len(predictions)),
        "pitcher_status_counts": {},
        "lineup_status_counts": {},
        "starter_confidence_status_counts": {},
        "missing_critical_field_counts": {},
        "context_status_counts": {},
        "betting_readiness_status_counts": {},
        "live_bet_candidate_count": 0,
        "average_betting_readiness_score": None,
        "risk_flag_counts": {},
        "practical_ready_count": 0,
        "official_ready_count": 0,
        "risk_blocked_count": 0,
    }

    pitcher_status_values: List[str] = []
    lineup_status_values: List[str] = []
    starter_confidence_values: List[str] = []
    context_status_values: List[str] = []
    missing_field_counts: Dict[str, int] = {}
    betting_score_values: List[float] = []

    for prediction in predictions:
        summary = _daily_summary(prediction)

        pitcher_status_values.append(summary.get("pitcher_status") or "unknown")
        lineup_status_values.append(summary.get("lineup_status") or "unknown")
        starter_confidence_values.append(
            summary.get("starter_confidence_status") or "unknown"
        )
        context_status_values.append(summary.get("status") or "unknown")

        missing_fields = summary.get("missing_critical_fields") or []
        if isinstance(missing_fields, list):
            if not missing_fields:
                missing_field_counts["none"] = (
                    missing_field_counts.get("none", 0) + 1
                )
            for field in missing_fields:
                field_key = _safe_str(field) or "unknown"
                missing_field_counts[field_key] = (
                    missing_field_counts.get(field_key, 0) + 1
                )
        else:
            missing_field_counts["unparseable"] = (
                missing_field_counts.get("unparseable", 0) + 1
            )

        readiness = prediction.get("betting_readiness") or {}
        readiness_status = str(
            prediction.get("betting_readiness_status")
            or readiness.get("betting_readiness_status")
            or "unknown"
        )

        betting_status_counts = prediction_context[
            "betting_readiness_status_counts"
        ]
        betting_status_counts[readiness_status] = (
            betting_status_counts.get(readiness_status, 0) + 1
        )

        if readiness_status == "official_ready":
            prediction_context["official_ready_count"] += 1
        elif readiness_status == "practical_ready":
            prediction_context["practical_ready_count"] += 1
        elif readiness_status == "risk_blocked":
            prediction_context["risk_blocked_count"] += 1

        if prediction.get("live_bet_candidate") is True:
            prediction_context["live_bet_candidate_count"] += 1

        score = prediction.get("betting_readiness_score")
        if score is None:
            score = readiness.get("betting_readiness_score")

        try:
            score_float = float(score)
            if not math.isnan(score_float) and not math.isinf(score_float):
                betting_score_values.append(score_float)
        except (TypeError, ValueError):
            pass

        flags = prediction.get("betting_risk_flags")
        if flags is None:
            flags = readiness.get("betting_risk_flags")

        if isinstance(flags, list):
            risk_counts = prediction_context["risk_flag_counts"]
            for flag in flags:
                key = str(flag)
                risk_counts[key] = risk_counts.get(key, 0) + 1

    prediction_context["pitcher_status_counts"] = _count_values(
        pitcher_status_values
    )
    prediction_context["lineup_status_counts"] = _count_values(lineup_status_values)
    prediction_context["starter_confidence_status_counts"] = _count_values(
        starter_confidence_values
    )
    prediction_context["missing_critical_field_counts"] = missing_field_counts
    prediction_context["context_status_counts"] = _count_values(context_status_values)

    if betting_score_values:
        prediction_context["average_betting_readiness_score"] = round(
            sum(betting_score_values) / len(betting_score_values),
            4,
        )

    report["prediction_context"] = prediction_context

    # ------------------------------------------------------------------
    # 4. Schema checks
    # ------------------------------------------------------------------
    context_columns = set(context_frame.columns) if context_frame is not None else set()

    starter_columns = {
        "home_starter_status",
        "away_starter_status",
        "home_starter_confidence",
        "away_starter_confidence",
        "home_starter_confidence_score",
        "away_starter_confidence_score",
    }
    closer_columns = {
        "home_closer_available_known",
        "away_closer_available_known",
        "home_closer_status",
        "away_closer_status",
        "home_closer_risk_score",
        "away_closer_risk_score",
    }

    prediction_has_starter_summary = any(
        bool(_daily_summary(prediction).get("starter_confidence_status"))
        or bool(_daily_summary(prediction).get("home_starter_status"))
        or bool(_daily_summary(prediction).get("away_starter_status"))
        for prediction in predictions
    )

    missing_only_lineup_when_starter_high_confidence = False
    starter_confirmation_still_missing = False

    for prediction in predictions:
        summary = _daily_summary(prediction)

        pitcher_status = _safe_str(summary.get("pitcher_status"))
        starter_confidence_status = _safe_str(
            summary.get("starter_confidence_status")
        )
        is_high_confidence = (
            pitcher_status in {"confirmed", "high_confidence_probable"}
            or starter_confidence_status == "known"
        )

        if not is_high_confidence:
            continue

        missing_fields = summary.get("missing_critical_fields") or []
        if not isinstance(missing_fields, list):
            continue

        non_lineup_missing = [
            field
            for field in missing_fields
            if "lineup" not in _safe_str(field).lower()
        ]

        if not non_lineup_missing:
            missing_only_lineup_when_starter_high_confidence = True

        if any(
            "starting_pitcher_confirmed" in _safe_str(field)
            for field in missing_fields
        ):
            starter_confirmation_still_missing = True

    prediction_has_betting_readiness = any(
        "betting_readiness" in prediction
        or "betting_readiness_status" in prediction
        for prediction in predictions
    )

    practical_ready_exists_when_missing_only_starter_confirmation = any(
        (
            prediction.get("betting_readiness_status") == "practical_ready"
            and prediction.get("effective_context_ready_for_betting") is True
            and _daily_summary(prediction).get("starter_confirmation_pending") is True
        )
        for prediction in predictions
    )

    schema_checks: Dict[str, Any] = {
        "context_has_starter_confidence_columns": starter_columns.issubset(
            context_columns
        ),
        "context_has_closer_known_columns": closer_columns.issubset(context_columns),
        "prediction_has_starter_confidence_summary": prediction_has_starter_summary,
        "prediction_missing_only_lineup_when_starter_high_confidence": (
            missing_only_lineup_when_starter_high_confidence
        ),
        "starter_confirmation_still_in_filtered_missing_fields": (
            starter_confirmation_still_missing
        ),
        "prediction_has_betting_readiness": prediction_has_betting_readiness,
        "practical_ready_exists_when_missing_only_starter_confirmation": (
            practical_ready_exists_when_missing_only_starter_confirmation
        ),
    }

    report["schema_checks"] = schema_checks

    # ------------------------------------------------------------------
    # 5. Recommendations
    # ------------------------------------------------------------------
    recommendations: List[str] = []

    if not schema_checks["context_has_starter_confidence_columns"]:
        recommendations.append(
            "Starter confidence columns are missing from daily_game_context.csv. "
            "Check scripts/daily_game_context.py COLUMNS and daily_context_collector integration."
        )

    if not schema_checks["context_has_closer_known_columns"]:
        recommendations.append(
            "Closer known/status/risk columns are missing from daily_game_context.csv. "
            "Check scripts/daily_game_context.py COLUMNS and closer_context_client integration."
        )

    if (
        schema_checks["context_has_starter_confidence_columns"]
        and not schema_checks["prediction_has_starter_confidence_summary"]
    ):
        recommendations.append(
            "Starter confidence exists in context but not prediction summary. "
            "Check prediction.py latest context mapping and build_daily_context_summary()."
        )

    if schema_checks["starter_confirmation_still_in_filtered_missing_fields"]:
        recommendations.append(
            "High-confidence starter rows still include starting_pitcher_confirmed "
            "inside filtered missing_critical_fields. Check context v3 filtering in prediction.py."
        )

    latest_rows = int(latest_check.get("latest_rows", 0) or 0)
    pending_lineups = int(
        latest_check["lineup_status_estimate_counts"].get("pending", 0) or 0
    )

    if latest_rows > 0 and pending_lineups == latest_rows:
        recommendations.append(
            "All latest lineups are pending. This can be normal before official lineup release; "
            "next upgrade should add a projected lineup source."
        )

    if latest_rows > 0 and int(latest_check.get("closer_known_count", 0) or 0) == 0:
        recommendations.append(
            "Closer availability is unknown for all latest games. "
            "Check closer_context_client, daily_context_collector, and schema persistence."
        )

    practical_ready_count = int(
        prediction_context.get("practical_ready_count", 0) or 0
    )
    if practical_ready_count > 0:
        recommendations.append(
            "Some games are practical_ready: official starter confirmation is pending, "
            "but effective betting context is available with reduced stake multiplier."
        )

    live_bet_candidate_count = int(
        prediction_context.get("live_bet_candidate_count", 0) or 0
    )
    if live_bet_candidate_count > 0:
        recommendations.append(
            f"{live_bet_candidate_count} live bet candidate(s) passed effective context, odds, model, and stake filters."
        )

    risk_flag_counts = prediction_context.get("risk_flag_counts", {})
    if isinstance(risk_flag_counts, dict):
        if risk_flag_counts.get("closer_high_fatigue", 0) > 0:
            recommendations.append(
                "Closer high fatigue risk detected; consider conservative stake multiplier or manual review."
            )
        if risk_flag_counts.get("early_model_small_sample", 0) > 0:
            recommendations.append(
                "Early model guard is active. Keep live betting disabled until sample size and CLV improve."
            )
        if risk_flag_counts.get("lineup_not_confirmed", 0) > 0:
            recommendations.append(
                "Lineup confirmation remains a betting blocker. Prioritize projected/confirmed lineup integration."
            )
        if risk_flag_counts.get("large_anti_market_edge_early_model", 0) > 0:
            recommendations.append(
                "Large anti-market edges are being blocked while the model is still early-stage."
            )
        
    if not recommendations:
        recommendations.append(
            "Context pipeline schema and prediction summary look healthy."
        )

    report["recommendations"] = recommendations

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    safe_report = _json_safe(report)

    with output_file.open("w", encoding="utf-8") as file_obj:
        json.dump(safe_report, file_obj, indent=2, ensure_ascii=True)

    return safe_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    diagnostic = build_context_diagnostic()

    summary = {
        "generated_at": diagnostic.get("generated_at"),
        "daily_context_rows": diagnostic.get("daily_context", {}).get("row_count", 0),
        "prediction_count": diagnostic.get("prediction_context", {}).get(
            "prediction_count",
            0,
        ),
        "context_has_starter_confidence_columns": diagnostic.get(
            "schema_checks",
            {},
        ).get("context_has_starter_confidence_columns"),
        "context_has_closer_known_columns": diagnostic.get(
            "schema_checks",
            {},
        ).get("context_has_closer_known_columns"),
        "prediction_has_starter_confidence_summary": diagnostic.get(
            "schema_checks",
            {},
        ).get("prediction_has_starter_confidence_summary"),
        "recommendations": diagnostic.get("recommendations", []),
        "report_written_to": "report/context_diagnostic.json",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
