from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
FINALIZED_SNAPSHOT_OUTCOMES_PATH = DATA_DIR / "finalized_snapshot_outcomes.csv"
LINEUP_QUALITY_CONTEXT_PATH = DATA_DIR / "lineup_quality_context.csv"

UNDERDOG_REPORT_PATH = REPORT_DIR / "underdog_diagnostic_report.json"
CONFIDENCE_REPORT_PATH = REPORT_DIR / "confidence_bucket_guardrail_report.json"
FEATURE_FRESHNESS_REPORT_PATH = REPORT_DIR / "feature_freshness_report.json"
LINEUP_QUALITY_REPORT_PATH = REPORT_DIR / "lineup_quality_report.json"

REPORT_PATH = REPORT_DIR / "model_correctness_report.json"

CLEAN_PIPELINE_VERSION = "baseline_v2_clean"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value) and not isinstance(value, (str, bool)):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "file_missing"
    try:
        return pd.read_csv(path), ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "file_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "json_not_object"
    return payload, ""


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
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "valid", "ok"})


def _probability_column(frame: pd.DataFrame) -> Optional[str]:
    for column in [
        "displayed_home_win_pct",
        "predicted_home_win_pct",
        "premarket_model_home_prob",
    ]:
        if column in frame.columns:
            return column
    return None


def _prepare_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "pipeline_version" in result.columns:
        preferred = result[result["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION].copy()
        if not preferred.empty:
            result = preferred

    if "snapshot_valid" in result.columns:
        result = result[_bool_series(result["snapshot_valid"])].copy()

    leakage_columns = [
        "home_win",
        "home_score",
        "away_score",
        "final_score",
        "home_final_score",
        "away_final_score",
        "settled_at",
        "actual_winner",
        "actual_result",
        "final_home_score",
        "final_away_score",
        "postgame_win_probability",
    ]
    result = result.drop(columns=[c for c in leakage_columns if c in result.columns], errors="ignore")

    prob_col = _probability_column(result)
    if prob_col is None:
        return pd.DataFrame()

    result["model_home_prob"] = pd.to_numeric(result[prob_col], errors="coerce")
    result = result[result["model_home_prob"].between(0, 1, inclusive="both")].copy()

    if "snapshot_created_at" in result.columns:
        result["_snapshot_sort_time"] = pd.to_datetime(
            result["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
        result = result.sort_values(["game_id", "_snapshot_sort_time"], kind="mergesort")
        result = result.groupby("game_id", as_index=False).tail(1)

    return result.reset_index(drop=True)


def _prepare_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "home_win" not in result.columns:
        if {"home_score", "away_score"}.issubset(set(result.columns)):
            home_score = pd.to_numeric(result["home_score"], errors="coerce")
            away_score = pd.to_numeric(result["away_score"], errors="coerce")
            result["home_win"] = (home_score > away_score).astype("Int64")
        else:
            return pd.DataFrame()

    result["home_win"] = pd.to_numeric(result["home_win"], errors="coerce")
    result = result[result["home_win"].isin([0, 1])].copy()
    result["home_win"] = result["home_win"].astype(int)

    return result[["game_id", "home_win"]].drop_duplicates("game_id", keep="last")


def _combined_outcomes(finalized: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    frames = []
    finalized_prepared = _prepare_outcomes(finalized)
    cache_prepared = _prepare_outcomes(cache)

    if not finalized_prepared.empty:
        frames.append(finalized_prepared)
    if not cache_prepared.empty:
        frames.append(cache_prepared)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["game_id"] = combined["game_id"].apply(_normalize_game_id)
    combined = combined[combined["game_id"] != ""].copy()
    return combined.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _load_joined_samples() -> tuple[pd.DataFrame, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    snapshots_raw, snapshot_error = _read_csv(SNAPSHOT_PATH)
    finalized_raw, finalized_error = _read_csv(FINALIZED_PATH)
    cache_raw, cache_error = _read_csv(FINALIZED_SNAPSHOT_OUTCOMES_PATH)

    if snapshot_error:
        errors.append(f"prediction_snapshots unavailable: {snapshot_error}")
    if finalized_error:
        warnings.append(f"finalized_games unavailable: {finalized_error}")
    if cache_error:
        warnings.append(f"finalized_snapshot_outcomes unavailable: {cache_error}")

    snapshots = _prepare_snapshots(snapshots_raw)
    outcomes = _combined_outcomes(finalized_raw, cache_raw)

    if snapshots.empty:
        warnings.append("No clean prediction snapshots available.")
        return pd.DataFrame(), warnings, errors

    if outcomes.empty:
        warnings.append("No trusted finalized outcomes available.")
        return pd.DataFrame(), warnings, errors

    joined = snapshots.merge(outcomes, on="game_id", how="inner")
    if joined.empty:
        warnings.append("No prediction snapshots join trusted finalized outcomes.")
        return pd.DataFrame(), warnings, errors

    return joined.reset_index(drop=True), warnings, errors


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.55:
        return "50_55"
    if confidence < 0.60:
        return "55_60"
    if confidence < 0.65:
        return "60_65"
    if confidence < 0.70:
        return "65_70"
    return "70_plus"


def _edge_bucket(edge: Any) -> str:
    try:
        value = abs(float(edge))
    except Exception:
        return "edge_unknown"

    if value < 0.01:
        return "edge_0_to_1"
    if value < 0.03:
        return "edge_1_to_3"
    if value < 0.05:
        return "edge_3_to_5"
    return "edge_5_plus"


def _accuracy_slice(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "correct": 0,
            "accuracy": None,
            "insufficient_sample": True,
        }

    sample_count = int(len(frame))
    correct = int(frame["model_pick_correct"].sum())
    return {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": float(correct / sample_count) if sample_count else None,
        "insufficient_sample": sample_count < 20,
    }


def _slice_by_column(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}

    result: dict[str, Any] = {}
    for value, group in frame.groupby(column, dropna=False):
        key = str(value if not pd.isna(value) else "missing")
        result[key] = _accuracy_slice(group)

    return result


def _load_lineup_context() -> pd.DataFrame:
    lineup, error = _read_csv(LINEUP_QUALITY_CONTEXT_PATH)
    if error or lineup.empty or "game_id" not in lineup.columns:
        return pd.DataFrame()
    lineup = lineup.copy()
    lineup["game_id"] = lineup["game_id"].apply(_normalize_game_id)
    return lineup[["game_id", "lineup_confidence_grade"]].drop_duplicates("game_id", keep="last")


def _game_freshness_lookup(freshness_report: dict[str, Any]) -> dict[str, str]:
    rows = freshness_report.get("game_level_freshness")
    if not isinstance(rows, list):
        return {}

    lookup: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        game_id = _normalize_game_id(row.get("game_id"))
        grade = str(row.get("overall_grade") or "D")
        if game_id:
            lookup[game_id] = grade

    return lookup


def build_report() -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    joined, joined_warnings, joined_errors = _load_joined_samples()
    warnings.extend(joined_warnings)
    errors.extend(joined_errors)

    underdog_report, underdog_error = _read_json(UNDERDOG_REPORT_PATH)
    confidence_report, confidence_error = _read_json(CONFIDENCE_REPORT_PATH)
    freshness_report, freshness_error = _read_json(FEATURE_FRESHNESS_REPORT_PATH)
    lineup_report, lineup_error = _read_json(LINEUP_QUALITY_REPORT_PATH)

    for label, error in [
        ("underdog_diagnostic_report", underdog_error),
        ("confidence_bucket_guardrail_report", confidence_error),
        ("feature_freshness_report", freshness_error),
        ("lineup_quality_report", lineup_error),
    ]:
        if error:
            warnings.append(f"{label} unavailable: {error}")

    if joined.empty:
        return {
            "generated_at": _utc_now(),
            "status": "skipped" if not errors else "partial",
            "sample_count": 0,
            "overall_accuracy": None,
            "slices": {},
            "recommended_filters": [],
            "blocked_filters": [],
            "warnings": warnings,
            "errors": errors,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }

    frame = joined.copy()
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame["market_no_vig_home_prob"] = pd.to_numeric(frame.get("market_no_vig_home_prob"), errors="coerce")
    frame["model_edge_home"] = pd.to_numeric(frame.get("model_edge_home"), errors="coerce")

    frame["model_pick_side"] = np.where(frame["model_home_prob"] >= 0.5, "home", "away")
    frame["market_favorite_side"] = np.where(frame["market_no_vig_home_prob"] >= 0.5, "home", "away")
    frame["market_role"] = np.where(
        frame["market_no_vig_home_prob"].notna() & (frame["model_pick_side"] != frame["market_favorite_side"]),
        "underdog_pick",
        "favorite_pick",
    )

    frame["model_pick_correct"] = (
        ((frame["model_pick_side"] == "home") & (frame["home_win"] == 1))
        | ((frame["model_pick_side"] == "away") & (frame["home_win"] == 0))
    )

    frame["confidence"] = np.maximum(frame["model_home_prob"], 1.0 - frame["model_home_prob"])
    frame["confidence_bucket"] = frame["confidence"].apply(_confidence_bucket)
    frame["edge_bucket"] = frame["model_edge_home"].apply(_edge_bucket)

    if "odds_quality_status" not in frame.columns:
        frame["odds_quality_status"] = "UNKNOWN"

    if "recommendation_status" not in frame.columns:
        frame["recommendation_status"] = "UNKNOWN"

    lineup_context = _load_lineup_context()
    if not lineup_context.empty:
        frame = frame.merge(lineup_context, on="game_id", how="left")
    else:
        frame["lineup_confidence_grade"] = "missing"

    frame["lineup_confidence_grade"] = frame["lineup_confidence_grade"].fillna("missing").astype(str)

    freshness_lookup = _game_freshness_lookup(freshness_report)
    frame["freshness_grade"] = frame["game_id"].apply(lambda value: freshness_lookup.get(_normalize_game_id(value), "missing"))

    overall = _accuracy_slice(frame)

    slices = {
        "model_pick_side": _slice_by_column(frame, "model_pick_side"),
        "market_role": _slice_by_column(frame, "market_role"),
        "confidence_bucket": _slice_by_column(frame, "confidence_bucket"),
        "edge_bucket": _slice_by_column(frame, "edge_bucket"),
        "lineup_grade": _slice_by_column(frame, "lineup_confidence_grade"),
        "freshness_grade": _slice_by_column(frame, "freshness_grade"),
        "odds_quality_status": _slice_by_column(frame, "odds_quality_status"),
        "recommendation_status": _slice_by_column(frame, "recommendation_status"),
    }

    recommended_filters: list[str] = []
    blocked_filters: list[str] = []

    for group_name, group_slices in slices.items():
        for slice_name, metrics in group_slices.items():
            sample_count = int(metrics.get("sample_count") or 0)
            accuracy = metrics.get("accuracy")
            if sample_count < 20 or accuracy is None:
                continue

            label = f"{group_name}:{slice_name}"
            if accuracy >= 0.55:
                recommended_filters.append(label)
            elif accuracy < 0.50:
                blocked_filters.append(label)

    unsafe_confidence = confidence_report.get("unsafe_buckets")
    if isinstance(unsafe_confidence, list):
        for bucket in unsafe_confidence:
            blocked_filters.append(f"confidence_bucket:{bucket}")

    underdog_recommendation = underdog_report.get("recommendation")
    if isinstance(underdog_recommendation, dict):
        if str(underdog_recommendation.get("underdog_policy")) != "PAPER_ENTRY_ALLOWED":
            blocked_filters.append("market_role:underdog_pick")

    return {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "sample_count": int(len(frame)),
        "overall_accuracy": overall.get("accuracy"),
        "overall": overall,
        "slices": slices,
        "recommended_filters": sorted(set(recommended_filters)),
        "blocked_filters": sorted(set(blocked_filters)),
        "warnings": sorted(set(warnings)),
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def main() -> None:
    report = build_report()
    _write_json(REPORT_PATH, report)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
