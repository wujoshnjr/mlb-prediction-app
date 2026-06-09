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

REPORT_PATH = REPORT_DIR / "confidence_bucket_guardrail_report.json"

CLEAN_PIPELINE_VERSION = "baseline_v2_clean"
EPSILON = 1e-15


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
    prepared_finalized = _prepare_outcomes(finalized)
    prepared_cache = _prepare_outcomes(cache)

    if not prepared_finalized.empty:
        frames.append(prepared_finalized)
    if not prepared_cache.empty:
        frames.append(prepared_cache)

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


def _safe_logloss(y_true: pd.Series, probabilities: pd.Series) -> Optional[float]:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(probabilities, errors="coerce").clip(EPSILON, 1 - EPSILON)
    valid = y.isin([0, 1]) & p.notna()
    if not valid.any():
        return None
    y = y[valid].astype(float)
    p = p[valid].astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _safe_brier(y_true: pd.Series, probabilities: pd.Series) -> Optional[float]:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(probabilities, errors="coerce")
    valid = y.isin([0, 1]) & p.between(0, 1, inclusive="both")
    if not valid.any():
        return None
    return float(((p[valid].astype(float) - y[valid].astype(float)) ** 2).mean())


def _safe_ece(y_true: pd.Series, probabilities: pd.Series, bins: int = 10) -> Optional[float]:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(probabilities, errors="coerce")
    valid = y.isin([0, 1]) & p.between(0, 1, inclusive="both")
    if not valid.any():
        return None

    y = y[valid].astype(float).to_numpy()
    p = p[valid].astype(float).to_numpy()

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)

        if not mask.any():
            continue

        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - observed)

    return float(ece)


def _bucket_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "correct": 0,
            "accuracy": None,
            "brier": None,
            "logloss": None,
            "ece": None,
            "unsafe": False,
            "insufficient_sample": True,
            "reasons": ["empty bucket"],
        }

    scored = frame.copy()
    scored["model_pick_side"] = np.where(scored["model_home_prob"] >= 0.5, "home", "away")
    scored["model_pick_correct"] = (
        ((scored["model_pick_side"] == "home") & (scored["home_win"] == 1))
        | ((scored["model_pick_side"] == "away") & (scored["home_win"] == 0))
    )

    sample_count = int(len(scored))
    correct = int(scored["model_pick_correct"].sum())
    accuracy = float(correct / sample_count) if sample_count else None
    brier = _safe_brier(scored["home_win"], scored["model_home_prob"])
    logloss = _safe_logloss(scored["home_win"], scored["model_home_prob"])
    ece = _safe_ece(scored["home_win"], scored["model_home_prob"])

    reasons: list[str] = []
    unsafe = False
    insufficient = sample_count < 20

    if insufficient:
        reasons.append("sample_count below 20")
    else:
        if accuracy is not None and accuracy < 0.50:
            unsafe = True
            reasons.append("accuracy below 50%")
        if ece is not None and ece > 0.15:
            unsafe = True
            reasons.append("ece above 0.15")

    return {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "brier": brier,
        "logloss": logloss,
        "ece": ece,
        "unsafe": unsafe,
        "insufficient_sample": insufficient,
        "reasons": reasons,
    }


def _slice_bucket(frame: pd.DataFrame, lower: float, upper: Optional[float]) -> pd.DataFrame:
    if upper is None:
        return frame[frame["confidence"] >= lower].copy()
    return frame[(frame["confidence"] >= lower) & (frame["confidence"] < upper)].copy()


def build_report() -> dict[str, Any]:
    joined, warnings, errors = _load_joined_samples()

    if joined.empty:
        return {
            "generated_at": _utc_now(),
            "status": "skipped" if not errors else "partial",
            "sample_count": 0,
            "buckets": {
                "50_55": _bucket_metrics(pd.DataFrame()),
                "55_60": _bucket_metrics(pd.DataFrame()),
                "60_65": _bucket_metrics(pd.DataFrame()),
                "65_70": _bucket_metrics(pd.DataFrame()),
                "70_plus": _bucket_metrics(pd.DataFrame()),
            },
            "global_policy": {
                "recommended_max_display_confidence": 0.65,
                "block_high_confidence_language": True,
                "default_shrink_alpha": 0.7,
                "reason": ["no joined settled samples"],
            },
            "unsafe_buckets": [],
            "warnings": warnings,
            "errors": errors,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }

    frame = joined.copy()
    frame["confidence"] = np.maximum(frame["model_home_prob"], 1.0 - frame["model_home_prob"])

    buckets = {
        "50_55": _bucket_metrics(_slice_bucket(frame, 0.50, 0.55)),
        "55_60": _bucket_metrics(_slice_bucket(frame, 0.55, 0.60)),
        "60_65": _bucket_metrics(_slice_bucket(frame, 0.60, 0.65)),
        "65_70": _bucket_metrics(_slice_bucket(frame, 0.65, 0.70)),
        "70_plus": _bucket_metrics(_slice_bucket(frame, 0.70, None)),
    }

    unsafe_buckets = [
        key for key, value in buckets.items()
        if value.get("unsafe")
    ]

    high_bucket_unsafe = any(
        key in unsafe_buckets
        for key in {"65_70", "70_plus"}
    )

    reason: list[str] = []
    if high_bucket_unsafe:
        reason.append("one or more high-confidence buckets are unsafe")
    if not unsafe_buckets:
        reason.append("no unsafe confidence bucket with sufficient sample")

    global_policy = {
        "recommended_max_display_confidence": 0.65 if high_bucket_unsafe else 0.75,
        "block_high_confidence_language": bool(high_bucket_unsafe),
        "default_shrink_alpha": 0.7 if high_bucket_unsafe else 0.85,
        "reason": reason,
    }

    return {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "sample_count": int(len(frame)),
        "buckets": buckets,
        "global_policy": global_policy,
        "unsafe_buckets": unsafe_buckets,
        "warnings": warnings,
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
