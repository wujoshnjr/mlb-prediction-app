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

REPORT_PATH = REPORT_DIR / "edge_sanity_guardrail_report.json"

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
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "valid", "ok"}
    )


def _probability_column(frame: pd.DataFrame) -> Optional[str]:
    for column in (
        "displayed_home_win_pct",
        "predicted_home_win_pct",
        "premarket_model_home_prob",
    ):
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
    if not prepared_finalized.empty:
        frames.append(prepared_finalized)

    prepared_cache = _prepare_outcomes(cache)
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

    y_arr = y[valid].astype(float).to_numpy()
    p_arr = p[valid].astype(float).to_numpy()

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:
            mask = (p_arr >= lower) & (p_arr <= upper)
        else:
            mask = (p_arr >= lower) & (p_arr < upper)

        if not mask.any():
            continue

        confidence = float(p_arr[mask].mean())
        observed = float(y_arr[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - observed)

    return float(ece)


def _bucket_name(edge_abs: float) -> str:
    if edge_abs < 0.01:
        return "edge_0_to_1"
    if edge_abs < 0.03:
        return "edge_1_to_3"
    if edge_abs < 0.05:
        return "edge_3_to_5"
    return "edge_5_plus"


def _bucket_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "correct": 0,
            "accuracy": None,
            "brier": None,
            "logloss": None,
            "ece": None,
            "insufficient_sample": True,
            "decision": "insufficient_sample",
        }

    scored = frame.copy()
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

    decision = "insufficient_sample"
    if sample_count >= 20:
        if accuracy is not None and accuracy >= 0.55 and ece is not None and ece <= 0.10:
            decision = "preferred_shadow_slice"
        elif accuracy is not None and accuracy < 0.50:
            decision = "weak_slice"
        else:
            decision = "tracking_only"

    return {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "brier": brier,
        "logloss": logloss,
        "ece": ece,
        "insufficient_sample": sample_count < 20,
        "decision": decision,
    }


def build_report() -> dict[str, Any]:
    joined, warnings, errors = _load_joined_samples()

    if joined.empty:
        return {
            "generated_at": _utc_now(),
            "status": "skipped" if not errors else "partial",
            "sample_count": 0,
            "buckets": {},
            "policy": {
                "preferred_edge_bucket": None,
                "large_edge_policy": "TRACKING_ONLY",
                "block_large_edge": True,
                "large_edge_reasons": ["No joined settled samples."],
                "notes": ["No joined settled samples."],
            },
            "warnings": warnings,
            "errors": errors,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }

    frame = joined.copy()
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame["market_no_vig_home_prob"] = pd.to_numeric(
        frame.get("market_no_vig_home_prob"),
        errors="coerce",
    )

    if "model_edge_home" in frame.columns:
        frame["edge_home"] = pd.to_numeric(frame["model_edge_home"], errors="coerce")
    else:
        frame["edge_home"] = frame["model_home_prob"] - frame["market_no_vig_home_prob"]

    frame["model_pick_side"] = np.where(frame["model_home_prob"] >= 0.5, "home", "away")
    frame["selected_edge"] = np.where(
        frame["model_pick_side"] == "home",
        frame["edge_home"],
        -frame["edge_home"],
    )
    frame["selected_edge"] = pd.to_numeric(frame["selected_edge"], errors="coerce")
    frame = frame[frame["selected_edge"].notna()].copy()

    frame["selected_edge_abs"] = frame["selected_edge"].abs()
    frame["edge_bucket"] = frame["selected_edge_abs"].apply(_bucket_name)

    buckets = {
        name: _bucket_metrics(frame[frame["edge_bucket"] == name])
        for name in ["edge_0_to_1", "edge_1_to_3", "edge_3_to_5", "edge_5_plus"]
    }

    preferred_edge_bucket = None
    for name in ["edge_3_to_5", "edge_1_to_3", "edge_0_to_1"]:
        metrics = buckets.get(name, {})
        if (
            (metrics.get("sample_count") or 0) >= 20
            and metrics.get("accuracy") is not None
            and metrics["accuracy"] >= 0.55
        ):
            preferred_edge_bucket = name
            break

    edge_5 = buckets.get("edge_5_plus", {})
    edge_3_to_5 = buckets.get("edge_3_to_5", {})

    block_large_edge = False
    large_edge_reasons: list[str] = []

    if (edge_5.get("sample_count") or 0) >= 30:
        edge_5_accuracy = edge_5.get("accuracy")
        edge_3_to_5_accuracy = edge_3_to_5.get("accuracy")

        if edge_5_accuracy is not None and edge_5_accuracy < 0.53:
            block_large_edge = True
            large_edge_reasons.append("edge_5_plus accuracy below 53%")

        if (
            edge_5_accuracy is not None
            and edge_3_to_5_accuracy is not None
            and edge_5_accuracy + 0.03 < edge_3_to_5_accuracy
        ):
            block_large_edge = True
            large_edge_reasons.append(
                "edge_5_plus underperforms edge_3_to_5 by more than 3 percentage points"
            )

    large_edge_policy = "TRACKING_ONLY" if block_large_edge else "MODEL_SIGNAL_ONLY"

    return {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "sample_count": int(len(frame)),
        "buckets": buckets,
        "policy": {
            "preferred_edge_bucket": preferred_edge_bucket,
            "large_edge_policy": large_edge_policy,
            "block_large_edge": block_large_edge,
            "large_edge_reasons": large_edge_reasons,
            "notes": [
                "Large model edge is not automatically better.",
                "Use edge_3_to_5 as preferred shadow slice only while evidence remains stable.",
            ],
        },
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
