from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
FINALIZED_SNAPSHOT_OUTCOMES_PATH = DATA_DIR / "finalized_snapshot_outcomes.csv"
MARKET_ODDS_PATH = DATA_DIR / "market_odds_history.csv"
SAMPLE_STATE_PATH = DATA_DIR / "sample_state.json"

REPORT_PATH = REPORT_DIR / "underdog_diagnostic_report.json"

CLEAN_PIPELINE_VERSION = "baseline_v2_clean"
MIN_BUCKET_SAMPLE = 30
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
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


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "valid", "ok"}


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
    if len(y_true) == 0:
        return None
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

    result = {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": accuracy,
        "brier": _safe_brier(scored["home_win"], scored["model_home_prob"]),
        "logloss": _safe_logloss(scored["home_win"], scored["model_home_prob"]),
        "ece": _safe_ece(scored["home_win"], scored["model_home_prob"]),
        "insufficient_sample": sample_count < MIN_BUCKET_SAMPLE,
        "decision": "insufficient_sample",
    }

    if sample_count >= MIN_BUCKET_SAMPLE:
        if accuracy is not None and accuracy >= 0.53 and (result["ece"] is not None and result["ece"] <= 0.08):
            result["decision"] = "usable_shadow_slice"
        elif accuracy is not None and accuracy < 0.50:
            result["decision"] = "weak_slice"
        else:
            result["decision"] = "tracking_only"

    return result


def _load_closing_moneyline() -> pd.DataFrame:
    market, error = _read_csv(MARKET_ODDS_PATH)
    if error or market.empty:
        return pd.DataFrame()

    required = {"game_id", "market", "side", "odds", "is_closing_snapshot"}
    if not required.issubset(set(market.columns)):
        return pd.DataFrame()

    result = market.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result["odds"] = pd.to_numeric(result["odds"], errors="coerce")
    result["is_closing_snapshot"] = _bool_series(result["is_closing_snapshot"])

    if "pipeline_version" in result.columns:
        result = result[result["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION].copy()

    result = result[
        (result["market"].astype(str).str.lower() == "moneyline")
        & (result["side"].astype(str).str.lower().isin(["home", "away"]))
        & result["is_closing_snapshot"]
        & (result["odds"] > 1.0)
    ].copy()

    return result


def _append_model_side_clv(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_side_clv"] = np.nan

    closing = _load_closing_moneyline()
    if closing.empty:
        return result

    closing_lookup = (
        closing.groupby(["game_id", "side"], as_index=False)["odds"]
        .median()
        .set_index(["game_id", "side"])["odds"]
        .to_dict()
    )

    clv_values = []

    for _, row in result.iterrows():
        side = row.get("model_pick_side")
        if side not in {"home", "away"}:
            clv_values.append(np.nan)
            continue

        entry_column = "home_moneyline_odds" if side == "home" else "away_moneyline_odds"
        entry_odds = _to_float(row.get(entry_column))
        closing_odds = _to_float(closing_lookup.get((_normalize_game_id(row.get("game_id")), side)))

        if entry_odds is None or closing_odds is None or entry_odds <= 1.0 or closing_odds <= 1.0:
            clv_values.append(np.nan)
            continue

        entry_probability = 1.0 / entry_odds
        closing_probability = 1.0 / closing_odds
        clv_values.append(closing_probability - entry_probability)

    result["model_side_clv"] = clv_values
    return result


def _picked_signed_feature(frame: pd.DataFrame, column: str, *, invert: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index)

    values = pd.to_numeric(frame[column], errors="coerce")
    if invert:
        values = -values

    signed_values = np.where(frame["model_pick_side"] == "home", values, -values)
    return pd.Series(signed_values, index=frame.index)


def build_report() -> dict[str, Any]:
    joined, warnings, errors = _load_joined_samples()
    sample_state, sample_state_error = _read_json(SAMPLE_STATE_PATH)
    if sample_state_error:
        warnings.append(f"sample_state unavailable: {sample_state_error}")

    if joined.empty:
        report = {
            "generated_at": _utc_now(),
            "status": "skipped" if not errors else "partial",
            "sample_count": 0,
            "underdog_sample_count": 0,
            "overall": _bucket_metrics(pd.DataFrame()),
            "buckets": {},
            "recommendation": {
                "underdog_policy": "TRACKING_ONLY",
                "allow_when": [],
                "block_when": ["insufficient settled underdog samples"],
                "notes": warnings,
            },
            "sample_state": sample_state,
            "warnings": warnings,
            "errors": errors,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }
        return report

    frame = joined.copy()
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame["market_no_vig_home_prob"] = pd.to_numeric(frame.get("market_no_vig_home_prob"), errors="coerce")
    frame["model_pick_side"] = np.where(frame["model_home_prob"] >= 0.5, "home", "away")
    frame["market_favorite_side"] = np.where(frame["market_no_vig_home_prob"] >= 0.5, "home", "away")
    frame["model_pick_is_underdog"] = (
        frame["market_no_vig_home_prob"].notna()
        & (frame["model_pick_side"] != frame["market_favorite_side"])
    )
    frame["confidence"] = np.maximum(frame["model_home_prob"], 1.0 - frame["model_home_prob"])

    frame = _append_model_side_clv(frame)

    starter_signal = _picked_signed_feature(frame, "pitcher_rating_diff")
    if starter_signal.isna().all():
        starter_signal = _picked_signed_feature(frame, "sp_fip_diff", invert=True)

    bullpen_signal = _picked_signed_feature(frame, "bullpen_availability_diff")

    frame["picked_starter_edge"] = pd.to_numeric(starter_signal, errors="coerce") > 0
    frame["picked_bullpen_edge"] = pd.to_numeric(bullpen_signal, errors="coerce") > 0

    if "lineup_context_available" in frame.columns:
        frame["lineup_context_bool"] = frame["lineup_context_available"].apply(_truthy)
    else:
        frame["lineup_context_bool"] = False

    underdogs = frame[frame["model_pick_is_underdog"]].copy()

    buckets = {
        "home_underdog": _bucket_metrics(underdogs[underdogs["model_pick_side"] == "home"]),
        "away_underdog": _bucket_metrics(underdogs[underdogs["model_pick_side"] == "away"]),
        "positive_clv_underdog": _bucket_metrics(underdogs[underdogs["model_side_clv"] > 0]),
        "non_positive_clv_underdog": _bucket_metrics(
            underdogs[
                underdogs["model_side_clv"].isna()
                | (underdogs["model_side_clv"] <= 0)
            ]
        ),
        "starter_edge_underdog": _bucket_metrics(underdogs[underdogs["picked_starter_edge"]]),
        "no_starter_edge_underdog": _bucket_metrics(underdogs[~underdogs["picked_starter_edge"]]),
        "bullpen_edge_underdog": _bucket_metrics(underdogs[underdogs["picked_bullpen_edge"]]),
        "lineup_context_available_underdog": _bucket_metrics(underdogs[underdogs["lineup_context_bool"]]),
        "lineup_context_missing_underdog": _bucket_metrics(underdogs[~underdogs["lineup_context_bool"]]),
        "high_confidence_underdog": _bucket_metrics(underdogs[underdogs["confidence"] >= 0.65]),
        "low_confidence_underdog": _bucket_metrics(underdogs[underdogs["confidence"] < 0.65]),
    }

    overall = _bucket_metrics(underdogs)

    allow_when: list[str] = []
    block_when: list[str] = []
    notes: list[str] = []

    policy = "TRACKING_ONLY"
    if overall["sample_count"] < MIN_BUCKET_SAMPLE:
        block_when.append("underdog sample is below minimum diagnostic threshold")
    elif (
        overall["accuracy"] is not None
        and overall["accuracy"] >= 0.53
        and overall["ece"] is not None
        and overall["ece"] <= 0.08
    ):
        policy = "PAPER_ENTRY_ALLOWED"
        allow_when.append("overall underdog slice accuracy >= 53% and ece <= 0.08")
    else:
        policy = "TRACKING_ONLY"
        block_when.append("overall underdog slice is not strong enough for paper entry upgrade")

    for key, value in buckets.items():
        if value.get("decision") == "weak_slice":
            block_when.append(f"{key} is weak")
        elif value.get("decision") == "usable_shadow_slice":
            allow_when.append(f"{key} is usable in shadow research")

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "sample_count": int(len(frame)),
        "underdog_sample_count": int(len(underdogs)),
        "overall": overall,
        "buckets": buckets,
        "recommendation": {
            "underdog_policy": policy,
            "allow_when": sorted(set(allow_when)),
            "block_when": sorted(set(block_when)),
            "notes": notes,
        },
        "sample_state": sample_state,
        "warnings": warnings,
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    return report


def main() -> None:
    report = build_report()
    _write_json(REPORT_PATH, report)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
