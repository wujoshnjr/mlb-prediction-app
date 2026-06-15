from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

SNAPSHOTS_CSV = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_SNAPSHOT_OUTCOMES_CSV = DATA_DIR / "finalized_snapshot_outcomes.csv"
LEGACY_FINALIZED_GAMES_CSV = DATA_DIR / "finalized_games.csv"

OUTPUT_JSON = REPORT_DIR / "walkforward_evaluation.json"
OUTPUT_CSV = REPORT_DIR / "walkforward_predictions.csv"

MIN_REQUIRED_OOS_PREDICTIONS = 300
MIN_FOLD_SIZE = 20
TARGET_FOLD_COUNT = 5
EPSILON = 1e-12

CSV_COLUMNS = [
    "fold_id",
    "game_id",
    "game_date",
    "prediction_created_at",
    "model_prob",
    "market_prob",
    "constant_50_prob",
    "outcome",
    "clv",
    "status",
]

MODEL_PROBABILITY_CANDIDATES = [
    "predicted_home_win_pct",
    "premarket_model_home_prob",
    "displayed_home_win_pct",
    "model_prob",
    "home_win_probability",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "rows": 0, "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return None, status
    status["rows"] = int(len(frame))
    return frame, status


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _clip_probability(value: Any) -> Optional[float]:
    parsed = _as_float(value)
    if parsed is None:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed = parsed / 100.0
    if parsed < 0.0 or parsed > 1.0:
        return None
    return float(max(EPSILON, min(1.0 - EPSILON, parsed)))


def _brier(probabilities: List[float], outcomes: List[int]) -> Optional[float]:
    if not probabilities:
        return None
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def _logloss(probabilities: List[float], outcomes: List[int]) -> Optional[float]:
    if not probabilities:
        return None
    p = np.asarray([_clip_probability(prob) for prob in probabilities], dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _accuracy(probabilities: List[float], outcomes: List[int]) -> Optional[float]:
    if not probabilities:
        return None
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    return float(np.mean((p >= 0.5).astype(int) == y))


def _balanced_accuracy(probabilities: List[float], outcomes: List[int]) -> Optional[float]:
    if not probabilities:
        return None
    p = (np.asarray(probabilities, dtype=float) >= 0.5).astype(int)
    y = np.asarray(outcomes, dtype=int)
    recalls: List[float] = []
    for label in (0, 1):
        mask = y == label
        if not bool(np.any(mask)):
            continue
        recalls.append(float(np.mean(p[mask] == y[mask])))
    if not recalls:
        return None
    return float(np.mean(recalls))


def _safe_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _normalize_game_id_value(value: Any) -> str:
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


def _normalise_game_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].apply(_normalize_game_id_value)
    return frame


def _attach_model_probability(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    diagnostics: dict[str, Any] = {"candidate_columns": [], "used_columns": [], "valid_model_probability_count": 0}
    frame["model_prob"] = np.nan
    for column in MODEL_PROBABILITY_CANDIDATES:
        if column not in frame.columns:
            continue
        candidate = frame[column].apply(_clip_probability)
        valid_count = int(candidate.notna().sum())
        diagnostics["candidate_columns"].append({"column": column, "valid_count": valid_count})
        before = int(frame["model_prob"].notna().sum())
        frame["model_prob"] = frame["model_prob"].where(frame["model_prob"].notna(), candidate)
        after = int(frame["model_prob"].notna().sum())
        if after > before:
            diagnostics["used_columns"].append(column)
    diagnostics["valid_model_probability_count"] = int(frame["model_prob"].notna().sum())
    return frame, diagnostics


def _estimate_available_oos_predictions(snapshots: Optional[pd.DataFrame]) -> int:
    if snapshots is None or snapshots.empty or "game_id" not in snapshots.columns:
        return 0
    frame = _normalise_game_id(snapshots)
    frame = frame[frame["game_id"] != ""].copy()
    if frame.empty:
        return 0
    frame, _ = _attach_model_probability(frame)
    frame = frame[frame["model_prob"].notna()].copy()
    if frame.empty:
        return 0
    return int(frame["game_id"].astype(str).nunique())


def _prepare_outcome_frame(finalized: Optional[pd.DataFrame]) -> pd.DataFrame:
    if finalized is None or finalized.empty or "game_id" not in finalized.columns:
        return pd.DataFrame(columns=["game_id", "home_win"])
    final = _normalise_game_id(finalized)
    final = final[final["game_id"] != ""].copy()
    if final.empty:
        return pd.DataFrame(columns=["game_id", "home_win"])
    if "home_win" not in final.columns and {"home_score", "away_score"}.issubset(final.columns):
        final["home_win"] = (
            pd.to_numeric(final["home_score"], errors="coerce")
            > pd.to_numeric(final["away_score"], errors="coerce")
        ).astype("Int64")
    if "home_win" not in final.columns:
        return pd.DataFrame(columns=["game_id", "home_win"])
    final["home_win"] = pd.to_numeric(final["home_win"], errors="coerce")
    final = final[final["home_win"].isin([0, 1])].copy()
    if final.empty:
        return pd.DataFrame(columns=["game_id", "home_win"])
    final = final.drop_duplicates("game_id", keep="last").copy()
    final["home_win"] = final["home_win"].astype(int)
    return final[["game_id", "home_win"]].copy()


def _prepare_settled_snapshots(
    snapshots: Optional[pd.DataFrame],
    finalized: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "snapshot_rows": 0,
        "snapshot_unique_games": 0,
        "outcome_rows": 0,
        "outcome_unique_games": 0,
        "linked_rows_before_probability_filter": 0,
        "linked_unique_games_before_probability_filter": 0,
        "settled_rows_after_probability_filter": 0,
        "missing_outcome_rows": 0,
        "probability_diagnostics": {},
    }
    if snapshots is None or snapshots.empty or "game_id" not in snapshots.columns:
        return pd.DataFrame(), diagnostics

    frame = _normalise_game_id(snapshots)
    frame = frame[frame["game_id"] != ""].copy()
    diagnostics["snapshot_rows"] = int(len(frame))
    diagnostics["snapshot_unique_games"] = int(frame["game_id"].nunique()) if not frame.empty else 0

    if "snapshot_created_at" in frame.columns:
        frame["_snapshot_dt"] = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    else:
        frame["_snapshot_dt"] = pd.NaT

    outcomes = _prepare_outcome_frame(finalized)
    diagnostics["outcome_rows"] = int(len(outcomes))
    diagnostics["outcome_unique_games"] = int(outcomes["game_id"].nunique()) if not outcomes.empty else 0
    if outcomes.empty:
        return pd.DataFrame(), diagnostics

    frame = frame.merge(outcomes, on="game_id", how="left")
    diagnostics["missing_outcome_rows"] = int(frame["home_win"].isna().sum())
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    diagnostics["linked_rows_before_probability_filter"] = int(len(frame))
    diagnostics["linked_unique_games_before_probability_filter"] = int(frame["game_id"].nunique()) if not frame.empty else 0
    if frame.empty:
        return frame, diagnostics

    frame = frame.sort_values("_snapshot_dt").groupby("game_id", as_index=False).tail(1).copy()
    frame, probability_diagnostics = _attach_model_probability(frame)
    diagnostics["probability_diagnostics"] = probability_diagnostics

    if "market_no_vig_home_prob" in frame.columns:
        frame["market_prob"] = frame["market_no_vig_home_prob"].apply(_clip_probability)
    else:
        frame["market_prob"] = np.nan

    frame["constant_50_prob"] = 0.5
    frame["outcome"] = frame["home_win"].astype(int)
    if "clv_home_moneyline" in frame.columns:
        frame["clv"] = pd.to_numeric(frame["clv_home_moneyline"], errors="coerce")
    else:
        frame["clv"] = np.nan

    frame = frame[frame["model_prob"].notna()].copy()
    frame = frame.sort_values(["_snapshot_dt", "game_id"]).reset_index(drop=True)
    diagnostics["settled_rows_after_probability_filter"] = int(len(frame))
    return frame, diagnostics


def _folds(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    if len(frame) < MIN_FOLD_SIZE:
        return [frame.copy()]
    fold_count = max(1, min(TARGET_FOLD_COUNT, len(frame) // MIN_FOLD_SIZE))
    return [fold.copy() for fold in np.array_split(frame, fold_count) if not fold.empty]


def _metric_pack(probabilities: List[float], outcomes: List[int]) -> Dict[str, Any]:
    predicted_positive_rate = float(np.mean(np.asarray(probabilities) >= 0.5)) if probabilities else None
    predicted_negative_rate = None if predicted_positive_rate is None else 1.0 - predicted_positive_rate
    return {
        "count": len(probabilities),
        "brier": _safe_round(_brier(probabilities, outcomes)),
        "logloss": _safe_round(_logloss(probabilities, outcomes)),
        "accuracy": _safe_round(_accuracy(probabilities, outcomes)),
        "balanced_accuracy": _safe_round(_balanced_accuracy(probabilities, outcomes)),
        "probability_mean": _safe_round(float(np.mean(probabilities)) if probabilities else None),
        "probability_std": _safe_round(float(np.std(probabilities)) if probabilities else None),
        "predicted_positive_rate": _safe_round(predicted_positive_rate),
        "predicted_negative_rate": _safe_round(predicted_negative_rate),
    }


def _fold_summary(fold_id: int, frame: pd.DataFrame) -> Dict[str, Any]:
    outcomes = [int(value) for value in frame["outcome"].tolist()]
    model_probs = [float(value) for value in frame["model_prob"].tolist()]
    market_frame = frame[frame["market_prob"].notna()]
    market_probs = [float(value) for value in market_frame["market_prob"].tolist()]
    market_outcomes = [int(value) for value in market_frame["outcome"].tolist()]
    constant_probs = [0.5 for _ in outcomes]

    model_metrics = _metric_pack(model_probs, outcomes)
    market_metrics = _metric_pack(market_probs, market_outcomes) if market_probs else None
    constant_metrics = _metric_pack(constant_probs, outcomes)

    reasons: List[str] = []
    if model_metrics["balanced_accuracy"] is not None and abs(float(model_metrics["balanced_accuracy"]) - 0.5) <= 0.03:
        reasons.append("balanced_accuracy_near_random")
    if model_metrics["predicted_positive_rate"] is not None and float(model_metrics["predicted_positive_rate"]) >= 0.85:
        reasons.append("single_class_positive_prediction_collapse")
    if model_metrics["predicted_negative_rate"] is not None and float(model_metrics["predicted_negative_rate"]) >= 0.85:
        reasons.append("single_class_negative_prediction_collapse")
    if model_metrics["probability_std"] is not None and float(model_metrics["probability_std"]) <= 0.015:
        reasons.append("probability_distribution_too_narrow")
    if model_metrics["brier"] is not None and constant_metrics["brier"] is not None and float(model_metrics["brier"]) >= float(constant_metrics["brier"]):
        reasons.append("model_does_not_beat_constant_50_brier")
    if market_metrics and model_metrics["brier"] is not None and market_metrics["brier"] is not None and float(model_metrics["brier"]) >= float(market_metrics["brier"]):
        reasons.append("model_does_not_beat_market_brier")

    collapse_detected = len(reasons) > 0
    return {
        "fold_id": fold_id,
        "start_snapshot_at": str(frame["_snapshot_dt"].min()) if "_snapshot_dt" in frame else None,
        "end_snapshot_at": str(frame["_snapshot_dt"].max()) if "_snapshot_dt" in frame else None,
        "sample_count": int(len(frame)),
        "model": model_metrics,
        "market_no_vig": market_metrics,
        "constant_50": constant_metrics,
        "collapse_detected": collapse_detected,
        "collapse_reasons": sorted(set(reasons)),
    }


def _write_predictions_csv(rows: pd.DataFrame, fold_ids: Dict[str, int]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for _, row in rows.iterrows():
            game_id = str(row.get("game_id"))
            writer.writerow(
                [
                    fold_ids.get(game_id, 0),
                    game_id,
                    row.get("game_date"),
                    row.get("snapshot_created_at"),
                    row.get("model_prob"),
                    row.get("market_prob"),
                    row.get("constant_50_prob"),
                    row.get("outcome"),
                    row.get("clv"),
                    "settled_oos",
                ]
            )


def _write_empty_predictions_csv() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)


def _choose_outcome_source(
    trusted_outcomes: Optional[pd.DataFrame],
    trusted_status: Dict[str, Any],
    legacy_finalized: Optional[pd.DataFrame],
    legacy_status: Dict[str, Any],
) -> tuple[Optional[pd.DataFrame], str]:
    if trusted_outcomes is not None and not trusted_outcomes.empty and not trusted_status.get("error"):
        return trusted_outcomes, "finalized_snapshot_outcomes"
    return legacy_finalized, "finalized_games"


def build_walkforward_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()
    snapshots, snapshots_status = _safe_read_csv(SNAPSHOTS_CSV)
    trusted_outcomes, trusted_outcomes_status = _safe_read_csv(FINALIZED_SNAPSHOT_OUTCOMES_CSV)
    legacy_finalized, legacy_finalized_status = _safe_read_csv(LEGACY_FINALIZED_GAMES_CSV)
    outcome_frame, outcome_source = _choose_outcome_source(
        trusted_outcomes,
        trusted_outcomes_status,
        legacy_finalized,
        legacy_finalized_status,
    )
    input_files = {
        "prediction_snapshots": snapshots_status,
        "finalized_snapshot_outcomes": trusted_outcomes_status,
        "legacy_finalized_games": legacy_finalized_status,
    }
    total_oos = _estimate_available_oos_predictions(snapshots)
    settled, linkage_diagnostics = _prepare_settled_snapshots(snapshots, outcome_frame)

    if settled.empty:
        report = {
            "generated_at": generated_at,
            "report_type": "walkforward_evaluation_report",
            "status": "insufficient_samples",
            "input_files": input_files,
            "outcome_source": outcome_source,
            "linkage_diagnostics": linkage_diagnostics,
            "min_required_oos_predictions": MIN_REQUIRED_OOS_PREDICTIONS,
            "total_oos_predictions": int(total_oos),
            "settled_oos_predictions": 0,
            "walkforward_ready": False,
            "fold_count": 0,
            "folds": [],
            "collapse_fold_count": 0,
            "promotion_allowed": False,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
            "recommendations": ["No settled prediction snapshots with valid model probabilities are available."],
        }
        OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        _write_empty_predictions_csv()
        return report

    fold_frames = _folds(settled)
    summaries: list[dict[str, Any]] = []
    fold_ids: Dict[str, int] = {}
    for index, fold in enumerate(fold_frames, start=1):
        for game_id in fold["game_id"].astype(str).tolist():
            fold_ids[game_id] = index
        summaries.append(_fold_summary(index, fold))

    collapse_fold_count = int(sum(1 for fold in summaries if fold.get("collapse_detected")))
    walkforward_ready = int(len(settled)) >= MIN_REQUIRED_OOS_PREDICTIONS
    status = "ok" if walkforward_ready and collapse_fold_count == 0 else "warning"
    if not walkforward_ready:
        status = "insufficient_samples"

    model_briers = [fold["model"].get("brier") for fold in summaries if isinstance(fold.get("model"), dict) and fold["model"].get("brier") is not None]
    model_loglosses = [fold["model"].get("logloss") for fold in summaries if isinstance(fold.get("model"), dict) and fold["model"].get("logloss") is not None]

    recommendations: List[str] = []
    if not walkforward_ready:
        recommendations.append("Walk-forward sample count is below 300; do not use these metrics for promotion.")
    if collapse_fold_count:
        recommendations.append("One or more walk-forward folds show prediction collapse; keep model in tracking-only mode.")
    recommendations.append("Use this report as a time-ordered diagnostic only; no live betting or model promotion is allowed from this report alone.")

    report = {
        "generated_at": generated_at,
        "report_type": "walkforward_evaluation_report",
        "status": status,
        "input_files": input_files,
        "outcome_source": outcome_source,
        "linkage_diagnostics": linkage_diagnostics,
        "min_required_oos_predictions": MIN_REQUIRED_OOS_PREDICTIONS,
        "min_fold_size": MIN_FOLD_SIZE,
        "target_fold_count": TARGET_FOLD_COUNT,
        "total_oos_predictions": int(total_oos),
        "settled_oos_predictions": int(len(settled)),
        "walkforward_ready": bool(walkforward_ready),
        "fold_count": len(summaries),
        "collapse_fold_count": collapse_fold_count,
        "collapse_rate": _safe_round(collapse_fold_count / len(summaries) if summaries else None),
        "median_model_brier": _safe_round(float(np.median(model_briers)) if model_briers else None),
        "median_model_logloss": _safe_round(float(np.median(model_loglosses)) if model_loglosses else None),
        "folds": summaries,
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
        "recommendations": recommendations,
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_predictions_csv(settled, fold_ids)
    return report


def main() -> None:
    report = build_walkforward_report()
    print(json.dumps({"status": report["status"], "output_path": str(OUTPUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
