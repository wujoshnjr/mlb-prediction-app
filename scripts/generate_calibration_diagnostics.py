from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from scripts.model_training_common import (
    PIPELINE_VERSION,
    build_training_frame,
    expected_calibration_error,
    find_market_probability_column,
    safe_float,
    write_json,
)

REPORT_PATH = Path("report/calibration_diagnostics_report.json")
MIN_CALIBRATION_SAMPLES = 300


PROBABILITY_COLUMNS = [
    "displayed_home_win_pct",
    "model_home_win_prob",
    "home_win_probability",
    "predicted_home_win_prob",
    "market_no_vig_home_prob",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_probability_column(frame: pd.DataFrame) -> Optional[str]:
    for column in PROBABILITY_COLUMNS:
        if column in frame.columns:
            return column

    market_column = find_market_probability_column(frame)
    return market_column


def _bucket_label(prob: float) -> str:
    if prob < 0.45:
        return "away_lean_below_45"
    if prob <= 0.55:
        return "low_confidence_45_55"
    if prob <= 0.65:
        return "medium_confidence_55_65"
    return "high_confidence_65_plus"


def _reliability_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    output: List[Dict[str, Any]] = []

    for index in range(n_bins):
        lower = bins[index]
        upper = bins[index + 1]
        if index == n_bins - 1:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob >= lower) & (prob < upper)

        count = int(mask.sum())
        if count == 0:
            output.append(
                {
                    "bin": index + 1,
                    "lower": float(lower),
                    "upper": float(upper),
                    "bin_count": 0,
                    "avg_predicted_prob": None,
                    "actual_win_rate": None,
                    "absolute_error": None,
                }
            )
            continue

        avg_prob = float(np.mean(prob[mask]))
        actual_rate = float(np.mean(y_true[mask]))

        output.append(
            {
                "bin": index + 1,
                "lower": float(lower),
                "upper": float(upper),
                "bin_count": count,
                "avg_predicted_prob": avg_prob,
                "actual_win_rate": actual_rate,
                "absolute_error": abs(avg_prob - actual_rate),
            }
        )

    return output


def _max_calibration_error(bins: List[Dict[str, Any]]) -> Optional[float]:
    errors = [
        safe_float(item.get("absolute_error"))
        for item in bins
        if safe_float(item.get("absolute_error")) is not None and int(item.get("bin_count", 0)) > 0
    ]
    return max(errors) if errors else None


def _slice_metrics(frame: pd.DataFrame, mask: pd.Series, prob_column: str) -> Dict[str, Any]:
    subset = frame[mask].copy()

    if subset.empty:
        return {
            "sample_count": 0,
            "ece": None,
            "mce": None,
            "brier": None,
            "logloss": None,
            "reliability_bins": [],
        }

    y = pd.to_numeric(subset["home_win"], errors="coerce")
    prob = pd.to_numeric(subset[prob_column], errors="coerce")
    valid = y.isin([0, 1]) & prob.notna()

    y_arr = y[valid].astype(int).to_numpy()
    p_arr = np.clip(prob[valid].astype(float).to_numpy(), 0.01, 0.99)

    if len(y_arr) == 0:
        return {
            "sample_count": 0,
            "ece": None,
            "mce": None,
            "brier": None,
            "logloss": None,
            "reliability_bins": [],
        }

    bins = _reliability_bins(y_arr, p_arr)

    return {
        "sample_count": int(len(y_arr)),
        "ece": expected_calibration_error(y_arr, p_arr),
        "mce": _max_calibration_error(bins),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "logloss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        "reliability_bins": bins,
    }


def _range_recommendations(bins: List[Dict[str, Any]]) -> tuple[list[str], list[str]]:
    overconfident: List[str] = []
    underconfident: List[str] = []

    for item in bins:
        if not item.get("bin_count"):
            continue

        avg_prob = safe_float(item.get("avg_predicted_prob"))
        actual = safe_float(item.get("actual_win_rate"))

        if avg_prob is None or actual is None:
            continue

        label = f"{item['lower']:.1f}-{item['upper']:.1f}"

        if avg_prob - actual > 0.08:
            overconfident.append(label)
        elif actual - avg_prob > 0.08:
            underconfident.append(label)

    return overconfident, underconfident


def build_report(
    *,
    snapshot_path: Path = Path("data/prediction_snapshots.csv"),
    finalized_path: Path = Path("data/finalized_games.csv"),
    report_path: Path = REPORT_PATH,
) -> Dict[str, Any]:
    warnings: List[str] = []
    blockers: List[str] = []

    training = build_training_frame(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        pipeline_version=PIPELINE_VERSION,
    )
    warnings.extend(training.get("warnings") or [])

    if not training.get("ok"):
        blockers.append(training.get("skip_reason") or "training frame unavailable")
        report = {
            "generated_at": _utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "status": "partial",
            "calibration_ready": False,
            "minimum_required_samples": MIN_CALIBRATION_SAMPLES,
            "sample_count": 0,
            "probability_column": None,
            "overall": {},
            "slices": {},
            "blockers": blockers,
            "warnings": warnings,
            "overconfident_ranges": [],
            "underconfident_ranges": [],
            "recommended_probability_shrinkage": "unavailable",
        }
        write_json(report_path, report)
        return report

    frame = training["frame"].copy()
    probability_column = _find_probability_column(frame)

    if probability_column is None:
        blockers.append("no model or market probability column available")
        report = {
            "generated_at": _utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "status": "partial",
            "calibration_ready": False,
            "minimum_required_samples": MIN_CALIBRATION_SAMPLES,
            "sample_count": int(len(frame)),
            "probability_column": None,
            "overall": {},
            "slices": {},
            "blockers": blockers,
            "warnings": warnings,
            "overconfident_ranges": [],
            "underconfident_ranges": [],
            "recommended_probability_shrinkage": "unavailable",
        }
        write_json(report_path, report)
        return report

    frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")

    valid = frame["home_win"].isin([0, 1]) & frame[probability_column].notna()
    frame = frame[valid].copy()
    frame["probability_bucket"] = frame[probability_column].clip(0.01, 0.99).apply(_bucket_label)

    overall = _slice_metrics(frame, pd.Series([True] * len(frame), index=frame.index), probability_column)
    overconfident, underconfident = _range_recommendations(overall.get("reliability_bins") or [])

    slices = {
        "overall": overall,
        "low_confidence_45_55": _slice_metrics(frame, frame["probability_bucket"] == "low_confidence_45_55", probability_column),
        "medium_confidence_55_65": _slice_metrics(frame, frame["probability_bucket"] == "medium_confidence_55_65", probability_column),
        "high_confidence_65_plus": _slice_metrics(frame, frame["probability_bucket"] == "high_confidence_65_plus", probability_column),
    }

    for column, label in (
        ("lineup_context_available", "lineup_confirmed"),
        ("starter_context_available", "starter_confirmed"),
    ):
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            slices[f"{label}_yes"] = _slice_metrics(frame, numeric > 0, probability_column)
            slices[f"{label}_no"] = _slice_metrics(frame, numeric <= 0, probability_column)

    sample_count = int(overall.get("sample_count") or 0)
    ece = safe_float(overall.get("ece"))

    if sample_count < MIN_CALIBRATION_SAMPLES:
        blockers.append(f"calibration sample count below threshold: {sample_count} < {MIN_CALIBRATION_SAMPLES}")

    if ece is not None and ece > 0.05:
        blockers.append(f"ECE above readiness threshold: {ece:.4f} > 0.05")

    recommended_shrinkage = "none"
    if ece is None:
        recommended_shrinkage = "unavailable"
    elif ece > 0.10:
        recommended_shrinkage = "strong"
    elif ece > 0.05:
        recommended_shrinkage = "moderate"
    elif overconfident:
        recommended_shrinkage = "light"

    report = {
        "generated_at": _utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "status": "ok",
        "calibration_ready": len(blockers) == 0,
        "minimum_required_samples": MIN_CALIBRATION_SAMPLES,
        "sample_count": sample_count,
        "probability_column": probability_column,
        "overall": overall,
        "slices": slices,
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "overconfident_ranges": overconfident,
        "underconfident_ranges": underconfident,
        "recommended_probability_shrinkage": recommended_shrinkage,
    }

    write_json(report_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
