from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.model_training_common import (
    classification_metrics,
    safe_float,
    write_json,
)

REPORT_DIR = Path("report")
DATA_DIR = Path("data")

WALK_FORWARD_REPORT_PATH = REPORT_DIR / "walk_forward_validation_report.json"
WALK_FORWARD_PREDICTIONS_PATH = DATA_DIR / "walk_forward_predictions.csv"
CALIBRATION_REPORT_PATH = REPORT_DIR / "calibration_diagnostics_report.json"
MODEL_COMPARISON_PATH = REPORT_DIR / "model_comparison_report.json"
PREDICTION_TRUST_PATH = REPORT_DIR / "prediction_trust_report.json"
SAMPLE_STATE_PATH = DATA_DIR / "sample_state.json"
OUTPUT_PATH = REPORT_DIR / "model_decision_guardrail_report.json"

MIN_PROMOTION_MODEL_OOS = 300
MIN_PROMOTION_CLEAN_SAMPLES = 500
MIN_TRAIN_ELIGIBLE_SAMPLES = 300
MAX_ECE = 0.05
MAX_BUCKET_ECE = 0.10
MIN_BUCKET_SAMPLE = 20
MAX_HIGH_CONFIDENCE_ECE = 0.15

ALPHA_GRID = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, "file_missing"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)

    if not isinstance(payload, dict):
        return {}, "json_not_object"

    return payload, None


def _read_predictions(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), "file_missing"

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    required = {"model_name", "predicted_prob", "actual_home_win"}
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame(), f"missing columns: {sorted(missing)}"

    frame = frame.copy()
    frame["predicted_prob"] = pd.to_numeric(frame["predicted_prob"], errors="coerce")
    frame["actual_home_win"] = pd.to_numeric(frame["actual_home_win"], errors="coerce")
    frame = frame.dropna(subset=["predicted_prob", "actual_home_win"])
    frame["actual_home_win"] = frame["actual_home_win"].astype(int)
    frame["predicted_prob"] = frame["predicted_prob"].clip(0.01, 0.99)

    if "snapshot_time" in frame.columns:
        frame["_sort_time"] = pd.to_datetime(frame["snapshot_time"], errors="coerce", utc=True)
        frame = frame.sort_values(["_sort_time", "model_name"], kind="mergesort", na_position="last")

    return frame.reset_index(drop=True), None


def _shrink_probability(prob: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(0.5 + alpha * (prob - 0.5), 0.01, 0.99)


def _split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    split_index = max(1, int(len(frame) * 0.70))
    if split_index >= len(frame):
        split_index = max(1, len(frame) - 1)

    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def _choose_shrink_alpha(model_frame: pd.DataFrame) -> dict[str, Any]:
    train_frame, validation_frame = _split_train_validation(model_frame)

    if len(train_frame) < 20 or len(validation_frame) < 5:
        return {
            "attempted": False,
            "skip_reason": "not enough rows for shrinkage selection",
            "recommended_alpha": 1.0,
            "raw_validation_metrics": {},
            "shrunk_validation_metrics": {},
            "candidates": [],
        }

    y_validation = validation_frame["actual_home_win"].to_numpy(dtype=int)
    raw_validation_prob = validation_frame["predicted_prob"].to_numpy(dtype=float)
    raw_metrics = classification_metrics(y_validation, raw_validation_prob)

    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for alpha in ALPHA_GRID:
        shrunk_prob = _shrink_probability(raw_validation_prob, alpha)
        metrics = classification_metrics(y_validation, shrunk_prob)

        row = {
            "alpha": alpha,
            "brier": metrics.get("brier"),
            "logloss": metrics.get("logloss"),
            "accuracy": metrics.get("accuracy"),
            "ece": metrics.get("ece"),
        }
        candidates.append(row)

        brier = safe_float(metrics.get("brier"))
        logloss = safe_float(metrics.get("logloss"))
        ece = safe_float(metrics.get("ece"))

        if brier is None or logloss is None:
            continue

        score = (
            logloss,
            brier,
            ece if ece is not None else 999.0,
        )

        if best is None or score < best["score"]:
            best = {
                "alpha": alpha,
                "score": score,
                "metrics": metrics,
            }

    if best is None:
        return {
            "attempted": True,
            "skip_reason": "no valid alpha candidate",
            "recommended_alpha": 1.0,
            "raw_validation_metrics": raw_metrics,
            "shrunk_validation_metrics": {},
            "candidates": candidates,
        }

    return {
        "attempted": True,
        "skip_reason": "",
        "recommended_alpha": best["alpha"],
        "raw_validation_metrics": raw_metrics,
        "shrunk_validation_metrics": best["metrics"],
        "candidates": candidates,
    }


def _bucket_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}

    output: dict[str, Any] = {}

    for value, group in frame.groupby(column):
        if group.empty:
            continue

        output[str(value)] = classification_metrics(
            group["actual_home_win"],
            group["predicted_prob"],
        )

    return output


def _flag_bad_slices(
    *,
    model_name: str,
    bucket_metrics: Dict[str, Any],
    side_metrics: Dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    high = bucket_metrics.get("high_65_plus") or {}
    high_count = int(high.get("sample_count") or 0)
    high_ece = safe_float(high.get("ece"))
    high_accuracy = safe_float(high.get("accuracy"))
    high_brier = safe_float(high.get("brier"))

    if high_count >= MIN_BUCKET_SAMPLE and (
        (high_ece is not None and high_ece > MAX_HIGH_CONFIDENCE_ECE)
        or (high_accuracy is not None and high_accuracy < 0.50)
    ):
        findings.append(
            {
                "model_name": model_name,
                "slice": "high_65_plus",
                "severity": "critical",
                "reason": "high confidence bucket is overconfident or below 50% accuracy",
                "sample_count": high_count,
                "accuracy": high_accuracy,
                "brier": high_brier,
                "ece": high_ece,
                "recommended_action": "cap_or_shrink_high_confidence_probabilities",
            }
        )

    for side in ("away", "underdog"):
        metrics = side_metrics.get(side) or {}
        sample_count = int(metrics.get("sample_count") or 0)
        ece = safe_float(metrics.get("ece"))
        accuracy = safe_float(metrics.get("accuracy"))
        brier = safe_float(metrics.get("brier"))

        if sample_count >= MIN_BUCKET_SAMPLE and (
            (accuracy is not None and accuracy < 0.50)
            or (ece is not None and ece > MAX_BUCKET_ECE)
        ):
            findings.append(
                {
                    "model_name": model_name,
                    "slice": side,
                    "severity": "warning",
                    "reason": f"{side} slice is weak or poorly calibrated",
                    "sample_count": sample_count,
                    "accuracy": accuracy,
                    "brier": brier,
                    "ece": ece,
                    "recommended_action": "tracking_only_until_slice_improves",
                }
            )

    return findings


def _model_oos_count(model_frame: pd.DataFrame) -> int:
    if model_frame.empty:
        return 0

    if "game_id" in model_frame.columns:
        return int(model_frame["game_id"].astype(str).nunique())

    return int(len(model_frame))


def _market_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty or "model_name" not in predictions.columns:
        return {}

    required_columns = {"actual_home_win", "predicted_prob"}
    if not required_columns.issubset(set(predictions.columns)):
        return {}

    market = predictions[predictions["model_name"] == "market_no_vig_baseline"].copy()
    if market.empty:
        return {}

    return classification_metrics(
        market["actual_home_win"],
        market["predicted_prob"],
    )
    

def build_report(
    *,
    walk_forward_report_path: Path = WALK_FORWARD_REPORT_PATH,
    walk_forward_predictions_path: Path = WALK_FORWARD_PREDICTIONS_PATH,
    calibration_report_path: Path = CALIBRATION_REPORT_PATH,
    model_comparison_path: Path = MODEL_COMPARISON_PATH,
    prediction_trust_path: Path = PREDICTION_TRUST_PATH,
    sample_state_path: Path = SAMPLE_STATE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    warnings: list[str] = []
    blockers: list[str] = []

    walk_forward_report, walk_forward_error = _read_json(walk_forward_report_path)
    calibration_report, calibration_error = _read_json(calibration_report_path)
    model_comparison, model_comparison_error = _read_json(model_comparison_path)
    prediction_trust, prediction_trust_error = _read_json(prediction_trust_path)
    sample_state, sample_state_error = _read_json(sample_state_path)
    predictions, predictions_error = _read_predictions(walk_forward_predictions_path)

    for name, error in (
        ("walk_forward_report", walk_forward_error),
        ("calibration_report", calibration_error),
        ("model_comparison", model_comparison_error),
        ("prediction_trust", prediction_trust_error),
        ("sample_state", sample_state_error),
        ("walk_forward_predictions", predictions_error),
    ):
        if error:
            warnings.append(f"{name} unavailable: {error}")

    clean_settled = int(sample_state.get("clean_settled_snapshots") or 0)
    train_eligible = int(sample_state.get("train_eligible_samples") or 0)
    calibration_ece = safe_float((calibration_report.get("overall") or {}).get("ece"))
    calibration_ready = bool(calibration_report.get("calibration_ready"))
    calibration_sample_count = int(calibration_report.get("sample_count") or 0)

    market = _market_metrics(predictions)
    market_brier = safe_float(market.get("brier"))
    market_logloss = safe_float(market.get("logloss"))

    model_reports: list[dict[str, Any]] = []
    critical_findings: list[dict[str, Any]] = []
    recommended_challenger = model_comparison.get("recommended_challenger")

    if predictions.empty:
        blockers.append("walk-forward predictions unavailable")
    else:
        for model_name, model_frame in predictions.groupby("model_name"):
            model_name = str(model_name)
            if model_name == "market_no_vig_baseline":
                continue

            model_oos = _model_oos_count(model_frame)
            raw_metrics = classification_metrics(
                model_frame["actual_home_win"],
                model_frame["predicted_prob"],
            )

            bucket = _bucket_metrics(model_frame, "confidence_bucket")
            side = _bucket_metrics(model_frame, "prediction_side")
            findings = _flag_bad_slices(
                model_name=model_name,
                bucket_metrics=bucket,
                side_metrics=side,
            )
            critical_findings.extend(findings)

            shrinkage = _choose_shrink_alpha(model_frame)

            model_brier = safe_float(raw_metrics.get("brier"))
            model_logloss = safe_float(raw_metrics.get("logloss"))
            model_ece = safe_float(raw_metrics.get("ece"))

            beats_market_brier = (
                model_brier is not None
                and market_brier is not None
                and model_brier < market_brier
            )
            beats_market_logloss = (
                model_logloss is not None
                and market_logloss is not None
                and model_logloss < market_logloss
            )

            model_blockers: list[str] = []

            if model_oos < MIN_PROMOTION_MODEL_OOS:
                model_blockers.append(
                    f"per-model OOS below threshold: {model_oos} < {MIN_PROMOTION_MODEL_OOS}"
                )

            if not (beats_market_brier or beats_market_logloss):
                model_blockers.append("does not beat market brier or logloss")

            if model_ece is None or model_ece > MAX_ECE:
                model_blockers.append(
                    f"model ECE not acceptable: {model_ece} > {MAX_ECE}"
                )

            if findings:
                model_blockers.append("weak slices detected")

            if model_oos >= MIN_PROMOTION_MODEL_OOS and not model_blockers:
                recommended_usage = "RESEARCH_REVIEW_CANDIDATE"
            elif beats_market_brier or beats_market_logloss:
                recommended_usage = "SHADOW_CHALLENGER_OBSERVE"
            else:
                recommended_usage = "TRACKING_ONLY"

            model_reports.append(
                {
                    "model_name": model_name,
                    "oos_count": model_oos,
                    "raw_metrics": raw_metrics,
                    "beats_market_brier": beats_market_brier,
                    "beats_market_logloss": beats_market_logloss,
                    "bucket_metrics": bucket,
                    "side_metrics": side,
                    "slice_findings": findings,
                    "shrinkage": shrinkage,
                    "recommended_usage": recommended_usage,
                    "promotion_blockers": model_blockers,
                    "shadow_only": True,
                    "production_model_replacement_allowed": False,
                }
            )

    if clean_settled < MIN_PROMOTION_CLEAN_SAMPLES:
        blockers.append(
            f"clean settled samples below promotion threshold: {clean_settled} < {MIN_PROMOTION_CLEAN_SAMPLES}"
        )

    if train_eligible < MIN_TRAIN_ELIGIBLE_SAMPLES:
        blockers.append(
            f"train eligible samples below threshold: {train_eligible} < {MIN_TRAIN_ELIGIBLE_SAMPLES}"
        )

    if calibration_sample_count < MIN_TRAIN_ELIGIBLE_SAMPLES:
        blockers.append(
            f"calibration samples below threshold: {calibration_sample_count} < {MIN_TRAIN_ELIGIBLE_SAMPLES}"
        )

    if not calibration_ready:
        blockers.append("calibration report is not ready")

    if calibration_ece is not None and calibration_ece > MAX_ECE:
        blockers.append(f"overall ECE above threshold: {calibration_ece} > {MAX_ECE}")

    high_confidence_blocked = any(
        finding.get("slice") == "high_65_plus"
        and finding.get("severity") == "critical"
        for finding in critical_findings
    )

    weak_slice_names = sorted(
        {
            str(finding.get("slice"))
            for finding in critical_findings
            if finding.get("slice")
        }
    )

    probability_policy = {
        "official_probability_change_allowed": False,
        "shadow_probability_shrinkage_allowed": True,
        "recommended_default_alpha": 0.70 if high_confidence_blocked else 0.85,
        "recommended_max_display_confidence": 0.65 if high_confidence_blocked else 0.75,
        "block_high_confidence_language": high_confidence_blocked,
        "blocked_or_downgraded_slices": weak_slice_names,
        "explanation": (
            "Because calibration and high-confidence slices are not promotion-ready, "
            "shadow probabilities should be shrunk toward 50% and high-confidence language should be disabled."
        ),
    }

    report = {
        "generated_at": _utc_now(),
        "status": "blocked" if blockers else "ok",
        "decision": "NO_PROMOTION_SHADOW_ONLY",
        "recommended_challenger": recommended_challenger,
        "clean_settled_samples": clean_settled,
        "train_eligible_samples": train_eligible,
        "calibration_sample_count": calibration_sample_count,
        "calibration_ready": calibration_ready,
        "calibration_ece": calibration_ece,
        "market_metrics": market,
        "models": model_reports,
        "critical_findings": critical_findings,
        "probability_policy": probability_policy,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "next_actions": [
            "Do not replace official prediction model.",
            "Do not lower sample gates.",
            "Use shadow shrinkage only for research display and diagnostics.",
            "Disable high-confidence language until high_65_plus ECE improves.",
            "Treat away and underdog slices as tracking-only until slice metrics improve.",
            "Prioritize calibration, Brier, LogLoss, and CLV over raw accuracy.",
        ],
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "user_funds_handled": False,
        "production_model_replacement_allowed": False,
    }

    write_json(output_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
