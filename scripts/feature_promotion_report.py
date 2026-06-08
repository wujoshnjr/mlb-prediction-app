from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.feature_schema import (
    CANDIDATE_SHADOW_FEATURES,
    FEATURE_METADATA,
    MODEL_FEATURES,
    TRACKING_ONLY_FEATURES,
)
from scripts.model_training_common import (
    PIPELINE_VERSION,
    build_feature_matrix,
    build_training_frame,
    classification_metrics,
    detect_leakage_columns,
    safe_float,
    time_ordered_split,
    write_json,
)


REPORT_PATH = Path("report/feature_promotion_report.json")

MIN_ABLATION_SAMPLES = 80
MIN_REVIEW_SAMPLES = 300
MIN_AVAILABILITY_CANDIDATE = 0.70
IDEAL_AVAILABILITY = 0.90
MIN_NON_ZERO_RATE = 0.01


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _feature_leakage_risk(feature: str) -> str:
    if detect_leakage_columns([feature]):
        return "high"

    metadata = FEATURE_METADATA.get(feature) or {}
    risk = str(metadata.get("leakage_risk", "low")).lower()

    if risk not in {"low", "medium", "high"}:
        return "low"

    return risk


def _series_stats(frame: pd.DataFrame, feature: str) -> Dict[str, Any]:
    if feature not in frame.columns:
        return {
            "feature_name": feature,
            "availability_rate": 0.0,
            "missing_rate": 1.0,
            "non_zero_rate": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "unique_count": 0,
            "data_type": "missing",
        }

    raw = frame[feature]
    numeric = pd.to_numeric(raw, errors="coerce")
    available = numeric.notna()
    availability_rate = float(available.mean()) if len(numeric) else 0.0
    missing_rate = float(1.0 - availability_rate)

    if available.any():
        available_values = numeric[available]
        non_zero_rate = float((available_values != 0).mean())
        mean = _finite_or_none(available_values.mean())
        std = _finite_or_none(available_values.std(ddof=0))
        min_value = _finite_or_none(available_values.min())
        max_value = _finite_or_none(available_values.max())
        unique_count = int(available_values.nunique(dropna=True))
    else:
        non_zero_rate = 0.0
        mean = None
        std = None
        min_value = None
        max_value = None
        unique_count = 0

    return {
        "feature_name": feature,
        "availability_rate": round(availability_rate, 6),
        "missing_rate": round(missing_rate, 6),
        "non_zero_rate": round(non_zero_rate, 6),
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "unique_count": unique_count,
        "data_type": str(raw.dtype),
    }


def _empty_ablation(reason: str) -> Dict[str, Any]:
    return {
        "attempted": False,
        "skip_reason": reason,
        "baseline": {},
        "with_feature": {},
        "delta_brier": None,
        "delta_logloss": None,
        "delta_accuracy": None,
        "delta_auc": None,
        "delta_ece": None,
    }


def _fit_predict_logistic(split: Dict[str, Any]) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if len(np.unique(split["y_train"])) < 2:
        return None, warnings, ["train target has one class"]

    try:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(split["X_train"], split["y_train"])
        prob = model.predict_proba(split["X_validation"])[:, 1]
        return np.asarray(prob, dtype=float), warnings, errors
    except Exception as exc:
        errors.append(str(exc))
        return None, warnings, errors


def _ablation_for_feature(frame: pd.DataFrame, feature: str) -> Dict[str, Any]:
    base_result = build_feature_matrix(
        frame,
        base_features=MODEL_FEATURES,
    )
    if not base_result.get("ok"):
        return _empty_ablation(base_result.get("skip_reason") or "baseline feature matrix unavailable")

    candidate_result = build_feature_matrix(
        frame,
        base_features=MODEL_FEATURES,
        candidate_features=[feature],
        allow_tracking_only=True,
    )
    if not candidate_result.get("ok"):
        return _empty_ablation(candidate_result.get("skip_reason") or "candidate feature matrix unavailable")

    base_split = time_ordered_split(
        base_result["X"],
        base_result["y"],
        base_result["frame"],
        min_train_samples=20,
        min_calibration_samples=5,
        min_validation_samples=5,
    )
    if not base_split.get("ok"):
        return _empty_ablation(base_split.get("skip_reason") or "baseline split unavailable")

    candidate_split = time_ordered_split(
        candidate_result["X"],
        candidate_result["y"],
        candidate_result["frame"],
        min_train_samples=20,
        min_calibration_samples=5,
        min_validation_samples=5,
    )
    if not candidate_split.get("ok"):
        return _empty_ablation(candidate_split.get("skip_reason") or "candidate split unavailable")

    base_prob, base_warnings, base_errors = _fit_predict_logistic(base_split)
    candidate_prob, candidate_warnings, candidate_errors = _fit_predict_logistic(candidate_split)

    if base_prob is None or candidate_prob is None:
        return {
            "attempted": True,
            "skip_reason": "model fitting failed",
            "baseline": {"warnings": base_warnings, "errors": base_errors},
            "with_feature": {"warnings": candidate_warnings, "errors": candidate_errors},
            "delta_brier": None,
            "delta_logloss": None,
            "delta_accuracy": None,
            "delta_auc": None,
            "delta_ece": None,
        }

    base_metrics = classification_metrics(base_split["y_validation"], base_prob)
    candidate_metrics = classification_metrics(candidate_split["y_validation"], candidate_prob)

    def delta(metric: str) -> Optional[float]:
        left = safe_float(candidate_metrics.get(metric))
        right = safe_float(base_metrics.get(metric))
        if left is None or right is None:
            return None
        return float(left - right)

    return {
        "attempted": True,
        "skip_reason": "",
        "baseline": base_metrics,
        "with_feature": candidate_metrics,
        "delta_brier": delta("brier"),
        "delta_logloss": delta("logloss"),
        "delta_accuracy": delta("accuracy"),
        "delta_auc": delta("auc"),
        "delta_ece": delta("ece"),
    }


def _recommended_status(
    *,
    sample_count: int,
    leakage_risk: str,
    availability_rate: float,
    non_zero_rate: float,
    ablation: Dict[str, Any],
) -> Tuple[str, List[str], List[str]]:
    blockers: List[str] = []
    warnings: List[str] = []

    if leakage_risk == "high":
        blockers.append("high leakage risk")
        return "exclude", blockers, warnings

    if availability_rate < MIN_AVAILABILITY_CANDIDATE:
        blockers.append(
            f"availability below candidate threshold: {availability_rate:.4f} < {MIN_AVAILABILITY_CANDIDATE}"
        )

    if non_zero_rate <= MIN_NON_ZERO_RATE:
        blockers.append(
            f"non_zero_rate too low: {non_zero_rate:.4f} <= {MIN_NON_ZERO_RATE}"
        )

    if availability_rate < IDEAL_AVAILABILITY and availability_rate >= MIN_AVAILABILITY_CANDIDATE:
        warnings.append(
            f"availability below ideal threshold: {availability_rate:.4f} < {IDEAL_AVAILABILITY}"
        )

    if blockers:
        return "tracking_only", blockers, warnings

    if sample_count < MIN_REVIEW_SAMPLES:
        blockers.append(f"insufficient samples for review: {sample_count} < {MIN_REVIEW_SAMPLES}")
        return "candidate_shadow", blockers, warnings

    if ablation.get("attempted"):
        delta_brier = safe_float(ablation.get("delta_brier"))
        delta_logloss = safe_float(ablation.get("delta_logloss"))
        delta_ece = safe_float(ablation.get("delta_ece"))

        if delta_brier is not None and delta_brier > 0:
            blockers.append(f"validation brier worsened: delta={delta_brier:.6f}")

        if delta_logloss is not None and delta_logloss > 0:
            blockers.append(f"validation logloss worsened: delta={delta_logloss:.6f}")

        if delta_ece is not None and delta_ece > 0.02:
            warnings.append(f"calibration worsened: delta_ece={delta_ece:.6f}")

    else:
        blockers.append(ablation.get("skip_reason") or "ablation unavailable")

    if blockers:
        return "candidate_shadow", blockers, warnings

    return "ready_for_review", blockers, warnings


def build_report(
    *,
    snapshot_path: Path = Path("data/prediction_snapshots.csv"),
    finalized_path: Path = Path("data/finalized_games.csv"),
    report_path: Path = REPORT_PATH,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    blockers: List[str] = []

    training_result = build_training_frame(
        snapshot_path=snapshot_path,
        finalized_path=finalized_path,
        pipeline_version=PIPELINE_VERSION,
    )

    warnings.extend(training_result.get("warnings") or [])
    errors.extend(training_result.get("errors") or [])

    if not training_result.get("ok"):
        blockers.append(training_result.get("skip_reason") or "training frame unavailable")
        report = {
            "generated_at": _utc_now(),
            "status": "partial",
            "pipeline_version": PIPELINE_VERSION,
            "sample_count": 0,
            "minimum_ablation_samples": MIN_ABLATION_SAMPLES,
            "minimum_review_samples": MIN_REVIEW_SAMPLES,
            "feature_count": 0,
            "candidate_shadow_count": 0,
            "ready_for_review_count": 0,
            "features": [],
            "priority_features": CANDIDATE_SHADOW_FEATURES,
            "blockers": blockers,
            "warnings": sorted(set(warnings)),
            "errors": errors,
            "recommendations": [
                "Fix finalized-joined training data before feature promotion research.",
            ],
        }
        write_json(report_path, report)
        return report

    frame = training_result["frame"]
    sample_count = int(len(frame))
    feature_reports: List[Dict[str, Any]] = []

    for feature in TRACKING_ONLY_FEATURES:
        stats = _series_stats(frame, feature)
        leakage_risk = _feature_leakage_risk(feature)
        metadata = FEATURE_METADATA.get(feature) or {}

        if sample_count >= MIN_ABLATION_SAMPLES and leakage_risk != "high":
            ablation = _ablation_for_feature(frame, feature)
        else:
            ablation = _empty_ablation(
                f"sample_count below ablation threshold: {sample_count} < {MIN_ABLATION_SAMPLES}"
                if sample_count < MIN_ABLATION_SAMPLES
                else "high leakage risk"
            )

        recommended_status, promotion_blockers, feature_warnings = _recommended_status(
            sample_count=sample_count,
            leakage_risk=leakage_risk,
            availability_rate=float(stats["availability_rate"]),
            non_zero_rate=float(stats["non_zero_rate"]),
            ablation=ablation,
        )

        feature_reports.append(
            {
                **stats,
                "group": metadata.get("group", "ungrouped"),
                "leakage_risk": leakage_risk,
                "allow_in_main_model": bool(metadata.get("allow_in_main_model", False)),
                "allow_in_shadow_model": bool(metadata.get("allow_in_shadow_model", feature in CANDIDATE_SHADOW_FEATURES)),
                "recommended_status": recommended_status,
                "promotion_blockers": promotion_blockers,
                "warnings": feature_warnings,
                "ablation": ablation,
            }
        )

    candidate_shadow_count = sum(
        1 for item in feature_reports if item.get("recommended_status") == "candidate_shadow"
    )
    ready_for_review_count = sum(
        1 for item in feature_reports if item.get("recommended_status") == "ready_for_review"
    )

    if sample_count < MIN_REVIEW_SAMPLES:
        warnings.append(
            f"Feature promotion review blocked by sample size: {sample_count} < {MIN_REVIEW_SAMPLES}"
        )

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "pipeline_version": PIPELINE_VERSION,
        "sample_count": sample_count,
        "minimum_ablation_samples": MIN_ABLATION_SAMPLES,
        "minimum_review_samples": MIN_REVIEW_SAMPLES,
        "feature_count": len(feature_reports),
        "candidate_shadow_count": candidate_shadow_count,
        "ready_for_review_count": ready_for_review_count,
        "features": feature_reports,
        "priority_features": CANDIDATE_SHADOW_FEATURES,
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "errors": errors,
        "recommendations": [
            "Do not move tracking-only features into MODEL_FEATURES directly.",
            "Use candidate_shadow features only in model lab until walk-forward and calibration evidence are stronger.",
            "Require stable OOS brier/logloss and calibration before ready_for_review.",
            "Current sample size may be too small for final feature promotion decisions.",
        ],
    }

    write_json(report_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
