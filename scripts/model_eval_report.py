# scripts/model_eval_report.py
"""Generate an out-of-sample model evaluation report.

This diagnostic script is intentionally read-only with respect to production
serving. It trains a temporary evaluation model from data/training_samples.csv,
writes OOS predictions for diagnostics, and never enables live betting,
automated wagering, or production model replacement.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    CalibratedClassifierCV = None  # type: ignore[assignment]
    SimpleImputer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    balanced_accuracy_score = None  # type: ignore[assignment]
    brier_score_loss = None  # type: ignore[assignment]
    confusion_matrix = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    log_loss = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    SKLEARN_IMPORT_ERROR = str(exc)

try:
    import config
except ImportError:  # pragma: no cover

    class config:  # type: ignore[no-redef]
        PIPELINE_VERSION = "baseline_v2_clean"


try:
    from scripts.feature_schema import (
        CORE_MODEL_FEATURES,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
    )
except ImportError:  # pragma: no cover
    from feature_schema import (  # type: ignore[no-redef]
        CORE_MODEL_FEATURES,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
    )


DATA_PATH = Path("data/training_samples.csv")
OOS_CSV_PATH = Path("data/oos_predictions_with_labels.csv")
REPORT_JSON_PATH = Path("report/model_eval_report.json")
COLLAPSE_REPORT_PATH = Path("report/prediction_collapse_report.json")

LABEL_COLUMNS = [
    "home_win",
    "y_true",
    "label",
    "result_home_win",
]

MIN_TOTAL_SAMPLES = 30
MIN_TRAIN_SAMPLES = 20
MIN_CALIBRATION_SAMPLES = 5
MIN_TEST_SAMPLES = 5

# Guardrail thresholds. These are intentionally conservative and report-only.
AUC_RANDOM_BAND_LOW = 0.48
AUC_RANDOM_BAND_HIGH = 0.52
BALANCED_ACCURACY_RANDOM_BAND_LOW = 0.48
BALANCED_ACCURACY_RANDOM_BAND_HIGH = 0.52
MAX_SINGLE_CLASS_PREDICTION_RATE = 0.85
MIN_PROBABILITY_STD = 0.02
BRIER_RANDOM_REFERENCE = 0.25
LOGLOSS_RANDOM_REFERENCE = 0.6931471805599453
BRIER_RANDOM_TOLERANCE = 0.015
LOGLOSS_RANDOM_TOLERANCE = 0.02


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_json_value(value: Any) -> Any:
    """Return a JSON-safe value with no NaN or Infinity."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): clean_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json_value(child) for child in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        if hasattr(value, "item"):
            return clean_json_value(value.item())
    except Exception:
        pass
    return str(value)


def write_json_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            clean_json_value(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def write_json_report(report: dict[str, Any], path: Path | None = None) -> None:
    output_path = path if path is not None else REPORT_JSON_PATH
    write_json_payload(report, output_path)


def get_collapse_report_path() -> Path:
    if REPORT_JSON_PATH != Path("report/model_eval_report.json"):
        return REPORT_JSON_PATH.parent / "prediction_collapse_report.json"
    return COLLAPSE_REPORT_PATH


def base_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "report_type": "model_eval_report",
        "status": "skipped",
        "training_source": str(DATA_PATH),
        "oos_prediction_output": str(OOS_CSV_PATH),
        "collapse_report_output": str(get_collapse_report_path()),
        "pipeline_version": str(getattr(config, "PIPELINE_VERSION", "baseline_v2_clean")),
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "features_used": list(CORE_MODEL_FEATURES),
        "label_column": "",
        "sample_count": 0,
        "train_sample_count": 0,
        "calibration_sample_count": 0,
        "test_sample_count": 0,
        "metrics": {},
        "baseline_metrics": {},
        "model_vs_baselines": {},
        "collapse_guardrail": {},
        "confusion_matrix": [],
        "warnings": [],
        "errors": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def find_label_column(frame: pd.DataFrame) -> str | None:
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            return column
    return None


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def skip_report(report: dict[str, Any], reason: str) -> dict[str, Any]:
    report["status"] = "skipped"
    report["errors"].append(reason)
    write_json_report(report)
    return report


def error_report(report: dict[str, Any], reason: str) -> dict[str, Any]:
    report["status"] = "error"
    report["errors"].append(reason)
    write_json_report(report)
    return report


def make_calibrator(estimator: Any) -> Any:
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    except Exception:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv="prefit")


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (clipped >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, clipped),
        "brier": float(brier_score_loss(y_true, clipped)),
        "logloss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "predicted_positive_rate": float(np.mean(y_pred)) if len(y_pred) else None,
        "predicted_negative_rate": float(1.0 - np.mean(y_pred)) if len(y_pred) else None,
        "probability_mean": float(np.mean(clipped)) if len(clipped) else None,
        "probability_std": float(np.std(clipped)) if len(clipped) else None,
        "probability_min": float(np.min(clipped)) if len(clipped) else None,
        "probability_max": float(np.max(clipped)) if len(clipped) else None,
    }


def constant_probability_baseline(
    y_true: np.ndarray,
    probability: float,
    name: str,
) -> dict[str, Any]:
    y_prob = np.full_like(y_true, fill_value=float(probability), dtype=float)
    metrics = probability_metrics(y_true, y_prob)
    metrics["baseline_name"] = name
    metrics["constant_probability"] = float(probability)
    return metrics


def build_baseline_metrics(
    y_train: np.ndarray,
    y_calib: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    reference_target = np.concatenate([y_train, y_calib]) if len(y_calib) else y_train
    base_rate = float(np.mean(reference_target)) if len(reference_target) else 0.5
    return {
        "constant_0_5": constant_probability_baseline(y_test, 0.5, "constant_0_5"),
        "base_rate_train_calibration": constant_probability_baseline(
            y_test,
            base_rate,
            "base_rate_train_calibration",
        ),
        "always_home": constant_probability_baseline(y_test, 1.0 - 1e-6, "always_home"),
        "always_away": constant_probability_baseline(y_test, 1e-6, "always_away"),
    }


def compare_model_to_baselines(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, baseline in baseline_metrics.items():
        comparison: dict[str, Any] = {}
        for key in ("brier", "logloss"):
            model_value = model_metrics.get(key)
            baseline_value = baseline.get(key)
            if model_value is not None and baseline_value is not None:
                comparison[f"delta_{key}"] = float(model_value) - float(baseline_value)
                comparison[f"beats_{key}"] = float(model_value) < float(baseline_value)
        model_auc = model_metrics.get("roc_auc")
        baseline_auc = baseline.get("roc_auc")
        if model_auc is not None and baseline_auc is not None:
            comparison["delta_roc_auc"] = float(model_auc) - float(baseline_auc)
            comparison["beats_roc_auc"] = float(model_auc) > float(baseline_auc)
        comparisons[name] = comparison
    return comparisons


def build_collapse_guardrail(
    metrics: dict[str, Any],
    confusion: list[list[int]],
    test_sample_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    auc = metrics.get("roc_auc")
    balanced_accuracy = metrics.get("balanced_accuracy")
    predicted_positive_rate = metrics.get("predicted_positive_rate")
    predicted_negative_rate = metrics.get("predicted_negative_rate")
    probability_std = metrics.get("probability_std")
    brier = metrics.get("brier")
    logloss = metrics.get("logloss")

    if auc is not None and AUC_RANDOM_BAND_LOW <= float(auc) <= AUC_RANDOM_BAND_HIGH:
        reasons.append("roc_auc_near_random")
    if (
        balanced_accuracy is not None
        and BALANCED_ACCURACY_RANDOM_BAND_LOW <= float(balanced_accuracy) <= BALANCED_ACCURACY_RANDOM_BAND_HIGH
    ):
        reasons.append("balanced_accuracy_near_random")
    if predicted_positive_rate is not None and float(predicted_positive_rate) >= MAX_SINGLE_CLASS_PREDICTION_RATE:
        reasons.append("single_class_positive_prediction_collapse")
    if predicted_negative_rate is not None and float(predicted_negative_rate) >= MAX_SINGLE_CLASS_PREDICTION_RATE:
        reasons.append("single_class_negative_prediction_collapse")
    if probability_std is not None and float(probability_std) < MIN_PROBABILITY_STD:
        reasons.append("probability_distribution_too_narrow")
    if brier is not None and abs(float(brier) - BRIER_RANDOM_REFERENCE) <= BRIER_RANDOM_TOLERANCE:
        reasons.append("brier_near_random_0_25")
    if logloss is not None and abs(float(logloss) - LOGLOSS_RANDOM_REFERENCE) <= LOGLOSS_RANDOM_TOLERANCE:
        reasons.append("logloss_near_random_0_693")
    if test_sample_count < 50:
        reasons.append("test_sample_count_too_small_for_stable_claims")

    collapsed = any(
        reason
        in {
            "roc_auc_near_random",
            "balanced_accuracy_near_random",
            "single_class_positive_prediction_collapse",
            "single_class_negative_prediction_collapse",
            "probability_distribution_too_narrow",
        }
        for reason in reasons
    )

    return {
        "status": "failed" if collapsed else "ok",
        "model_has_no_discrimination_power": bool(collapsed),
        "do_not_promote": bool(collapsed),
        "collapse_reasons": reasons,
        "thresholds": {
            "auc_random_band_low": AUC_RANDOM_BAND_LOW,
            "auc_random_band_high": AUC_RANDOM_BAND_HIGH,
            "balanced_accuracy_random_band_low": BALANCED_ACCURACY_RANDOM_BAND_LOW,
            "balanced_accuracy_random_band_high": BALANCED_ACCURACY_RANDOM_BAND_HIGH,
            "max_single_class_prediction_rate": MAX_SINGLE_CLASS_PREDICTION_RATE,
            "min_probability_std": MIN_PROBABILITY_STD,
            "brier_random_reference": BRIER_RANDOM_REFERENCE,
            "logloss_random_reference": LOGLOSS_RANDOM_REFERENCE,
        },
        "confusion_matrix": confusion,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def write_collapse_report(
    model_eval_report: dict[str, Any],
    collapse_guardrail: dict[str, Any],
) -> None:
    payload = {
        "generated_at": utc_now(),
        "report_type": "prediction_collapse_report",
        "status": collapse_guardrail.get("status", "unknown"),
        "model_has_no_discrimination_power": collapse_guardrail.get("model_has_no_discrimination_power", False),
        "do_not_promote": collapse_guardrail.get("do_not_promote", True),
        "collapse_reasons": collapse_guardrail.get("collapse_reasons", []),
        "source_model_eval_report": str(REPORT_JSON_PATH),
        "sample_count": model_eval_report.get("sample_count"),
        "test_sample_count": model_eval_report.get("test_sample_count"),
        "metrics": model_eval_report.get("metrics", {}),
        "baseline_metrics": model_eval_report.get("baseline_metrics", {}),
        "model_vs_baselines": model_eval_report.get("model_vs_baselines", {}),
        "confusion_matrix": model_eval_report.get("confusion_matrix", []),
        "recommendations": [
            "Keep model in tracking-only mode until OOS discrimination and calibration beat simple baselines.",
            "Investigate feature quality and expand finalized samples before artifact promotion.",
            "Do not use hit rate alone; require Brier/logloss/AUC improvement over baselines.",
        ],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    write_json_payload(payload, get_collapse_report_path())


def load_clean_training_samples(report: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    if not DATA_PATH.exists():
        return pd.DataFrame(), None
    try:
        frame = pd.read_csv(DATA_PATH)
    except Exception as exc:
        report["errors"].append(f"unable_to_read_training_samples:{exc}")
        return pd.DataFrame(), None
    if frame.empty:
        report["errors"].append("training_samples_empty")
        return pd.DataFrame(), None
    label_column = find_label_column(frame)
    if label_column is None:
        report["errors"].append("training_samples_missing_label_column")
        return pd.DataFrame(), None
    frame = frame.copy()
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame[frame[label_column].isin([0, 1])].copy()
    frame[label_column] = frame[label_column].astype(int)
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str).str.strip()
        frame = frame[frame["game_id"] != ""].copy()
        frame = frame.drop_duplicates("game_id", keep="first")
    pipeline_version = str(getattr(config, "PIPELINE_VERSION", "baseline_v2_clean"))
    if "pipeline_version" in frame.columns:
        frame = frame[frame["pipeline_version"].astype(str) == pipeline_version].copy()
    missing_core_features = [
        feature for feature in CORE_MODEL_FEATURES if feature not in frame.columns
    ]
    if missing_core_features:
        report["errors"].append("missing_core_features")
        report["missing_core_features"] = missing_core_features
        return pd.DataFrame(), label_column
    if frame.empty:
        report["errors"].append("no_clean_training_samples_after_contract_filters")
        return pd.DataFrame(), label_column
    return frame.reset_index(drop=True), label_column


def sort_for_time_order(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "snapshot_created_at" in work.columns:
        work["_eval_sort_time"] = pd.to_datetime(work["snapshot_created_at"], errors="coerce", utc=True)
    elif "game_date" in work.columns:
        work["_eval_sort_time"] = pd.to_datetime(work["game_date"], errors="coerce", utc=True)
    else:
        work["_eval_sort_time"] = pd.NaT
    if work["_eval_sort_time"].notna().any():
        work = work.sort_values(["_eval_sort_time"]).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)
    return work


def generate_report() -> dict[str, Any]:
    report = base_report()

    if SKLEARN_IMPORT_ERROR is not None:
        return skip_report(report, f"sklearn_unavailable:{SKLEARN_IMPORT_ERROR}")

    frame, label_column = load_clean_training_samples(report)
    if label_column:
        report["label_column"] = label_column

    if not DATA_PATH.exists():
        return skip_report(report, "training_samples_missing")

    if frame.empty or label_column is None:
        return skip_report(
            report,
            report["errors"][-1] if report["errors"] else "no_clean_training_samples",
        )

    frame = sort_for_time_order(frame)
    sample_count = int(len(frame))
    report["sample_count"] = sample_count

    if sample_count < MIN_TOTAL_SAMPLES:
        return skip_report(report, f"insufficient_samples:{sample_count}<{MIN_TOTAL_SAMPLES}")

    for feature in CORE_MODEL_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    matrix = frame[CORE_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    target = frame[label_column].to_numpy(dtype=int)

    row_count = len(frame)
    train_end = int(row_count * 0.70)
    calibration_end = int(row_count * 0.85)

    train_count = train_end
    calibration_count = calibration_end - train_end
    test_count = row_count - calibration_end

    report["train_sample_count"] = int(train_count)
    report["calibration_sample_count"] = int(calibration_count)
    report["test_sample_count"] = int(test_count)

    if train_count < MIN_TRAIN_SAMPLES or calibration_count < MIN_CALIBRATION_SAMPLES or test_count < MIN_TEST_SAMPLES:
        return skip_report(report, "not_enough_samples_after_train_calibration_test_split")

    x_train = matrix[:train_end]
    y_train = target[:train_end]
    x_calib = matrix[train_end:calibration_end]
    y_calib = target[train_end:calibration_end]
    x_test = matrix[calibration_end:]
    y_test = target[calibration_end:]

    for name, subset in (("train", y_train), ("calibration", y_calib), ("test", y_test)):
        if len(np.unique(subset)) < 2:
            return skip_report(report, f"{name}_set_contains_only_one_target_class")

    base_model = Pipeline(
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

    try:
        base_model.fit(x_train, y_train)
        calibrated_model = make_calibrator(base_model)
        calibrated_model.fit(x_calib, y_calib)
        y_prob = calibrated_model.predict_proba(x_test)[:, 1]
    except Exception as exc:
        return error_report(report, f"model_eval_failed:{exc}")

    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = probability_metrics(y_test, y_prob)
    baseline_metrics = build_baseline_metrics(y_train, y_calib, y_test)
    model_vs_baselines = compare_model_to_baselines(metrics, baseline_metrics)
    confusion = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    collapse_guardrail = build_collapse_guardrail(metrics, confusion, test_count)

    report["status"] = "ok"
    report["metrics"] = metrics
    report["baseline_metrics"] = baseline_metrics
    report["model_vs_baselines"] = model_vs_baselines
    report["collapse_guardrail"] = collapse_guardrail
    report["confusion_matrix"] = confusion

    if collapse_guardrail.get("model_has_no_discrimination_power"):
        report["warnings"].append("model_has_no_discrimination_power")
    if any(
        comparison.get("delta_brier") is not None and comparison.get("delta_brier") >= 0
        for comparison in model_vs_baselines.values()
    ):
        report["warnings"].append("model_does_not_beat_all_brier_baselines")

    manual_prob_col = first_existing_column(
        frame,
        ["manual_prob_home", "manual_no_odds_pred", "manual_pred", "manual_prediction"],
    )
    final_prob_col = first_existing_column(
        frame,
        ["final_prob_home", "predicted_home_win_pct", "displayed_home_win_pct"],
    )
    metadata_columns = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "selected_side",
        "edge_bucket",
        "odds_quality_status",
        "model_source",
    ]
    test_rows = frame.iloc[calibration_end:].copy()
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(test_rows.iterrows()):
        manual_value = pd.to_numeric(row.get(manual_prob_col), errors="coerce") if manual_prob_col else np.nan
        final_value = pd.to_numeric(row.get(final_prob_col), errors="coerce") if final_prob_col else np.nan
        record: dict[str, Any] = {
            "game_id": row.get("game_id"),
            "game_date": str(row.get("game_date"))[:10] if "game_date" in row and pd.notna(row.get("game_date")) else None,
            "y_true": int(y_test[position]),
            "y_prob": float(y_prob[position]),
            "y_pred": int(y_pred[position]),
            "manual_prob_home": float(manual_value) if pd.notna(manual_value) else None,
            "final_prob_home": float(final_value) if pd.notna(final_value) else None,
            "model_eval_source": "temporary_time_ordered_logistic_eval",
        }
        for column in metadata_columns:
            if column not in record:
                record[column] = row.get(column) if column in test_rows.columns else None
        records.append(record)

    OOS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OOS_CSV_PATH, index=False, encoding="utf-8")
    write_json_report(report)
    write_collapse_report(report, collapse_guardrail)
    return report


if __name__ == "__main__":
    generate_report()
