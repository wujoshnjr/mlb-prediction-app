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


def write_json_report(report: dict[str, Any], path: Path = REPORT_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            clean_json_value(report),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def base_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "report_type": "model_eval_report",
        "status": "skipped",
        "training_source": str(DATA_PATH),
        "oos_prediction_output": str(OOS_CSV_PATH),
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
        work["_eval_sort_time"] = pd.to_datetime(
            work["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
    elif "game_date" in work.columns:
        work["_eval_sort_time"] = pd.to_datetime(
            work["game_date"],
            errors="coerce",
            utc=True,
        )
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
        return skip_report(
            report,
            f"insufficient_samples:{sample_count}<{MIN_TOTAL_SAMPLES}",
        )

    for feature in CORE_MODEL_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    matrix = frame[CORE_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(
        dtype=float
    )
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

    if (
        train_count < MIN_TRAIN_SAMPLES
        or calibration_count < MIN_CALIBRATION_SAMPLES
        or test_count < MIN_TEST_SAMPLES
    ):
        return skip_report(report, "not_enough_samples_after_train_calibration_test_split")

    x_train = matrix[:train_end]
    y_train = target[:train_end]

    x_calib = matrix[train_end:calibration_end]
    y_calib = target[train_end:calibration_end]

    x_test = matrix[calibration_end:]
    y_test = target[calibration_end:]

    for name, subset in (
        ("train", y_train),
        ("calibration", y_calib),
        ("test", y_test),
    ):
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

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "brier": float(brier_score_loss(y_test, y_prob)),
        "logloss": float(log_loss(y_test, y_prob, labels=[0, 1])),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    report["status"] = "ok"
    report["metrics"] = metrics
    report["confusion_matrix"] = cm

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
        record: dict[str, Any] = {
            "game_id": row.get("game_id"),
            "game_date": (
                str(row.get("game_date"))[:10]
                if "game_date" in row and pd.notna(row.get("game_date"))
                else None
            ),
            "y_true": int(y_test[position]),
            "y_prob": float(y_prob[position]),
            "y_pred": int(y_pred[position]),
            "manual_prob_home": (
                float(row.get(manual_prob_col))
                if manual_prob_col
                and pd.notna(pd.to_numeric(row.get(manual_prob_col), errors="coerce"))
                else None
            ),
            "final_prob_home": (
                float(row.get(final_prob_col))
                if final_prob_col
                and pd.notna(pd.to_numeric(row.get(final_prob_col), errors="coerce"))
                else None
            ),
            "model_eval_source": "temporary_time_ordered_logistic_eval",
        }

        for column in metadata_columns:
            if column not in record:
                record[column] = row.get(column) if column in test_rows.columns else None

        records.append(record)

    OOS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OOS_CSV_PATH, index=False, encoding="utf-8")

    write_json_report(report)
    return report


if __name__ == "__main__":
    generate_report()
