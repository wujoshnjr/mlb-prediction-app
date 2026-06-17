from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    SimpleImputer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    SKLEARN_ERROR = str(exc)

try:
    from scripts.feature_schema import CORE_MODEL_FEATURES, MODEL_FEATURE_VERSION, get_model_feature_schema_hash
except Exception:  # pragma: no cover
    from feature_schema import CORE_MODEL_FEATURES, MODEL_FEATURE_VERSION, get_model_feature_schema_hash  # type: ignore[no-redef]

DATA_PATH = Path("data/training_samples.csv")
OUTPUT_JSON = Path("report/core_model_repair_report.json")
OUTPUT_CSV = Path("report/core_model_repair_rows.csv")

TARGET_FOLD_COUNT = 5
MIN_TRAIN_ROWS = 80
MIN_TEST_ROWS = 20
EPSILON = 1e-12
PROMOTION_IMPROVEMENT = -0.005

LABEL_COLUMNS = ["home_win", "y_true", "label", "result_home_win"]
BASELINE_PROB_COLUMNS = [
    "market_no_vig_home_prob",
    "premarket_model_home_prob",
    "elo_home_prob",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    try:
        if hasattr(value, "item"):
            return json_safe(value.item())
    except Exception:
        pass
    return value if isinstance(value, str) else str(value)


def safe_round(value: Any, digits: int = 6) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return round(parsed, digits) if math.isfinite(parsed) else None


def find_label_column(frame: pd.DataFrame) -> str | None:
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            return column
    return None


def load_frame() -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    status = {"path": str(DATA_PATH), "exists": DATA_PATH.exists(), "rows": 0, "valid_labeled_rows": 0, "error": ""}
    if not DATA_PATH.exists():
        status["error"] = "training_samples_missing"
        return pd.DataFrame(), status, None
    try:
        frame = pd.read_csv(DATA_PATH)
    except Exception as exc:
        status["error"] = f"read_failed:{exc}"
        return pd.DataFrame(), status, None
    status["rows"] = int(len(frame))
    label_column = find_label_column(frame)
    if label_column is None:
        status["error"] = "label_column_missing"
        return pd.DataFrame(), status, None
    frame = frame.copy()
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame[frame[label_column].isin([0, 1])].copy()
    frame[label_column] = frame[label_column].astype(int)
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str).str.strip()
        frame = frame[frame["game_id"] != ""].drop_duplicates("game_id", keep="first").copy()
    if "snapshot_created_at" in frame.columns:
        frame["_sort_dt"] = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    elif "game_date" in frame.columns:
        frame["_sort_dt"] = pd.to_datetime(frame["game_date"], errors="coerce", utc=True)
    else:
        frame["_sort_dt"] = pd.NaT
    sort_cols = ["_sort_dt"]
    if "game_id" in frame.columns:
        sort_cols.append("game_id")
    frame = frame.sort_values(sort_cols).reset_index(drop=True)
    status["valid_labeled_rows"] = int(len(frame))
    return frame, status, label_column


def time_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(frame))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_idx in np.array_split(indices, TARGET_FOLD_COUNT):
        if len(test_idx) < MIN_TEST_ROWS:
            continue
        train_idx = indices[indices < int(test_idx[0])]
        if len(train_idx) >= MIN_TRAIN_ROWS:
            folds.append((train_idx, test_idx))
    return folds


def metric_pack(probabilities: list[float], outcomes: list[int]) -> dict[str, Any]:
    if not probabilities:
        return {"count": 0, "brier": None, "logloss": None, "accuracy": None, "balanced_accuracy": None}
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    labels = (clipped >= 0.5).astype(int)
    recalls: list[float] = []
    for label in (0, 1):
        mask = y == label
        if bool(np.any(mask)):
            recalls.append(float(np.mean(labels[mask] == y[mask])))
    positive_rate = float(np.mean(labels == 1))
    return {
        "count": int(len(p)),
        "brier": safe_round(float(np.mean((clipped - y) ** 2))),
        "logloss": safe_round(float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))),
        "accuracy": safe_round(float(np.mean(labels == y))),
        "balanced_accuracy": safe_round(float(np.mean(recalls)) if recalls else None),
        "probability_mean": safe_round(float(np.mean(clipped))),
        "probability_std": safe_round(float(np.std(clipped))),
        "predicted_positive_rate": safe_round(positive_rate),
        "predicted_negative_rate": safe_round(1.0 - positive_rate),
        "probability_min": safe_round(float(np.min(clipped))),
        "probability_max": safe_round(float(np.max(clipped))),
    }


def core_features_available(frame: pd.DataFrame) -> list[str]:
    features = []
    for feature in CORE_MODEL_FEATURES:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) > 0 and float(np.nanvar(values.to_numpy(dtype=float))) > 1e-12:
            features.append(feature)
    return features


def fit_logistic(train: pd.DataFrame, test: pd.DataFrame, label_column: str, features: list[str], *, c_value: float, class_weight: str | None) -> tuple[list[float], str]:
    if SKLEARN_ERROR is not None:
        return [], f"sklearn_unavailable:{SKLEARN_ERROR}"
    if not features:
        return [], "no_features"
    y_train = train[label_column].astype(int)
    if y_train.nunique() < 2:
        return [], "single_class_train"
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=3000,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )
    try:
        model.fit(train[features].apply(pd.to_numeric, errors="coerce"), y_train)
        probabilities = model.predict_proba(test[features].apply(pd.to_numeric, errors="coerce"))[:, 1]
    except Exception as exc:
        return [], f"fit_failed:{exc}"
    return [float(max(EPSILON, min(1.0 - EPSILON, value))) for value in probabilities], "ok"


def evaluate(
    frame: pd.DataFrame,
    label_column: str,
    name: str,
    predictor: Callable[[pd.DataFrame, pd.DataFrame], tuple[list[float], str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_prob: list[float] = []
    all_y: list[int] = []
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    errors: list[str] = []
    for fold_id, (train_idx, test_idx) in enumerate(time_folds(frame), start=1):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        probabilities, status = predictor(train, test)
        outcomes = [int(value) for value in test[label_column].tolist()]
        if status != "ok":
            errors.append(f"fold_{fold_id}:{status}")
            folds.append({"fold_id": fold_id, "status": status, "train_rows": int(len(train)), "test_rows": int(len(test))})
            continue
        all_prob.extend(probabilities)
        all_y.extend(outcomes)
        folds.append({"fold_id": fold_id, "status": "ok", "train_rows": int(len(train)), "test_rows": int(len(test)), "metrics": metric_pack(probabilities, outcomes)})
        for source, probability, outcome in zip(test.to_dict("records"), probabilities, outcomes):
            rows.append(
                {
                    "candidate": name,
                    "fold_id": fold_id,
                    "game_id": source.get("game_id"),
                    "game_date": source.get("game_date"),
                    "home_team": source.get("home_team"),
                    "away_team": source.get("away_team"),
                    "research_home_win_probability": probability,
                    "outcome": outcome,
                }
            )
    return {
        "candidate": name,
        "status": "ok" if all_prob else "failed",
        "sample_count": len(all_prob),
        "successful_fold_count": int(sum(1 for fold in folds if fold.get("status") == "ok")),
        "metrics": metric_pack(all_prob, all_y),
        "folds": folds,
        "errors": errors,
    }, rows


def delta(candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> float | None:
    c_value = (candidate.get("metrics") or {}).get(metric)
    b_value = (baseline.get("metrics") or {}).get(metric)
    if c_value is None or b_value is None:
        return None
    return safe_round(float(c_value) - float(b_value))


def collapse_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    pos_rate = metrics.get("predicted_positive_rate")
    neg_rate = metrics.get("predicted_negative_rate")
    prob_std = metrics.get("probability_std")
    if pos_rate is not None and float(pos_rate) >= 0.85:
        reasons.append("single_class_positive_prediction_collapse")
    if neg_rate is not None and float(neg_rate) >= 0.85:
        reasons.append("single_class_negative_prediction_collapse")
    if prob_std is not None and float(prob_std) < 0.02:
        reasons.append("probability_distribution_too_narrow")
    return reasons


def attach_decision(candidate: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deltas = {
        baseline_name: {
            "brier_delta": delta(candidate, baseline, "brier"),
            "logloss_delta": delta(candidate, baseline, "logloss"),
            "accuracy_delta": delta(candidate, baseline, "accuracy"),
        }
        for baseline_name, baseline in baselines.items()
    }
    blockers: list[str] = []
    for baseline_name in ("constant_50", "expanding_train_home_rate"):
        item = deltas.get(baseline_name) or {}
        brier_delta = item.get("brier_delta")
        logloss_delta = item.get("logloss_delta")
        if brier_delta is None or brier_delta > PROMOTION_IMPROVEMENT:
            blockers.append(f"does_not_improve_brier_vs_{baseline_name}")
        if logloss_delta is None or logloss_delta > PROMOTION_IMPROVEMENT:
            blockers.append(f"does_not_improve_logloss_vs_{baseline_name}")
    blockers.extend(collapse_reasons(candidate.get("metrics") or {}))
    candidate["deltas_vs_baselines"] = deltas
    candidate["decision"] = "keep_for_repeated_research" if not blockers else "reject_for_active_model"
    candidate["blockers"] = sorted(set(blockers))
    candidate["promotion_allowed"] = False
    candidate["live_betting_allowed"] = False
    candidate["automated_wagering_allowed"] = False
    return candidate


def write_rows(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = ["candidate", "fold_id", "game_id", "game_date", "home_team", "away_team", "research_home_win_probability", "outcome"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def build_report() -> dict[str, Any]:
    frame, sample_status, label_column = load_frame()
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "report_type": "core_model_repair_report",
        "status": "skipped",
        "training_samples": sample_status,
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "core_features": list(CORE_MODEL_FEATURES),
        "candidate_results": [],
        "best_candidate": None,
        "recommendations": [],
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }
    if frame.empty or label_column is None:
        report["status"] = "skipped"
        report["recommendations"].append("Training samples unavailable; do not change active model.")
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        write_rows([])
        return report

    features = core_features_available(frame)
    report["available_core_features"] = features
    all_rows: list[dict[str, Any]] = []

    def constant_50(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[float], str]:
        return [0.5] * len(test), "ok"

    def expanding_home_rate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[float], str]:
        rate = float(train[label_column].mean()) if len(train) else 0.5
        return [max(EPSILON, min(1.0 - EPSILON, rate))] * len(test), "ok"

    candidate_specs: list[tuple[str, Callable[[pd.DataFrame, pd.DataFrame], tuple[list[float], str]]]] = [
        ("constant_50", constant_50),
        ("expanding_train_home_rate", expanding_home_rate),
    ]

    for column in BASELINE_PROB_COLUMNS:
        if column in frame.columns:
            def make_column_predictor(prob_column: str) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[list[float], str]]:
                def predictor(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[float], str]:
                    values = pd.to_numeric(test[prob_column], errors="coerce").fillna(float(train[label_column].mean()) if len(train) else 0.5)
                    return [float(max(EPSILON, min(1.0 - EPSILON, value))) for value in values.tolist()], "ok"
                return predictor
            candidate_specs.append((f"column__{column}", make_column_predictor(column)))

    for c_value in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        for class_weight in (None, "balanced"):
            class_label = "balanced" if class_weight else "plain"
            name = f"logistic_core_c{c_value:g}_{class_label}"
            def make_logistic_predictor(c: float, weight: str | None) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[list[float], str]]:
                return lambda train, test: fit_logistic(train, test, label_column, features, c_value=c, class_weight=weight)
            candidate_specs.append((name, make_logistic_predictor(c_value, class_weight)))

    raw_results: list[dict[str, Any]] = []
    for name, predictor in candidate_specs:
        summary, rows = evaluate(frame, label_column, name, predictor)
        raw_results.append(summary)
        all_rows.extend(rows)

    baselines = {item["candidate"]: item for item in raw_results if item["candidate"] in {"constant_50", "expanding_train_home_rate"}}
    candidate_results = [attach_decision(dict(item), baselines) for item in raw_results]
    candidate_results.sort(key=lambda item: ((item.get("metrics") or {}).get("brier") is None, (item.get("metrics") or {}).get("brier") or 999, str(item.get("candidate")))) )
    eligible = [item for item in candidate_results if item.get("decision") == "keep_for_repeated_research"]
    best = eligible[0] if eligible else candidate_results[0] if candidate_results else None

    report["status"] = "warning" if eligible else "blocked"
    report["candidate_results"] = candidate_results
    report["best_candidate"] = best
    report["summary"] = {
        "candidate_count": len(candidate_results),
        "eligible_research_candidate_count": len(eligible),
        "fold_count": len(time_folds(frame)),
        "oos_prediction_count": int(max((item.get("sample_count") or 0) for item in candidate_results) if candidate_results else 0),
    }
    report["recommendations"] = [
        "Do not replace the active model unless a candidate repeatedly beats constant_50 and expanding_train_home_rate on Brier and logloss.",
        "If all candidates remain blocked, prioritize more settled pregame snapshots before adding model complexity.",
        "Use this report for research only; live betting and automated wagering remain disabled.",
    ]
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_rows(all_rows)
    return report


def main() -> int:
    report = build_report()
    summary = report.get("summary") or {}
    print(json.dumps({"status": report.get("status"), "candidate_count": summary.get("candidate_count"), "best_candidate": (report.get("best_candidate") or {}).get("candidate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
