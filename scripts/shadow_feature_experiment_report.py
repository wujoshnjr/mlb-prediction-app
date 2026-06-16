from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORT_DIR = Path("report")
DATA_DIR = Path("data")
TRAINING_SAMPLES_PATH = DATA_DIR / "training_samples.csv"
CANDIDATE_REPORT_PATH = REPORT_DIR / "feature_promotion_candidate_report.json"
OUTPUT_JSON = REPORT_DIR / "shadow_feature_experiment_report.json"
OUTPUT_CSV = REPORT_DIR / "shadow_feature_experiment_rows.csv"

CORE_FEATURES = [
    "elo_diff",
    "bt_strength_diff",
    "sp_era_diff",
    "pitcher_rating_diff",
    "dynamic_park_factor",
    "winrate_diff",
    "timezone_diff",
]

TARGET_FOLD_COUNT = 5
MIN_TRAIN_ROWS = 80
MIN_TEST_ROWS = 20
EPSILON = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def metric_pack(probabilities: list[float], outcomes: list[int]) -> dict[str, Any]:
    if not probabilities:
        return {"count": 0}
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    pred = (p >= 0.5).astype(int)
    recalls: list[float] = []
    for label in (0, 1):
        mask = y == label
        if bool(np.any(mask)):
            recalls.append(float(np.mean(pred[mask] == y[mask])))
    pos_rate = float(np.mean(pred == 1))
    return {
        "count": int(len(p)),
        "brier": safe_round(float(np.mean((p - y) ** 2))),
        "logloss": safe_round(float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))),
        "accuracy": safe_round(float(np.mean(pred == y))),
        "balanced_accuracy": safe_round(float(np.mean(recalls)) if recalls else None),
        "probability_mean": safe_round(float(np.mean(p))),
        "probability_std": safe_round(float(np.std(p))),
        "positive_rate": safe_round(pos_rate),
        "negative_rate": safe_round(1.0 - pos_rate),
    }


def collect_candidates() -> tuple[list[str], list[str]]:
    report = read_json(CANDIDATE_REPORT_PATH)
    recommended: list[str] = []
    now: list[str] = []
    for item in report.get("recommended_shadow_set") or []:
        if not isinstance(item, dict) or not item.get("feature"):
            continue
        feature = str(item["feature"])
        recommended.append(feature)
        if item.get("promotion_status") == "shadow_candidate_now":
            now.append(feature)
    return list(dict.fromkeys(recommended)), list(dict.fromkeys(now))


def prepare_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    status = {"path": str(TRAINING_SAMPLES_PATH), "exists": TRAINING_SAMPLES_PATH.exists(), "rows": 0, "error": ""}
    if not TRAINING_SAMPLES_PATH.exists():
        status["error"] = "file_missing"
        return pd.DataFrame(), status
    try:
        frame = pd.read_csv(TRAINING_SAMPLES_PATH)
    except Exception as exc:
        status["error"] = str(exc)
        return pd.DataFrame(), status
    status["rows"] = int(len(frame))
    if "home_win" not in frame.columns:
        status["error"] = "home_win_missing"
        return pd.DataFrame(), status
    frame = frame.copy()
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame = frame[frame["home_win"].isin([0, 1])].copy()
    if "snapshot_created_at" in frame.columns:
        frame["_snapshot_dt"] = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    elif "game_date" in frame.columns:
        frame["_snapshot_dt"] = pd.to_datetime(frame["game_date"], errors="coerce", utc=True)
    else:
        frame["_snapshot_dt"] = pd.NaT
    frame = frame.sort_values(["_snapshot_dt", "game_id" if "game_id" in frame.columns else "home_win"]).reset_index(drop=True)
    status["valid_labeled_rows"] = int(len(frame))
    return frame, status


def available_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    output: list[str] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) > 0:
            output.append(feature)
    return output


def time_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    if frame.empty:
        return []
    indices = np.arange(len(frame))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_idx in np.array_split(indices, TARGET_FOLD_COUNT):
        if len(test_idx) < MIN_TEST_ROWS:
            continue
        train_idx = indices[indices < int(test_idx[0])]
        if len(train_idx) >= MIN_TRAIN_ROWS:
            folds.append((train_idx, test_idx))
    return folds


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[list[float], str]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return [], f"dependency_missing:{exc}"
    if not features:
        return [], "no_features"
    y = train["home_win"].astype(int)
    if y.nunique() < 2:
        return [], "single_class_train"
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    try:
        model.fit(train[features].apply(pd.to_numeric, errors="coerce"), y)
        probabilities = model.predict_proba(test[features].apply(pd.to_numeric, errors="coerce"))[:, 1]
    except Exception as exc:
        return [], f"fit_failed:{exc}"
    return [float(max(EPSILON, min(1.0 - EPSILON, value))) for value in probabilities], "ok"


def evaluate_set(frame: pd.DataFrame, name: str, features: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_prob: list[float] = []
    all_y: list[int] = []
    rows: list[dict[str, Any]] = []
    folds = []
    errors: list[str] = []
    for fold_id, (train_idx, test_idx) in enumerate(time_folds(frame), start=1):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        probs, status = fit_predict(train, test, features)
        y = [int(value) for value in test["home_win"].tolist()]
        if status != "ok":
            errors.append(f"fold_{fold_id}:{status}")
            folds.append({"fold_id": fold_id, "status": status, "train_rows": int(len(train)), "test_rows": int(len(test))})
            continue
        all_prob.extend(probs)
        all_y.extend(y)
        folds.append({"fold_id": fold_id, "status": "ok", "train_rows": int(len(train)), "test_rows": int(len(test)), "metrics": metric_pack(probs, y)})
        for source, prob, outcome in zip(test.to_dict("records"), probs, y):
            rows.append(
                {
                    "experiment": name,
                    "fold_id": fold_id,
                    "game_id": source.get("game_id"),
                    "game_date": source.get("game_date"),
                    "home_team": source.get("home_team"),
                    "away_team": source.get("away_team"),
                    "research_home_win_probability": prob,
                    "outcome": outcome,
                }
            )
    return {
        "experiment": name,
        "status": "ok" if all_prob else "failed",
        "feature_count": len(features),
        "features": features,
        "sample_count": len(all_prob),
        "successful_fold_count": int(sum(1 for fold in folds if fold.get("status") == "ok")),
        "metrics": metric_pack(all_prob, all_y),
        "folds": folds,
        "errors": errors,
    }, rows


def write_rows(rows: list[dict[str, Any]]) -> None:
    columns = ["experiment", "fold_id", "game_id", "game_date", "home_team", "away_team", "research_home_win_probability", "outcome"]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def build_report() -> dict[str, Any]:
    frame, sample_status = prepare_frame()
    recommended, now = collect_candidates()
    core = available_features(frame, CORE_FEATURES)
    now_available = available_features(frame, now)
    recommended_available = available_features(frame, recommended)
    sets = [
        ("core_only", core),
        ("core_plus_shadow_now", list(dict.fromkeys(core + now_available))),
        ("core_plus_recommended_shadow_set", list(dict.fromkeys(core + recommended_available))),
    ]
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for name, features in sets:
        summary, rows = evaluate_set(frame, name, features)
        summaries.append(summary)
        all_rows.extend(rows)
    by_name = {str(item["experiment"]): item for item in summaries}
    core_summary = by_name.get("core_only", {})
    comparisons = []
    for name in ["core_plus_shadow_now", "core_plus_recommended_shadow_set"]:
        candidate = by_name.get(name, {})
        comparison = {"experiment": name}
        for metric in ["brier", "logloss", "accuracy", "balanced_accuracy"]:
            base = (core_summary.get("metrics") or {}).get(metric)
            value = (candidate.get("metrics") or {}).get(metric)
            comparison[f"{metric}_delta_vs_core"] = safe_round(float(value) - float(base)) if value is not None and base is not None else None
        comparisons.append(comparison)
    report = {
        "generated_at": utc_now(),
        "report_type": "shadow_feature_experiment_report",
        "status": "warning",
        "training_samples": sample_status,
        "candidate_source": str(CANDIDATE_REPORT_PATH),
        "feature_sets": {"core_only": core, "shadow_now": now_available, "recommended_shadow_set": recommended_available},
        "experiments": summaries,
        "comparisons_vs_core": comparisons,
        "recommendations": [
            "Research-only output. Do not change active predictions from this report.",
            "Only consider feature promotion after repeated reruns improve Brier/logloss versus core_only.",
            "Keep missing indicators in the imputation pipeline before any active-model promotion.",
        ],
        "promotion_allowed": False,
        "deployment_allowed": False,
        "public_prediction_change_allowed": False,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_rows(all_rows)
    return report


def main() -> int:
    report = build_report()
    print(json.dumps({"status": report["status"], "experiment_count": len(report["experiments"]), "output_path": str(OUTPUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
