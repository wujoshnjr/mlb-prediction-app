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
OUTPUT_JSON = REPORT_DIR / "shadow_feature_ablation_report.json"
OUTPUT_CSV = REPORT_DIR / "shadow_feature_ablation_rows.csv"

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
REQUIRED_BRIER_IMPROVEMENT = -0.005
REQUIRED_LOGLOSS_IMPROVEMENT = -0.005


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


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
        return {"count": 0, "brier": None, "logloss": None, "accuracy": None, "balanced_accuracy": None}
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    labels = (p >= 0.5).astype(int)
    recalls: list[float] = []
    for label in (0, 1):
        mask = y == label
        if bool(np.any(mask)):
            recalls.append(float(np.mean(labels[mask] == y[mask])))
    pos_rate = float(np.mean(labels == 1))
    return {
        "count": int(len(p)),
        "brier": safe_round(float(np.mean((p - y) ** 2))),
        "logloss": safe_round(float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))),
        "accuracy": safe_round(float(np.mean(labels == y))),
        "balanced_accuracy": safe_round(float(np.mean(recalls)) if recalls else None),
        "probability_mean": safe_round(float(np.mean(p))),
        "probability_std": safe_round(float(np.std(p))),
        "positive_rate": safe_round(pos_rate),
        "negative_rate": safe_round(1.0 - pos_rate),
    }


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
    sort_key = "game_id" if "game_id" in frame.columns else "home_win"
    frame = frame.sort_values(["_snapshot_dt", sort_key]).reset_index(drop=True)
    status["valid_labeled_rows"] = int(len(frame))
    return frame, status


def available_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    available: list[str] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) > 0:
            available.append(feature)
    return available


def time_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(frame))
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for test_idx in np.array_split(indices, TARGET_FOLD_COUNT):
        if len(test_idx) < MIN_TEST_ROWS:
            continue
        train_idx = indices[indices < int(test_idx[0])]
        if len(train_idx) >= MIN_TRAIN_ROWS:
            output.append((train_idx, test_idx))
    return output


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


def evaluate(frame: pd.DataFrame, name: str, features: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_prob: list[float] = []
    all_y: list[int] = []
    folds: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for fold_id, (train_idx, test_idx) in enumerate(time_folds(frame), start=1):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        probabilities, status = fit_predict(train, test, features)
        outcomes = [int(value) for value in test["home_win"].tolist()]
        if status != "ok":
            errors.append(f"fold_{fold_id}:{status}")
            folds.append({"fold_id": fold_id, "status": status, "train_rows": int(len(train)), "test_rows": int(len(test))})
            continue
        all_prob.extend(probabilities)
        all_y.extend(outcomes)
        folds.append({"fold_id": fold_id, "status": "ok", "train_rows": int(len(train)), "test_rows": int(len(test)), "metrics": metric_pack(probabilities, outcomes)})
        for row, probability, outcome in zip(test.to_dict("records"), probabilities, outcomes):
            csv_rows.append(
                {
                    "experiment": name,
                    "fold_id": fold_id,
                    "game_id": row.get("game_id"),
                    "game_date": row.get("game_date"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "research_home_prob": probability,
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
    }, csv_rows


def delta(candidate: dict[str, Any], core: dict[str, Any], metric: str) -> float | None:
    c_value = (candidate.get("metrics") or {}).get(metric)
    b_value = (core.get("metrics") or {}).get(metric)
    if c_value is None or b_value is None:
        return None
    return safe_round(float(c_value) - float(b_value))


def decision_from_deltas(brier_delta: float | None, logloss_delta: float | None) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if brier_delta is None:
        blockers.append("missing_brier_delta")
    elif brier_delta > REQUIRED_BRIER_IMPROVEMENT:
        blockers.append("brier_not_enough_improvement")
    if logloss_delta is None:
        blockers.append("missing_logloss_delta")
    elif logloss_delta > REQUIRED_LOGLOSS_IMPROVEMENT:
        blockers.append("logloss_not_enough_improvement")
    return ("keep_for_repeat_shadow_test", []) if not blockers else ("reject_for_now", blockers)


def candidate_features_by_group(candidate_report: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    features: list[str] = []
    groups: dict[str, list[str]] = {}
    for item in candidate_report.get("recommended_shadow_set") or []:
        if not isinstance(item, dict) or not item.get("feature"):
            continue
        feature = str(item["feature"])
        group = str(item.get("feature_group") or "ungrouped")
        features.append(feature)
        groups.setdefault(group, []).append(feature)
    return list(dict.fromkeys(features)), {key: list(dict.fromkeys(value)) for key, value in groups.items()}


def build_report() -> dict[str, Any]:
    candidate_report = read_json(CANDIDATE_REPORT_PATH)
    frame, sample_status = prepare_frame()
    candidate_features, grouped_features = candidate_features_by_group(candidate_report)
    core_features = available_features(frame, CORE_FEATURES)
    candidate_features = available_features(frame, candidate_features)
    grouped_features = {group: available_features(frame, features) for group, features in grouped_features.items()}
    grouped_features = {group: features for group, features in grouped_features.items() if features}

    all_rows: list[dict[str, Any]] = []
    core_summary, rows = evaluate(frame, "core_only", core_features)
    all_rows.extend(rows)

    feature_results: list[dict[str, Any]] = []
    for feature in candidate_features:
        summary, rows = evaluate(frame, f"core_plus__{feature}", list(dict.fromkeys(core_features + [feature])))
        all_rows.extend(rows)
        brier_delta = delta(summary, core_summary, "brier")
        logloss_delta = delta(summary, core_summary, "logloss")
        decision, blockers = decision_from_deltas(brier_delta, logloss_delta)
        feature_results.append(
            {
                "feature": feature,
                "experiment": summary["experiment"],
                "decision": decision,
                "blockers": blockers,
                "sample_count": summary["sample_count"],
                "metrics": summary["metrics"],
                "deltas_vs_core": {
                    "brier_delta_vs_core": brier_delta,
                    "logloss_delta_vs_core": logloss_delta,
                    "accuracy_delta_vs_core": delta(summary, core_summary, "accuracy"),
                    "balanced_accuracy_delta_vs_core": delta(summary, core_summary, "balanced_accuracy"),
                },
            }
        )

    group_results: list[dict[str, Any]] = []
    for group, features in grouped_features.items():
        summary, rows = evaluate(frame, f"core_plus_group__{group}", list(dict.fromkeys(core_features + features)))
        all_rows.extend(rows)
        brier_delta = delta(summary, core_summary, "brier")
        logloss_delta = delta(summary, core_summary, "logloss")
        decision, blockers = decision_from_deltas(brier_delta, logloss_delta)
        group_results.append(
            {
                "feature_group": group,
                "features": features,
                "experiment": summary["experiment"],
                "decision": decision,
                "blockers": blockers,
                "sample_count": summary["sample_count"],
                "metrics": summary["metrics"],
                "deltas_vs_core": {
                    "brier_delta_vs_core": brier_delta,
                    "logloss_delta_vs_core": logloss_delta,
                    "accuracy_delta_vs_core": delta(summary, core_summary, "accuracy"),
                    "balanced_accuracy_delta_vs_core": delta(summary, core_summary, "balanced_accuracy"),
                },
            }
        )

    feature_results.sort(key=lambda item: ((item["deltas_vs_core"].get("brier_delta_vs_core") is None), item["deltas_vs_core"].get("brier_delta_vs_core") or 999, item["feature"]))
    group_results.sort(key=lambda item: ((item["deltas_vs_core"].get("brier_delta_vs_core") is None), item["deltas_vs_core"].get("brier_delta_vs_core") or 999, item["feature_group"]))
    kept_features = [item for item in feature_results if item["decision"] == "keep_for_repeat_shadow_test"]
    kept_groups = [item for item in group_results if item["decision"] == "keep_for_repeat_shadow_test"]

    report = {
        "generated_at": utc_now(),
        "report_type": "shadow_feature_ablation_report",
        "status": "warning" if kept_features or kept_groups else "blocked",
        "training_samples": sample_status,
        "candidate_source": str(CANDIDATE_REPORT_PATH),
        "policy": {
            "active_model_promotion_allowed": False,
            "public_prediction_change_allowed": False,
            "required_brier_delta_vs_core": REQUIRED_BRIER_IMPROVEMENT,
            "required_logloss_delta_vs_core": REQUIRED_LOGLOSS_IMPROVEMENT,
            "reason": "Each shadow feature or group must improve Brier and logloss versus core_only before repeated shadow testing.",
        },
        "core_baseline": core_summary,
        "summary": {
            "candidate_feature_count": len(candidate_features),
            "candidate_group_count": len(group_results),
            "kept_feature_count": len(kept_features),
            "kept_group_count": len(kept_groups),
            "rejected_feature_count": len(feature_results) - len(kept_features),
            "rejected_group_count": len(group_results) - len(kept_groups),
        },
        "feature_ablation_results": feature_results,
        "group_ablation_results": group_results,
        "next_actions": [
            "Reject any feature or group that worsens Brier/logloss versus core_only.",
            "Repeat only the kept candidates in future shadow reruns; do not change public outputs.",
            "If all candidates are rejected, prioritize data backfill and source freshness before more model changes.",
        ],
        "promotion_allowed": False,
        "deployment_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_rows(all_rows)
    return report


def write_rows(rows: list[dict[str, Any]]) -> None:
    columns = ["experiment", "fold_id", "game_id", "game_date", "home_team", "away_team", "research_home_prob", "outcome"]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def main() -> int:
    report = build_report()
    print(json.dumps({"status": report["status"], "candidate_feature_count": report["summary"]["candidate_feature_count"], "output_path": str(OUTPUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
