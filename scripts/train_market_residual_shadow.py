from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.feature_schema import EXPECTED_FEATURES
from scripts.market_residual_model import (
    logit_to_probability,
    probability_to_logit,
)


MODEL_OUTPUT_PATH = Path("data/market_residual_model.pkl")
REPORT_OUTPUT_PATH = Path("report/market_residual_training_report.json")


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        return pd.read_csv(path), ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


def _clip_probability(probability: float) -> float:
    return float(np.clip(probability, 1e-7, 1.0 - 1e-7))


def _brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def _logloss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(
        -np.mean(
            outcomes * np.log(probabilities)
            + (1.0 - outcomes) * np.log(1.0 - probabilities)
        )
    )


def _expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    if len(probabilities) == 0:
        return 1.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    ece = 0.0

    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)

        count = int(mask.sum())
        if count == 0:
            continue

        avg_confidence = float(probabilities[mask].mean())
        avg_accuracy = float(outcomes[mask].mean())
        ece += (count / total) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def _prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()

    if "home_win" not in working.columns:
        return pd.DataFrame()

    working["home_win_numeric"] = pd.to_numeric(
        working["home_win"],
        errors="coerce",
    )

    working["market_no_vig_home_prob_numeric"] = pd.to_numeric(
        working.get("market_no_vig_home_prob"),
        errors="coerce",
    )

    working = working[
        working["home_win_numeric"].isin([0.0, 1.0])
        & working["market_no_vig_home_prob_numeric"].between(0.01, 0.99)
    ].copy()

    for feature in EXPECTED_FEATURES:
        if feature not in working.columns:
            working[feature] = 0.0
        working[feature] = pd.to_numeric(working[feature], errors="coerce").fillna(0.0)

    if "snapshot_created_at" in working.columns:
        working["sort_time"] = pd.to_datetime(
            working["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
    elif "game_date" in working.columns:
        working["sort_time"] = pd.to_datetime(
            working["game_date"],
            errors="coerce",
            utc=True,
        )
    else:
        working["sort_time"] = pd.NaT

    if working["sort_time"].notna().any():
        working = working.sort_values("sort_time")
    else:
        working = working.reset_index(drop=True)

    return working


def train_market_residual_shadow(
    snapshots_path: str = "data/prediction_snapshots.csv",
    model_output_path: str = str(MODEL_OUTPUT_PATH),
    report_output_path: str = str(REPORT_OUTPUT_PATH),
    min_samples: int = 120,
) -> Dict[str, Any]:
    generated_at = _current_utc_iso()
    frame, error = _safe_read_csv(Path(snapshots_path))

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "input_file": {
            "path": snapshots_path,
            "exists": frame is not None,
            "error": error,
            "rows": int(len(frame)) if frame is not None else 0,
        },
        "trained": False,
        "skipped": False,
        "skip_reason": "",
        "sample_count": 0,
        "train_count": 0,
        "validation_count": 0,
        "feature_count": len(EXPECTED_FEATURES),
        "market_brier": None,
        "residual_brier": None,
        "market_logloss": None,
        "residual_logloss": None,
        "market_ece": None,
        "residual_ece": None,
        "residual_beats_market_brier": False,
        "residual_beats_market_logloss": False,
        "residual_beats_market_ece": False,
        "promote_candidate": False,
        "promotion_blockers": [],
        "model_output_path": model_output_path,
    }

    if frame is None or frame.empty:
        report["skipped"] = True
        report["skip_reason"] = "missing_or_empty_snapshot_file"
        _write_report(report, report_output_path)
        return report

    prepared = _prepare_training_frame(frame)
    sample_count = int(len(prepared))
    report["sample_count"] = sample_count

    if sample_count < min_samples:
        report["skipped"] = True
        report["skip_reason"] = f"sample_count_below_min_samples_{min_samples}"
        _write_report(report, report_output_path)
        return report

    split_index = int(sample_count * 0.80)
    train = prepared.iloc[:split_index].copy()
    validation = prepared.iloc[split_index:].copy()

    report["train_count"] = int(len(train))
    report["validation_count"] = int(len(validation))

    x_train = train[EXPECTED_FEATURES].to_numpy(dtype=float)
    y_train = train["home_win_numeric"].to_numpy(dtype=float)
    p_train_market = train["market_no_vig_home_prob_numeric"].to_numpy(dtype=float)

    x_val = validation[EXPECTED_FEATURES].to_numpy(dtype=float)
    y_val = validation["home_win_numeric"].to_numpy(dtype=float)
    p_val_market = validation["market_no_vig_home_prob_numeric"].to_numpy(dtype=float)

    train_margin = np.array(
        [probability_to_logit(_clip_probability(prob)) for prob in p_train_market],
        dtype=float,
    )
    val_margin = np.array(
        [probability_to_logit(_clip_probability(prob)) for prob in p_val_market],
        dtype=float,
    )

    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        init_score=train_margin,
        feature_name=list(EXPECTED_FEATURES),
    )
    val_data = lgb.Dataset(
        x_val,
        label=y_val,
        init_score=val_margin,
        feature_name=list(EXPECTED_FEATURES),
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 10,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    delta_val = model.predict(x_val, raw_score=True)
    residual_probs = np.array(
        [
            logit_to_probability(probability_to_logit(market_prob) + delta)
            for market_prob, delta in zip(p_val_market, delta_val)
        ],
        dtype=float,
    )

    market_brier = _brier(p_val_market, y_val)
    residual_brier = _brier(residual_probs, y_val)
    market_logloss = _logloss(p_val_market, y_val)
    residual_logloss = _logloss(residual_probs, y_val)
    market_ece = _expected_calibration_error(p_val_market, y_val)
    residual_ece = _expected_calibration_error(residual_probs, y_val)

    report["trained"] = True
    report["market_brier"] = market_brier
    report["residual_brier"] = residual_brier
    report["market_logloss"] = market_logloss
    report["residual_logloss"] = residual_logloss
    report["market_ece"] = market_ece
    report["residual_ece"] = residual_ece
    report["residual_beats_market_brier"] = bool(residual_brier < market_brier)
    report["residual_beats_market_logloss"] = bool(residual_logloss < market_logloss)
    report["residual_beats_market_ece"] = bool(residual_ece < market_ece)

    blockers: List[str] = []
    if sample_count < 300:
        blockers.append("sample_count_below_300")
    if not report["residual_beats_market_brier"]:
        blockers.append("residual_brier_not_better_than_market")
    if not report["residual_beats_market_logloss"]:
        blockers.append("residual_logloss_not_better_than_market")
    if residual_ece > 0.03:
        blockers.append("residual_ece_above_0_03")

    report["promotion_blockers"] = blockers
    report["promote_candidate"] = len(blockers) == 0

    output_file = Path(model_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as file:
        pickle.dump(
            {
                "model_type": "lightgbm_market_residual_shadow",
                "model": model,
                "features": list(EXPECTED_FEATURES),
                "trained_at": generated_at,
                "sample_count": sample_count,
                "validation_metrics": {
                    "market_brier": market_brier,
                    "residual_brier": residual_brier,
                    "market_logloss": market_logloss,
                    "residual_logloss": residual_logloss,
                    "market_ece": market_ece,
                    "residual_ece": residual_ece,
                },
            },
            file,
        )

    _write_report(report, report_output_path)
    return report


def _write_report(report: Dict[str, Any], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    result = train_market_residual_shadow()
    print(
        json.dumps(
            {
                "generated_at": result["generated_at"],
                "trained": result["trained"],
                "skipped": result["skipped"],
                "skip_reason": result["skip_reason"],
                "sample_count": result["sample_count"],
                "market_brier": result["market_brier"],
                "residual_brier": result["residual_brier"],
                "market_logloss": result["market_logloss"],
                "residual_logloss": result["residual_logloss"],
                "promote_candidate": result["promote_candidate"],
                "promotion_blockers": result["promotion_blockers"],
                "report_written_to": str(REPORT_OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
