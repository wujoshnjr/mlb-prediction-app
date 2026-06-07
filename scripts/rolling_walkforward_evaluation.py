from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
MARKET_ODDS_PATH = DATA_DIR / "market_odds_history.csv"

OUTPUT_JSON = REPORT_DIR / "rolling_walkforward_evaluation.json"
OUTPUT_CSV = REPORT_DIR / "rolling_walkforward_predictions.csv"

MIN_OOS_PREDICTIONS = 300

CSV_COLUMNS = [
    "fold_id",
    "game_id",
    "game_date",
    "prediction_created_at",
    "model_prob",
    "market_prob",
    "outcome",
    "clv",
    "status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
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


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _clip_prob(value: float, eps: float = 1e-7) -> float:
    return max(eps, min(1.0 - eps, float(value)))


def _brier(probs: List[float], outcomes: List[int]) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def _logloss(probs: List[float], outcomes: List[int]) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None
    total = 0.0
    for prob, outcome in zip(probs, outcomes):
        p = _clip_prob(prob)
        total += -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))
    return total / len(probs)


def _first_existing(row: pd.Series, columns: List[str]) -> Optional[Any]:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value):
                return value
    return None


def _prepare_rows(
    snapshots: Optional[pd.DataFrame],
    finalized: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if snapshots is None or finalized is None:
        return pd.DataFrame()

    if snapshots.empty or finalized.empty:
        return pd.DataFrame()

    if "game_id" not in snapshots.columns or "game_id" not in finalized.columns:
        return pd.DataFrame()

    snap = snapshots.copy()
    final = finalized.copy()

    snap["game_id"] = snap["game_id"].astype(str)
    final["game_id"] = final["game_id"].astype(str)

    if "snapshot_created_at" in snap.columns:
        snap["_snapshot_dt"] = pd.to_datetime(snap["snapshot_created_at"], errors="coerce", utc=True)
        snap = snap.sort_values("_snapshot_dt").groupby("game_id", as_index=False).tail(1)

    if "home_win" not in final.columns:
        if {"home_score", "away_score"}.issubset(final.columns):
            final["home_win"] = (
                pd.to_numeric(final["home_score"], errors="coerce")
                > pd.to_numeric(final["away_score"], errors="coerce")
            ).astype("Int64")
        else:
            return pd.DataFrame()

    final_cols = ["game_id", "home_win"]
    if "game_date" in final.columns:
        final_cols.append("game_date")

    merged = snap.merge(
        final[final_cols].drop_duplicates("game_id"),
        on="game_id",
        how="inner",
        suffixes=("", "_final"),
    )

    records: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        model_prob = _to_float(
            _first_existing(
                row,
                [
                    "displayed_home_win_pct",
                    "predicted_home_win_pct",
                    "model_prob",
                    "premarket_model_home_prob",
                    "home_win_probability",
                ],
            )
        )
        market_prob = _to_float(
            _first_existing(
                row,
                [
                    "market_no_vig_home_prob",
                    "market_prob",
                    "market_home_prob",
                ],
            )
        )
        outcome_value = _to_float(row.get("home_win"))
        if model_prob is None or outcome_value is None:
            continue

        outcome = int(outcome_value)
        if outcome not in (0, 1):
            continue

        game_date = _first_existing(row, ["game_date", "game_date_final", "start_time"])
        prediction_created_at = _first_existing(row, ["snapshot_created_at", "created_at"])

        clv = _to_float(_first_existing(row, ["clv", "clv_home_moneyline", "clv_away_moneyline"]))

        records.append(
            {
                "game_id": str(row.get("game_id")),
                "game_date": game_date,
                "prediction_created_at": prediction_created_at,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "outcome": outcome,
                "clv": clv,
                "status": "settled",
            }
        )

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["_sort_dt"] = pd.to_datetime(
            frame["game_date"].fillna(frame["prediction_created_at"]),
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values("_sort_dt")

    return frame


def _write_empty_csv() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots, snapshot_status = _read_csv(SNAPSHOT_PATH)
    finalized, finalized_status = _read_csv(FINALIZED_PATH)
    _, market_status = _read_csv(MARKET_ODDS_PATH)

    input_files = {
        "prediction_snapshots": snapshot_status,
        "finalized_games": finalized_status,
        "market_odds_history": market_status,
    }

    errors: List[str] = []
    warnings: List[str] = []

    rows = _prepare_rows(snapshots, finalized)

    if len(rows) < MIN_OOS_PREDICTIONS:
        _write_empty_csv()
        report = {
            "generated_at": _utc_now(),
            "status": "insufficient_samples",
            "input_files": input_files,
            "fold_count": 0,
            "total_oos_predictions": int(len(rows)),
            "model_brier": None,
            "market_brier": None,
            "model_logloss": None,
            "market_logloss": None,
            "model_beats_market_brier": None,
            "model_beats_market_logloss": None,
            "avg_clv": None,
            "positive_clv_rate": None,
            "min_required_oos_predictions": MIN_OOS_PREDICTIONS,
            "errors": errors,
            "warnings": warnings,
            "recommendations": [
                f"Need at least {MIN_OOS_PREDICTIONS} settled rows for rolling walk-forward evaluation."
            ],
        }
        OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return report

    rows = rows.copy()
    rows["fold_id"] = 1

    model_probs = rows["model_prob"].astype(float).tolist()
    outcomes = rows["outcome"].astype(int).tolist()

    market_valid = rows.dropna(subset=["market_prob"])
    market_probs = market_valid["market_prob"].astype(float).tolist()
    market_outcomes = market_valid["outcome"].astype(int).tolist()

    clv_values = pd.to_numeric(rows["clv"], errors="coerce").dropna()

    model_brier = _brier(model_probs, outcomes)
    model_logloss = _logloss(model_probs, outcomes)
    market_brier = _brier(market_probs, market_outcomes) if market_probs else None
    market_logloss = _logloss(market_probs, market_outcomes) if market_probs else None

    rows[CSV_COLUMNS].to_csv(OUTPUT_CSV, index=False)

    report = {
        "generated_at": _utc_now(),
        "status": "ok",
        "input_files": input_files,
        "fold_count": 1,
        "total_oos_predictions": int(len(rows)),
        "model_brier": round(model_brier, 6) if model_brier is not None else None,
        "market_brier": round(market_brier, 6) if market_brier is not None else None,
        "model_logloss": round(model_logloss, 6) if model_logloss is not None else None,
        "market_logloss": round(market_logloss, 6) if market_logloss is not None else None,
        "model_beats_market_brier": (
            model_brier < market_brier
            if model_brier is not None and market_brier is not None
            else None
        ),
        "model_beats_market_logloss": (
            model_logloss < market_logloss
            if model_logloss is not None and market_logloss is not None
            else None
        ),
        "avg_clv": round(float(clv_values.mean()), 6) if not clv_values.empty else None,
        "positive_clv_rate": (
            round(float((clv_values > 0).mean()), 4)
            if not clv_values.empty
            else None
        ),
        "min_required_oos_predictions": MIN_OOS_PREDICTIONS,
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "This is a rolling walk-forward proxy using saved snapshot probabilities; full retraining folds can be added later."
        ],
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
