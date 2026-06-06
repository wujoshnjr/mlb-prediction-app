from __future__ import annotations

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
OUTPUT_PATH = REPORT_DIR / "lineup_starter_slice_report.json"

MIN_SLICE_SAMPLE = 10


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


def _normalise_lineup_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "confirmed" in text:
        return "confirmed"
    if "projected" in text:
        return "projected"
    if "pending" in text:
        return "pending"
    return "unknown"


def _normalise_starter_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "confirmed" in text:
        return "confirmed"
    if "probable" in text or "projected" in text:
        return "probable"
    return "unknown"


def _normalise_grade(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"A", "B", "C", "D", "F"} else "unknown"


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

    merged = snap.merge(
        final[["game_id", "home_win"]].drop_duplicates("game_id"),
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
                    "model_prob",
                    "predicted_home_win_pct",
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
        outcome = _to_float(row.get("home_win"))

        if model_prob is None or outcome is None:
            continue

        outcome_int = int(outcome)
        if outcome_int not in (0, 1):
            continue

        lineup_raw = _first_existing(
            row,
            [
                "lineup_status",
                "home_projected_lineup_status",
                "away_projected_lineup_status",
                "projected_lineup_status",
            ],
        )
        starter_raw = _first_existing(
            row,
            [
                "starter_status",
                "starter_confidence_status",
                "home_starter_status",
                "away_starter_status",
                "probable_starter_status",
            ],
        )
        grade_raw = _first_existing(
            row,
            [
                "data_quality_grade",
                "data_quality_status_grade",
                "quality_grade",
            ],
        )

        clv = _to_float(_first_existing(row, ["clv", "clv_home_moneyline", "clv_away_moneyline"]))

        live_candidate_raw = str(row.get("live_bet_candidate", "") or "").strip().lower()
        live_candidate = live_candidate_raw in {"true", "1", "yes"}

        records.append(
            {
                "game_id": str(row.get("game_id")),
                "lineup_status": _normalise_lineup_status(lineup_raw),
                "starter_status": _normalise_starter_status(starter_raw),
                "data_quality_grade": _normalise_grade(grade_raw),
                "model_prob": model_prob,
                "market_prob": market_prob,
                "outcome": outcome_int,
                "clv": clv,
                "live_bet_candidate": live_candidate,
            }
        )

    return pd.DataFrame(records)


def _slice_metrics(frame: pd.DataFrame, dimension: str, value: str) -> Dict[str, Any]:
    group = frame[frame[dimension] == value] if dimension in frame.columns else pd.DataFrame()
    count = int(len(group))
    settled_count = count

    reasons: List[str] = []
    if count < MIN_SLICE_SAMPLE:
        reasons.append("insufficient_slice_sample")
    reasons.append("live_betting_disabled")

    if count == 0:
        return {
            "slice_type": dimension,
            "slice": value,
            "count": 0,
            "settled_count": 0,
            "model_brier": None,
            "market_brier": None,
            "model_logloss": None,
            "market_logloss": None,
            "avg_clv": None,
            "positive_clv_rate": None,
            "live_bet_candidate_count": 0,
            "block_live_bet": True,
            "block_reasons": reasons,
        }

    model_probs = group["model_prob"].astype(float).tolist()
    outcomes = group["outcome"].astype(int).tolist()

    market_group = group.dropna(subset=["market_prob"])
    market_probs = market_group["market_prob"].astype(float).tolist()
    market_outcomes = market_group["outcome"].astype(int).tolist()

    clv_values = pd.to_numeric(group["clv"], errors="coerce").dropna()

    return {
        "slice_type": dimension,
        "slice": value,
        "count": count,
        "settled_count": settled_count,
        "model_brier": round(_brier(model_probs, outcomes), 6) if model_probs else None,
        "market_brier": (
            round(_brier(market_probs, market_outcomes), 6)
            if market_probs
            else None
        ),
        "model_logloss": round(_logloss(model_probs, outcomes), 6) if model_probs else None,
        "market_logloss": (
            round(_logloss(market_probs, market_outcomes), 6)
            if market_probs
            else None
        ),
        "avg_clv": round(float(clv_values.mean()), 6) if not clv_values.empty else None,
        "positive_clv_rate": (
            round(float((clv_values > 0).mean()), 4)
            if not clv_values.empty
            else None
        ),
        "live_bet_candidate_count": int(group["live_bet_candidate"].sum()),
        "block_live_bet": True,
        "block_reasons": reasons,
    }


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots, snapshot_status = _read_csv(SNAPSHOT_PATH)
    finalized, finalized_status = _read_csv(FINALIZED_PATH)

    input_files = {
        "prediction_snapshots": snapshot_status,
        "finalized_games": finalized_status,
    }

    errors: List[str] = []
    warnings: List[str] = []

    rows = _prepare_rows(snapshots, finalized)

    dimensions = {
        "lineup_status": ["confirmed", "projected", "pending", "unknown"],
        "starter_status": ["confirmed", "probable", "unknown"],
        "data_quality_grade": ["A", "B", "C", "D", "F", "unknown"],
    }

    slices: List[Dict[str, Any]] = []
    for dimension, values in dimensions.items():
        for value in values:
            slices.append(_slice_metrics(rows, dimension, value))

    status = "ok" if not rows.empty else "insufficient_samples"
    if snapshots is None or finalized is None:
        status = "partial"
        warnings.append("Missing one or more input files.")

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": input_files,
        "evaluated_rows": int(len(rows)),
        "slices": slices,
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Use these slices for research only; live betting remains blocked."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
