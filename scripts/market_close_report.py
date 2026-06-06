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
ODDS_PATH = DATA_DIR / "market_odds_history.csv"
OUTPUT_PATH = REPORT_DIR / "market_close_report.json"


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


def _first_existing(row: pd.Series, columns: List[str]) -> Optional[Any]:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value):
                return value
    return None


def _suspicious_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    home_odds = _to_float(
        _first_existing(
            row,
            ["home_moneyline_odds", "home_odds", "home_closing_odds", "closing_home_odds"],
        )
    )
    away_odds = _to_float(
        _first_existing(
            row,
            ["away_moneyline_odds", "away_odds", "away_closing_odds", "closing_away_odds"],
        )
    )

    if home_odds is None:
        reasons.append("missing_home_odds")
    elif home_odds <= 1.0 or home_odds > 20.0:
        reasons.append(f"suspicious_home_odds:{home_odds}")

    if away_odds is None:
        reasons.append("missing_away_odds")
    elif away_odds <= 1.0 or away_odds > 20.0:
        reasons.append(f"suspicious_away_odds:{away_odds}")

    market_prob = _to_float(row.get("market_no_vig_home_prob"))
    if market_prob is not None and not (0.0 <= market_prob <= 1.0):
        reasons.append(f"market_no_vig_home_prob_out_of_range:{market_prob}")

    return reasons


def _latest_odds_by_game(odds: Optional[pd.DataFrame]) -> pd.DataFrame:
    if odds is None or odds.empty or "game_id" not in odds.columns:
        return pd.DataFrame()

    frame = odds.copy()
    frame["game_id"] = frame["game_id"].astype(str)

    time_col = None
    for candidate in ("odds_captured_at", "captured_at", "snapshot_created_at", "created_at"):
        if candidate in frame.columns:
            time_col = candidate
            break

    if time_col:
        frame["_odds_dt"] = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
        frame = frame.sort_values("_odds_dt").groupby("game_id", as_index=False).tail(1)
    else:
        frame = frame.groupby("game_id", as_index=False).tail(1)

    return frame


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots, snapshot_status = _read_csv(SNAPSHOT_PATH)
    odds, odds_status = _read_csv(ODDS_PATH)

    input_files = {
        "prediction_snapshots": snapshot_status,
        "market_odds_history": odds_status,
    }

    errors: List[str] = []
    warnings: List[str] = []

    if snapshots is None:
        report = {
            "generated_at": _utc_now(),
            "status": "partial",
            "input_files": input_files,
            "prediction_count": 0,
            "games_with_opening_odds": 0,
            "games_with_closing_odds": 0,
            "closing_odds_coverage_rate": 0.0,
            "clv_available_count": 0,
            "clv_available_rate": 0.0,
            "avg_time_from_snapshot_to_close_minutes": None,
            "missing_closing_game_ids": [],
            "suspicious_odds_count": 0,
            "suspicious_odds_examples": [],
            "errors": errors,
            "warnings": ["prediction_snapshots.csv missing or unreadable"],
            "recommendations": ["Generate prediction snapshots to evaluate closing odds coverage."],
        }
        OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return report

    snap = snapshots.copy()
    if "game_id" in snap.columns:
        snap["game_id"] = snap["game_id"].astype(str)

    prediction_count = int(len(snap))

    opening_columns = {"home_moneyline_odds", "away_moneyline_odds"}
    games_with_opening = 0
    if opening_columns.issubset(snap.columns):
        games_with_opening = int(
            snap.dropna(subset=list(opening_columns))["game_id"].nunique()
            if "game_id" in snap.columns
            else len(snap.dropna(subset=list(opening_columns)))
        )

    latest_odds = _latest_odds_by_game(odds)
    games_with_closing = 0
    missing_closing_game_ids: List[str] = []

    if not latest_odds.empty and "game_id" in snap.columns:
        closing_ids = set(latest_odds["game_id"].dropna().astype(str))
        prediction_ids = set(snap["game_id"].dropna().astype(str))
        games_with_closing = len(prediction_ids & closing_ids)
        missing_closing_game_ids = sorted(prediction_ids - closing_ids)[:50]
    else:
        warnings.append("market_odds_history.csv missing or does not contain game_id")

    clv_columns = [col for col in ("clv", "clv_home_moneyline", "clv_away_moneyline") if col in snap.columns]
    clv_available_count = 0
    if clv_columns:
        clv_available_count = int(snap[clv_columns].notna().any(axis=1).sum())

    suspicious_examples: List[Dict[str, Any]] = []
    for _, row in snap.iterrows():
        reasons = _suspicious_reasons(row)
        if reasons:
            suspicious_examples.append(
                {
                    "game_id": row.get("game_id"),
                    "reasons": reasons,
                }
            )

    closing_rate = (
        games_with_closing / max(1, len(set(snap["game_id"].dropna().astype(str))))
        if "game_id" in snap.columns
        else 0.0
    )

    clv_rate = clv_available_count / prediction_count if prediction_count else 0.0

    status = "ok"
    if prediction_count == 0:
        status = "partial"
    if games_with_closing == 0:
        status = "partial"

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": input_files,
        "prediction_count": prediction_count,
        "games_with_opening_odds": games_with_opening,
        "games_with_closing_odds": games_with_closing,
        "closing_odds_coverage_rate": round(closing_rate, 4),
        "clv_available_count": clv_available_count,
        "clv_available_rate": round(clv_rate, 4),
        "avg_time_from_snapshot_to_close_minutes": None,
        "missing_closing_game_ids": missing_closing_game_ids,
        "suspicious_odds_count": len(suspicious_examples),
        "suspicious_odds_examples": suspicious_examples[:20],
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Store closing odds per game to improve CLV and market-close diagnostics."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
