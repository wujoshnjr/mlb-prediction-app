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


def _bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
        .astype(bool)
    )


def _selected_side(row: pd.Series) -> str:
    side = str(row.get("moneyline_selected_side", "") or row.get("side", "") or "").strip().lower()
    if side in {"home", "away"}:
        return side

    recommendation = str(
        row.get("moneyline_recommendation", "")
        or row.get("recommendation", "")
        or ""
    ).strip().lower()
    home_team = str(row.get("home_team", "") or "").strip().lower()
    away_team = str(row.get("away_team", "") or "").strip().lower()

    if home_team and home_team in recommendation:
        return "home"
    if away_team and away_team in recommendation:
        return "away"

    edge = _to_float(row.get("model_edge_home"))
    if edge is not None:
        return "home" if edge >= 0 else "away"

    return "unknown"


def _entry_odds(row: pd.Series, side: str) -> Optional[float]:
    if side == "home":
        return _to_float(row.get("home_moneyline_odds"))
    if side == "away":
        return _to_float(row.get("away_moneyline_odds"))
    return None


def _market_moneyline_odds_by_game(
    odds: Optional[pd.DataFrame],
    *,
    closing_only: bool,
) -> pd.DataFrame:
    columns = ["game_id", "closing_home_odds", "closing_away_odds"]
    if odds is None or odds.empty:
        return pd.DataFrame(columns=columns)

    required = {"game_id", "market", "side", "odds"}
    if not required.issubset(odds.columns):
        return pd.DataFrame(columns=columns)

    frame = odds.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["market"] = frame["market"].astype(str).str.strip().str.lower()
    frame["side"] = frame["side"].astype(str).str.strip().str.lower()
    frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")

    frame = frame[
        (frame["market"] == "moneyline")
        & frame["side"].isin(["home", "away"])
        & frame["odds"].notna()
        & (frame["odds"] > 1.0)
    ].copy()

    if frame.empty:
        return pd.DataFrame(columns=columns)

    if "captured_at" in frame.columns:
        frame["_captured_at"] = pd.to_datetime(frame["captured_at"], errors="coerce", utc=True)
        frame = frame.sort_values("_captured_at")

    if closing_only:
        if "is_closing_snapshot" not in frame.columns:
            return pd.DataFrame(columns=columns)
        frame = frame[_bool_series(frame["is_closing_snapshot"])].copy()
        if frame.empty:
            return pd.DataFrame(columns=columns)
    else:
        if "is_closing_snapshot" in frame.columns:
            closing = frame[_bool_series(frame["is_closing_snapshot"])].copy()
        else:
            closing = pd.DataFrame()

        group_keys = ["game_id", "side"]
        if "bookmaker_key" in frame.columns:
            group_keys.append("bookmaker_key")

        latest = frame.groupby(group_keys, as_index=False).tail(1)

        if not closing.empty:
            latest_keys = latest[group_keys].astype(str).agg("|".join, axis=1)
            closing_keys = set(closing[group_keys].astype(str).agg("|".join, axis=1))
            latest = latest[~latest_keys.isin(closing_keys)]
            frame = pd.concat([latest, closing], ignore_index=True)
        else:
            frame = latest

    aggregated = (
        frame.groupby(["game_id", "side"], as_index=False)["odds"]
        .mean()
        .copy()
    )

    if aggregated.empty:
        return pd.DataFrame(columns=columns)

    pivot = aggregated.pivot_table(
        index="game_id",
        columns="side",
        values="odds",
        aggfunc="mean",
    ).reset_index()

    if "home" not in pivot.columns:
        pivot["home"] = None
    if "away" not in pivot.columns:
        pivot["away"] = None

    pivot = pivot.rename(
        columns={
            "home": "closing_home_odds",
            "away": "closing_away_odds",
        }
    )

    return pivot[columns].copy()


def _merge_closing_odds(
    snapshots: pd.DataFrame,
    closing_odds: pd.DataFrame,
) -> pd.DataFrame:
    if snapshots.empty or closing_odds.empty or "game_id" not in snapshots.columns:
        return snapshots

    frame = snapshots.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    merged = frame.merge(
        closing_odds,
        on="game_id",
        how="left",
        suffixes=("", "_from_market"),
    )

    for column in ("closing_home_odds", "closing_away_odds"):
        source = f"{column}_from_market"
        if source not in merged.columns:
            continue

        source_values = pd.to_numeric(merged[source], errors="coerce")

        if column in merged.columns:
            current_values = pd.to_numeric(merged[column], errors="coerce")
            merged[column] = current_values.combine_first(source_values)
        else:
            merged[column] = source_values

        merged.drop(columns=[source], inplace=True)

    return merged


def _suspicious_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    home_odds = _to_float(
        _first_existing(
            row,
            ["home_moneyline_odds", "home_odds"],
        )
    )
    away_odds = _to_float(
        _first_existing(
            row,
            ["away_moneyline_odds", "away_odds"],
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


def _compute_selected_clv(row: pd.Series) -> Optional[float]:
    side = _selected_side(row)
    if side not in {"home", "away"}:
        return None

    entry = _entry_odds(row, side)
    closing = (
        _to_float(row.get("closing_home_odds"))
        if side == "home"
        else _to_float(row.get("closing_away_odds"))
    )

    if entry is None or closing is None:
        return None

    if entry <= 1.0 or closing <= 1.0:
        return None

    return float(math.log(entry) - math.log(closing))


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
            "avg_clv": None,
            "positive_clv_rate": None,
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

    closing_odds = _market_moneyline_odds_by_game(odds, closing_only=True)

    games_with_closing = 0
    missing_closing_game_ids: List[str] = []

    if not closing_odds.empty and "game_id" in snap.columns:
        closing_ids = set(closing_odds["game_id"].dropna().astype(str))
        prediction_ids = set(snap["game_id"].dropna().astype(str))
        games_with_closing = len(prediction_ids & closing_ids)
        missing_closing_game_ids = sorted(prediction_ids - closing_ids)[:50]
    else:
        warnings.append("No closing moneyline odds were available from market_odds_history.csv")

    snap_with_closing = _merge_closing_odds(snap, closing_odds)

    clv_values: List[float] = []
    if not snap_with_closing.empty:
        for _, row in snap_with_closing.iterrows():
            clv = _compute_selected_clv(row)
            if clv is not None:
                clv_values.append(clv)

    clv_available_count = len(clv_values)
    clv_rate = clv_available_count / prediction_count if prediction_count else 0.0
    avg_clv = sum(clv_values) / clv_available_count if clv_available_count else None
    positive_clv_rate = (
        len([value for value in clv_values if value > 0]) / clv_available_count
        if clv_available_count
        else None
    )

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

    status = "ok"
    if prediction_count == 0:
        status = "partial"
    elif games_with_closing == 0:
        status = "partial"

    recommendations = []
    if clv_available_count == 0:
        recommendations.append(
            "Per-pick CLV is unavailable because no prediction snapshots matched closing moneyline odds."
        )
    else:
        recommendations.append(
            "Per-pick CLV is now derived from prediction snapshots joined to closing moneyline odds."
        )

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
        "avg_clv": round(avg_clv, 6) if avg_clv is not None else None,
        "positive_clv_rate": round(positive_clv_rate, 4) if positive_clv_rate is not None else None,
        "avg_time_from_snapshot_to_close_minutes": None,
        "missing_closing_game_ids": missing_closing_game_ids,
        "suspicious_odds_count": len(suspicious_examples),
        "suspicious_odds_examples": suspicious_examples[:20],
        "errors": errors,
        "warnings": warnings,
        "recommendations": recommendations,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
