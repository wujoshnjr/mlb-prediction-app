from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

PREDICTION_PATH = REPORT_DIR / "prediction.json"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
MARKET_ODDS_PATH = DATA_DIR / "market_odds_history.csv"

LEDGER_PATH = DATA_DIR / "paper_trading_ledger.csv"
OUTPUT_REPORT = REPORT_DIR / "paper_trading_ledger_report.json"

LEDGER_COLUMNS = [
    "ledger_id",
    "created_at",
    "game_id",
    "game_date",
    "side",
    "decimal_odds",
    "model_prob",
    "market_prob",
    "edge",
    "paper_stake_units",
    "potential_profit_units",
    "settled",
    "outcome",
    "profit_loss_units",
    "clv",
    "recommendation_status",
    "data_quality_grade",
    "notes",
]

DEFAULT_PAPER_STAKE_UNITS = 0.25


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Optional[Any], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status
    return data, status


def _read_csv(path: Path) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
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


def _predictions(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("predictions", "today_predictions", "games"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _ledger_id(game_id: str, side: str, created_date: str) -> str:
    raw = f"{game_id}|{side}|{created_date}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_ledger() -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    try:
        frame = pd.read_csv(LEDGER_PATH)
    except Exception:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    return frame[LEDGER_COLUMNS]


def _side_from_prediction(prediction: Dict[str, Any]) -> Optional[str]:
    side = str(prediction.get("side", "") or "").strip().lower()
    if side in {"home", "away"}:
        return side

    edge = _to_float(prediction.get("model_edge_home") or prediction.get("edge"))
    if edge is not None:
        return "home" if edge >= 0 else "away"

    recommendation = str(
        prediction.get("recommendation")
        or prediction.get("moneyline_recommendation")
        or ""
    ).lower()

    home_team = str(prediction.get("home_team", "") or "").lower()
    away_team = str(prediction.get("away_team", "") or "").lower()

    if "home" in recommendation or (home_team and home_team in recommendation):
        return "home"
    if "away" in recommendation or (away_team and away_team in recommendation):
        return "away"

    return None


def _is_paper_candidate(prediction: Dict[str, Any]) -> bool:
    status = str(prediction.get("recommendation_status", "") or "").lower()
    recommendation = str(
        prediction.get("recommendation")
        or prediction.get("moneyline_recommendation")
        or ""
    ).lower()

    if "no_bet" in status or "no bet" in status:
        return False
    if "pass" in recommendation or "no_bet" in recommendation or "no bet" in recommendation:
        return False

    if "paper" in status:
        return True

    # If a recommendation exists but live betting is false, record as paper tracking.
    if recommendation and prediction.get("live_betting_allowed") is False:
        return True

    return False


def _data_quality_grade(prediction: Dict[str, Any]) -> str:
    dq = prediction.get("data_quality_status")
    if isinstance(dq, dict):
        return str(dq.get("data_quality_grade") or dq.get("grade") or "")
    return str(prediction.get("data_quality_grade") or "")


def _settle_entry(entry: Dict[str, Any], finalized: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if finalized is None or finalized.empty or "game_id" not in finalized.columns:
        return entry

    game_id = str(entry.get("game_id"))
    frame = finalized.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    match = frame[frame["game_id"] == game_id]

    if match.empty:
        return entry

    row = match.iloc[0]

    home_win = row.get("home_win")
    if pd.isna(home_win):
        if "home_score" in row.index and "away_score" in row.index:
            home_score = _to_float(row.get("home_score"))
            away_score = _to_float(row.get("away_score"))
            if home_score is not None and away_score is not None and home_score != away_score:
                home_win = 1 if home_score > away_score else 0

    if pd.isna(home_win):
        return entry

    side = entry.get("side")
    stake = _to_float(entry.get("paper_stake_units")) or 0.0
    odds = _to_float(entry.get("decimal_odds")) or 0.0

    won = (int(home_win) == 1 and side == "home") or (int(home_win) == 0 and side == "away")

    entry["settled"] = True
    entry["outcome"] = "win" if won else "loss"
    entry["profit_loss_units"] = round(stake * (odds - 1.0), 6) if won else -stake

    return entry


def build_ledger() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prediction_data, prediction_status = _read_json(PREDICTION_PATH)
    finalized, finalized_status = _read_csv(FINALIZED_PATH)
    _, market_status = _read_csv(MARKET_ODDS_PATH)

    input_files = {
        "prediction": prediction_status,
        "finalized_games": finalized_status,
        "market_odds_history": market_status,
    }

    predictions = _predictions(prediction_data)
    ledger = _load_ledger()

    created_date = datetime.now(timezone.utc).date().isoformat()

    existing_keys = set()
    if not ledger.empty:
        for _, row in ledger.iterrows():
            existing_keys.add((str(row.get("game_id")), str(row.get("side")), str(row.get("created_at", ""))[:10]))

    new_entries: List[Dict[str, Any]] = []

    for prediction in predictions:
        if not _is_paper_candidate(prediction):
            continue

        game_id = str(prediction.get("game_id") or "")
        if not game_id:
            continue

        side = _side_from_prediction(prediction)
        if side not in {"home", "away"}:
            continue

        odds = _to_float(
            prediction.get("home_moneyline_odds")
            if side == "home"
            else prediction.get("away_moneyline_odds")
        )
        if odds is None or odds <= 1.0:
            continue

        duplicate_key = (game_id, side, created_date)
        if duplicate_key in existing_keys:
            continue

        stake = DEFAULT_PAPER_STAKE_UNITS
        model_prob = _to_float(
            prediction.get("model_prob")
            or prediction.get("predicted_home_win_pct")
            or prediction.get("premarket_model_home_prob")
            or prediction.get("home_win_probability")
        )
        market_prob = _to_float(prediction.get("market_no_vig_home_prob") or prediction.get("market_prob"))
        edge = _to_float(prediction.get("model_edge_home") or prediction.get("edge"))

        entry = {
            "ledger_id": _ledger_id(game_id, side, created_date),
            "created_at": _utc_now(),
            "game_id": game_id,
            "game_date": prediction.get("game_date"),
            "side": side,
            "decimal_odds": odds,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": edge,
            "paper_stake_units": stake,
            "potential_profit_units": round(stake * (odds - 1.0), 6),
            "settled": False,
            "outcome": None,
            "profit_loss_units": None,
            "clv": prediction.get("clv"),
            "recommendation_status": prediction.get("recommendation_status"),
            "data_quality_grade": _data_quality_grade(prediction),
            "notes": "paper trade only; live stake is always zero",
        }

        entry = _settle_entry(entry, finalized)
        new_entries.append(entry)
        existing_keys.add(duplicate_key)

    if new_entries:
        new_frame = pd.DataFrame(new_entries)
        combined = pd.concat([ledger, new_frame], ignore_index=True)
    else:
        combined = ledger

    for column in LEDGER_COLUMNS:
        if column not in combined.columns:
            combined[column] = None

    combined[LEDGER_COLUMNS].to_csv(LEDGER_PATH, index=False)

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if predictions else "partial",
        "input_files": input_files,
        "ledger_count": int(len(combined)),
        "new_entries": int(len(new_entries)),
        "live_stake_units": 0.0,
        "errors": [],
        "warnings": [] if predictions else ["No predictions available for paper ledger."],
        "recommendations": [
            "Ledger records paper trades only. Live stake remains zero."
        ],
    }

    OUTPUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_ledger()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
