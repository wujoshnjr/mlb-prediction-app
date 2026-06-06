from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

LEDGER_PATH = DATA_DIR / "paper_trading_ledger.csv"
PREDICTION_PATH = REPORT_DIR / "prediction.json"
OUTPUT_PATH = REPORT_DIR / "risk_exposure_report.json"


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


def _read_json_status(path: Path) -> Dict[str, Any]:
    status = {"path": str(path), "exists": path.exists(), "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
    return status


def _to_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _settled_mask(frame: pd.DataFrame) -> pd.Series:
    if "settled" not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    return frame["settled"].astype(str).str.lower().isin({"true", "1", "yes"})


def _edge_bucket(value: Any) -> str:
    try:
        edge = abs(float(value))
    except Exception:
        return "unknown"
    if edge < 0.03:
        return "below_3pct"
    if edge < 0.05:
        return "3_to_5pct"
    if edge < 0.08:
        return "5_to_8pct"
    return "8pct_plus"


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ledger, ledger_status = _read_csv(LEDGER_PATH)
    prediction_status = _read_json_status(PREDICTION_PATH)

    input_files = {
        "paper_trading_ledger": ledger_status,
        "prediction": prediction_status,
    }

    if ledger is None or ledger.empty:
        report = {
            "generated_at": _utc_now(),
            "status": "partial",
            "input_files": input_files,
            "total_open_paper_units": 0.0,
            "total_settled_units": 0.0,
            "daily_paper_units": {},
            "max_single_game_units": 0.0,
            "exposure_by_side": {},
            "exposure_by_team": {},
            "exposure_by_edge_bucket": {},
            "exposure_by_data_quality_grade": {},
            "live_exposure_units": 0.0,
            "risk_status": "normal",
            "risk_blockers": [],
            "errors": [],
            "warnings": ["paper_trading_ledger.csv missing or empty"],
            "recommendations": ["Generate paper_trading_ledger.csv before assessing exposure."],
        }
        OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return report

    frame = ledger.copy()
    stake = _to_numeric_series(frame, "paper_stake_units")
    settled = _settled_mask(frame)
    open_frame = frame[~settled].copy()
    open_stake = stake.loc[open_frame.index]

    live_exposure = 0.0
    if "live_stake_units" in frame.columns:
        live_exposure = float(pd.to_numeric(frame["live_stake_units"], errors="coerce").fillna(0.0).sum())

    risk_blockers = []
    errors = []
    if live_exposure > 0:
        risk_blockers.append("live_stake_units > 0")
        errors.append("Live exposure detected; this must remain zero.")

    daily_paper_units: Dict[str, float] = {}
    if "created_at" in frame.columns:
        created_dates = pd.to_datetime(frame["created_at"], errors="coerce", utc=True).dt.date.astype(str)
        frame["_created_date"] = created_dates
        daily = frame.groupby("_created_date")["paper_stake_units"].apply(
            lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())
        )
        daily_paper_units = {str(k): round(float(v), 6) for k, v in daily.to_dict().items()}

    max_single_game = 0.0
    if "game_id" in open_frame.columns and not open_frame.empty:
        game_exposure = open_frame.assign(_stake=open_stake).groupby("game_id")["_stake"].sum()
        if not game_exposure.empty:
            max_single_game = float(game_exposure.max())

    exposure_by_side = {}
    if "side" in open_frame.columns:
        exposure_by_side = {
            str(k): round(float(v), 6)
            for k, v in open_frame.assign(_stake=open_stake).groupby("side")["_stake"].sum().to_dict().items()
        }

    exposure_by_team: Dict[str, float] = {}

    exposure_by_edge_bucket = {}
    if "edge" in open_frame.columns:
        temp = open_frame.assign(
            _stake=open_stake,
            _edge_bucket=open_frame["edge"].apply(_edge_bucket),
        )
        exposure_by_edge_bucket = {
            str(k): round(float(v), 6)
            for k, v in temp.groupby("_edge_bucket")["_stake"].sum().to_dict().items()
        }

    exposure_by_data_quality_grade = {}
    if "data_quality_grade" in open_frame.columns:
        exposure_by_data_quality_grade = {
            str(k): round(float(v), 6)
            for k, v in open_frame.assign(_stake=open_stake)
            .groupby("data_quality_grade")["_stake"]
            .sum()
            .to_dict()
            .items()
        }

    status = "failed" if errors else "ok"

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": input_files,
        "total_open_paper_units": round(float(open_stake.sum()), 6),
        "total_settled_units": round(float(stake.loc[settled].sum()), 6),
        "daily_paper_units": daily_paper_units,
        "max_single_game_units": round(max_single_game, 6),
        "exposure_by_side": exposure_by_side,
        "exposure_by_team": exposure_by_team,
        "exposure_by_edge_bucket": exposure_by_edge_bucket,
        "exposure_by_data_quality_grade": exposure_by_data_quality_grade,
        "live_exposure_units": round(live_exposure, 6),
        "risk_status": "failed" if errors else "normal",
        "risk_blockers": risk_blockers,
        "errors": errors,
        "warnings": [],
        "recommendations": [
            "Live exposure must remain zero. Paper exposure should stay small until model evidence improves."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
