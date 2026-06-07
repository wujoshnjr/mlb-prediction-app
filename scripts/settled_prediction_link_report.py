from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
OUTPUT_PATH = REPORT_DIR / "settled_prediction_link_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "error": "",
    }

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


def _normalize_game_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _normalize_team(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace("  ", " ")
    )


def _prepare_snapshots(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if frame is None or frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "game_date" in result.columns:
        result["game_date"] = result["game_date"].astype(str).str.slice(0, 10)
    else:
        result["game_date"] = ""

    for column in ("home_team", "away_team"):
        if column not in result.columns:
            result[column] = ""

    result["_home_team_norm"] = result["home_team"].apply(_normalize_team)
    result["_away_team_norm"] = result["away_team"].apply(_normalize_team)

    return result


def _prepare_finalized(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if frame is None or frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "game_date" in result.columns:
        result["game_date"] = result["game_date"].astype(str).str.slice(0, 10)
    else:
        result["game_date"] = ""

    if "home_win" not in result.columns:
        if {"home_score", "away_score"}.issubset(result.columns):
            result["home_win"] = (
                pd.to_numeric(result["home_score"], errors="coerce")
                > pd.to_numeric(result["away_score"], errors="coerce")
            ).astype("Int64")
        else:
            result["home_win"] = pd.NA

    result["home_win"] = pd.to_numeric(result["home_win"], errors="coerce")
    result = result[result["home_win"].isin([0, 1])].copy()

    for column in ("home_team", "away_team"):
        if column not in result.columns:
            result[column] = ""

    result["_home_team_norm"] = result["home_team"].apply(_normalize_team)
    result["_away_team_norm"] = result["away_team"].apply(_normalize_team)

    return result.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _sample_unlinked(frame: pd.DataFrame, linked_ids: set[str], limit: int = 25) -> List[Dict[str, Any]]:
    if frame.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        game_id = str(row.get("game_id") or "").strip()
        if not game_id or game_id in linked_ids:
            continue

        rows.append(
            {
                "game_id": game_id,
                "game_date": row.get("game_date"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "start_time": row.get("start_time"),
            }
        )

        if len(rows) >= limit:
            break

    return rows


def _date_range(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    if frame.empty or "game_date" not in frame.columns:
        return {"min": None, "max": None}

    dates = frame["game_date"].dropna().astype(str)
    dates = dates[dates.str.strip() != ""]
    if dates.empty:
        return {"min": None, "max": None}

    return {
        "min": str(dates.min()),
        "max": str(dates.max()),
    }


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots_raw, snapshot_status = _read_csv(SNAPSHOT_PATH)
    finalized_raw, finalized_status = _read_csv(FINALIZED_PATH)

    snapshots = _prepare_snapshots(snapshots_raw)
    finalized = _prepare_finalized(finalized_raw)

    errors: List[str] = []
    warnings: List[str] = []

    if snapshots_raw is None:
        errors.append("prediction_snapshots.csv missing or unreadable")

    if finalized_raw is None:
        errors.append("finalized_games.csv missing or unreadable")

    snapshot_game_ids = set(snapshots["game_id"].dropna().astype(str)) if not snapshots.empty else set()
    finalized_game_ids = set(finalized["game_id"].dropna().astype(str)) if not finalized.empty else set()
    linked_game_ids = snapshot_game_ids & finalized_game_ids

    linked_rows = pd.DataFrame()
    if not snapshots.empty and not finalized.empty:
        linked_rows = snapshots.merge(
            finalized[
                [
                    "game_id",
                    "game_date",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "home_win",
                ]
            ],
            on="game_id",
            how="inner",
            suffixes=("_snapshot", "_final"),
        )

    linked_snapshot_row_count = int(len(linked_rows))
    linked_game_count = int(len(linked_game_ids))
    snapshot_game_count = int(len(snapshot_game_ids))
    finalized_game_count = int(len(finalized_game_ids))

    link_rate = (
        linked_game_count / snapshot_game_count
        if snapshot_game_count
        else 0.0
    )

    if snapshot_game_count > 0 and linked_game_count == 0:
        warnings.append(
            "No prediction snapshot game_id currently links to finalized_games.csv. "
            "Baseline, calibration, and rolling OOS evidence will remain at zero until finalized_games is updated."
        )

    today = datetime.now(timezone.utc).date().isoformat()
    pending_past = pd.DataFrame()
    if not snapshots.empty and "game_date" in snapshots.columns:
        pending_past = snapshots[
            (snapshots["game_date"].astype(str) < today)
            & (~snapshots["game_id"].astype(str).isin(linked_game_ids))
        ].copy()

    status = "failed" if errors else "ok"

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": {
            "prediction_snapshots": snapshot_status,
            "finalized_games": finalized_status,
        },
        "snapshot_rows": int(len(snapshots_raw)) if snapshots_raw is not None else 0,
        "snapshot_game_count": snapshot_game_count,
        "finalized_rows": int(len(finalized_raw)) if finalized_raw is not None else 0,
        "finalized_game_count": finalized_game_count,
        "linked_game_count": linked_game_count,
        "linked_snapshot_row_count": linked_snapshot_row_count,
        "link_rate": round(link_rate, 6),
        "unlinked_snapshot_game_count": int(max(0, snapshot_game_count - linked_game_count)),
        "pending_past_snapshot_game_count": int(pending_past["game_id"].nunique()) if not pending_past.empty else 0,
        "snapshot_date_range": _date_range(snapshots),
        "finalized_date_range": _date_range(finalized),
        "sample_linked_game_ids": sorted(list(linked_game_ids))[:25],
        "sample_unlinked_snapshots": _sample_unlinked(snapshots, linked_game_ids),
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Use finalized_games.csv as the only trusted outcome source for baseline, calibration, ROI, and rolling OOS evaluation.",
            "Run update_results.py before evaluation reports so newly finalized snapshot games are appended to finalized_games.csv.",
            "If linked_game_count remains zero, inspect MLB game_id source consistency between prediction snapshots and finalized game collection.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
