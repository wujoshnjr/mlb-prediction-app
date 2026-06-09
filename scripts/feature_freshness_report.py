from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
MARKET_ODDS_PATH = DATA_DIR / "market_odds_history.csv"
DAILY_CONTEXT_PATH = DATA_DIR / "daily_game_context.csv"
SAVANT_TOP3_CONTEXT_PATH = DATA_DIR / "savant_top3_context.csv"
WEATHER_CONTEXT_PATH = DATA_DIR / "weather_context.csv"

REPORT_PATH = REPORT_DIR / "feature_freshness_report.json"

SOURCE_CONFIG = {
    "odds": {
        "path": MARKET_ODDS_PATH,
        "time_columns": ["last_update", "captured_at", "snapshot_created_at", "created_at", "updated_at"],
    },
    "prediction_snapshots": {
        "path": SNAPSHOT_PATH,
        "time_columns": ["snapshot_created_at", "generated_at", "created_at", "updated_at"],
    },
    "daily_context": {
        "path": DAILY_CONTEXT_PATH,
        "time_columns": ["captured_at", "generated_at", "created_at", "updated_at"],
    },
    "lineup": {
        "path": DAILY_CONTEXT_PATH,
        "time_columns": ["lineup_captured_at", "captured_at", "generated_at", "created_at", "updated_at"],
    },
    "savant_top3": {
        "path": SAVANT_TOP3_CONTEXT_PATH,
        "time_columns": ["captured_at", "generated_at", "created_at", "updated_at"],
    },
    "weather": {
        "path": WEATHER_CONTEXT_PATH,
        "time_columns": ["captured_at", "weather_captured_at", "generated_at", "created_at", "updated_at"],
    },
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_str() -> str:
    return _utc_now().isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value) and not isinstance(value, (str, bool)):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "file_missing"
    try:
        return pd.read_csv(path), ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)


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
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _grade_from_age(age_minutes: Optional[float]) -> str:
    if age_minutes is None:
        return "D"
    if age_minutes <= 30:
        return "A"
    if age_minutes <= 120:
        return "B"
    if age_minutes <= 360:
        return "C"
    return "D"


def _parse_latest_time(frame: pd.DataFrame, time_columns: list[str]) -> tuple[Optional[pd.Timestamp], str]:
    if frame.empty:
        return None, ""

    for column in time_columns:
        if column not in frame.columns:
            continue

        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        parsed = parsed.dropna()
        if not parsed.empty:
            return parsed.max(), column

    return None, ""


def _source_freshness(name: str, config: dict[str, Any], now: datetime) -> dict[str, Any]:
    frame, error = _read_csv(config["path"])
    warnings: list[str] = []

    if error:
        return {
            "path": str(config["path"]),
            "exists": config["path"].exists(),
            "row_count": 0,
            "latest_timestamp": None,
            "timestamp_column": "",
            "age_minutes": None,
            "grade": "D",
            "warnings": [f"{name} unavailable: {error}"],
        }

    latest, column = _parse_latest_time(frame, list(config.get("time_columns", [])))
    if latest is None:
        warnings.append(f"{name} has no parseable timestamp column")
        return {
            "path": str(config["path"]),
            "exists": True,
            "row_count": int(len(frame)),
            "latest_timestamp": None,
            "timestamp_column": "",
            "age_minutes": None,
            "grade": "D",
            "warnings": warnings,
        }

    latest_dt = latest.to_pydatetime()
    age_minutes = max(0.0, (now - latest_dt).total_seconds() / 60.0)

    return {
        "path": str(config["path"]),
        "exists": True,
        "row_count": int(len(frame)),
        "latest_timestamp": latest_dt.isoformat(),
        "timestamp_column": column,
        "age_minutes": float(age_minutes),
        "grade": _grade_from_age(age_minutes),
        "warnings": warnings,
    }


def _grade_rank(grade: str) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(grade), 3)


def _global_grade(freshness: dict[str, Any]) -> str:
    grades = [
        str(value.get("grade", "D"))
        for value in freshness.values()
        if isinstance(value, dict)
    ]
    if not grades:
        return "D"

    worst = max(grades, key=_grade_rank)
    return worst


def _latest_by_game(frame: pd.DataFrame, time_columns: list[str]) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    selected_col = next((col for col in time_columns if col in result.columns), None)
    if selected_col:
        result["_freshness_time"] = pd.to_datetime(result[selected_col], errors="coerce", utc=True)
        result = result.sort_values(["game_id", "_freshness_time"], kind="mergesort")
        result = result.groupby("game_id", as_index=False).tail(1)
    else:
        result = result.drop_duplicates("game_id", keep="last")

    return result.reset_index(drop=True)


def _game_level_freshness(now: datetime) -> list[dict[str, Any]]:
    snapshots, snapshot_error = _read_csv(SNAPSHOT_PATH)
    if snapshot_error or snapshots.empty or "game_id" not in snapshots.columns:
        return []

    base = snapshots.copy()
    base["game_id"] = base["game_id"].apply(_normalize_game_id)
    base = base[base["game_id"] != ""].copy()

    if "snapshot_created_at" in base.columns:
        base["_sort_time"] = pd.to_datetime(base["snapshot_created_at"], errors="coerce", utc=True)
        base = base.sort_values(["game_id", "_sort_time"], kind="mergesort")
        base = base.groupby("game_id", as_index=False).tail(1)
    else:
        base = base.drop_duplicates("game_id", keep="last")

    for column in ["game_date", "home_team", "away_team"]:
        if column not in base.columns:
            base[column] = ""

    rows = base[["game_id", "game_date", "home_team", "away_team"]].copy()

    source_game_grades: dict[str, dict[str, str]] = {}

    for source_name, config in SOURCE_CONFIG.items():
        frame, error = _read_csv(config["path"])
        if error or frame.empty or "game_id" not in frame.columns:
            continue

        latest = _latest_by_game(frame, list(config.get("time_columns", [])))
        if latest.empty:
            continue

        timestamp, column = _parse_latest_time(latest, list(config.get("time_columns", [])))
        if not column:
            continue

        parsed = pd.to_datetime(latest[column], errors="coerce", utc=True)
        latest = latest.copy()
        latest["_age_minutes"] = parsed.apply(
            lambda value: None if pd.isna(value) else max(0.0, (now - value.to_pydatetime()).total_seconds() / 60.0)
        )
        latest[f"{source_name}_grade"] = latest["_age_minutes"].apply(_grade_from_age)

        source_game_grades[source_name] = (
            latest[["game_id", f"{source_name}_grade"]]
            .set_index("game_id")[f"{source_name}_grade"]
            .to_dict()
        )

    output: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        game_id = _normalize_game_id(row.get("game_id"))
        grades = {
            source: source_map.get(game_id, "D")
            for source, source_map in source_game_grades.items()
        }
        output.append(
            {
                "game_id": game_id,
                "game_date": str(row.get("game_date") or ""),
                "home_team": str(row.get("home_team") or ""),
                "away_team": str(row.get("away_team") or ""),
                "grades": grades,
                "overall_grade": _global_grade({key: {"grade": value} for key, value in grades.items()}),
            }
        )

    return output


def build_report() -> dict[str, Any]:
    now = _utc_now()

    freshness = {
        name: _source_freshness(name, config, now)
        for name, config in SOURCE_CONFIG.items()
    }

    stale_sources = [
        name for name, value in freshness.items()
        if isinstance(value, dict) and value.get("grade") in {"C", "D"}
    ]

    recommendations: list[str] = []
    if stale_sources:
        recommendations.append("Some sources are stale or missing; keep affected games tracking-only.")
    if "lineup" in stale_sources:
        recommendations.append("Lineup source is stale or missing; do not upgrade lineup-sensitive slices.")

    warnings: list[str] = []
    for source, value in freshness.items():
        if isinstance(value, dict):
            warnings.extend(str(item) for item in value.get("warnings", []))

    game_level = _game_level_freshness(now)

    return {
        "generated_at": _utc_now_str(),
        "status": "ok",
        "freshness": freshness,
        "game_level_freshness": game_level,
        "global_grade": _global_grade(freshness),
        "stale_sources": stale_sources,
        "recommendations": recommendations,
        "warnings": warnings,
        "errors": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def main() -> None:
    report = build_report()
    _write_json(REPORT_PATH, report)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
