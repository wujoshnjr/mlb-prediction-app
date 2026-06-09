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
DAILY_CONTEXT_PATH = DATA_DIR / "daily_game_context.csv"
SAVANT_TOP3_CONTEXT_PATH = DATA_DIR / "savant_top3_context.csv"
PROJECTED_LINEUP_CONTEXT_PATH = DATA_DIR / "projected_lineup_context.csv"

OUTPUT_CSV_PATH = DATA_DIR / "lineup_quality_context.csv"
REPORT_PATH = REPORT_DIR / "lineup_quality_report.json"

CLEAN_PIPELINE_VERSION = "baseline_v2_clean"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "valid", "ok"}


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "null", "[]", "{}"})


def _numeric_positive(value: Any) -> bool:
    try:
        parsed = float(value)
        return math.isfinite(parsed) and parsed > 0
    except Exception:
        return False


def _latest_by_game(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    time_columns = [
        "captured_at",
        "snapshot_created_at",
        "generated_at",
        "updated_at",
        "created_at",
        "last_update",
    ]

    selected_time_column = next((column for column in time_columns if column in result.columns), None)
    if selected_time_column:
        result["_sort_time"] = pd.to_datetime(
            result[selected_time_column],
            errors="coerce",
            utc=True,
        )
        result = result.sort_values(["game_id", "_sort_time"], kind="mergesort")
        result = result.groupby("game_id", as_index=False).tail(1)
    else:
        result = result.drop_duplicates("game_id", keep="last")

    return result.reset_index(drop=True)


def _prepare_snapshot_base(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "pipeline_version" in result.columns:
        preferred = result[result["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION].copy()
        if not preferred.empty:
            result = preferred

    if "snapshot_valid" in result.columns:
        valid = result["snapshot_valid"].astype(str).str.strip().str.lower()
        result = result[valid.isin({"true", "1", "yes", "y", "valid", "ok"})].copy()

    result = _latest_by_game(result)

    for column in ["game_date", "home_team", "away_team"]:
        if column not in result.columns:
            result[column] = ""

    return result[["game_id", "game_date", "home_team", "away_team"]].drop_duplicates("game_id", keep="last")


def _column_value(row: pd.Series, candidates: list[str]) -> Any:
    for column in candidates:
        if column in row.index:
            value = row.get(column)
            if _present(value):
                return value
    return None


def _side_confirmed(row: pd.Series, side: str) -> bool:
    candidates = [
        f"{side}_lineup_confirmed",
        f"{side}_confirmed_lineup_available",
        f"{side}_lineup_available",
    ]
    value = _column_value(row, candidates)
    if value is not None:
        return _truthy(value) or _numeric_positive(value)

    count_value = _column_value(row, [f"{side}_lineup_player_count"])
    return _numeric_positive(count_value) and float(count_value) >= 9


def _side_projected(row: pd.Series, side: str) -> bool:
    candidates = [
        f"{side}_projected_lineup_available",
        f"{side}_projected_players_available",
        f"{side}_projected_lineup_player_count",
        f"{side}_projected_lineup_json",
        f"{side}_projected_lineup",
    ]
    value = _column_value(row, candidates)
    if value is None:
        return False
    return _truthy(value) or _numeric_positive(value) or _present(value)


def _side_top3_ids(row: pd.Series, side: str) -> bool:
    value = _column_value(row, [f"{side}_top3_player_ids", f"{side}_top3_ids"])
    return _present(value)


def _side_top3_savant(row: pd.Series, side: str) -> bool:
    value = _column_value(
        row,
        [
            f"{side}_top3_savant_available_count",
            f"{side}_top3_woba_available_count",
            f"{side}_top3_statcast_available_count",
        ],
    )
    return _numeric_positive(value)


def _score_side(
    *,
    confirmed: bool,
    projected: bool,
    top3_ids: bool,
    top3_savant: bool,
) -> int:
    score = 0
    if confirmed:
        score += 40
    if projected:
        score += 25
    if top3_ids:
        score += 20
    if top3_savant:
        score += 20
    return min(score, 100)


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 30:
        return "C"
    return "D"


def build_lineup_quality() -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    errors: list[str] = []

    snapshots_raw, snapshot_error = _read_csv(SNAPSHOT_PATH)
    daily_raw, daily_error = _read_csv(DAILY_CONTEXT_PATH)
    savant_raw, savant_error = _read_csv(SAVANT_TOP3_CONTEXT_PATH)
    projected_raw, projected_error = _read_csv(PROJECTED_LINEUP_CONTEXT_PATH)

    for label, error in [
        ("prediction_snapshots", snapshot_error),
        ("daily_game_context", daily_error),
        ("savant_top3_context", savant_error),
        ("projected_lineup_context", projected_error),
    ]:
        if error:
            warnings.append(f"{label} unavailable: {error}")

    base = _prepare_snapshot_base(snapshots_raw)
    if base.empty:
        errors.append("No clean prediction snapshots available.")
        output = pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "home_lineup_quality_score",
                "away_lineup_quality_score",
                "lineup_quality_diff",
                "home_top3_available",
                "away_top3_available",
                "home_projected_lineup_available",
                "away_projected_lineup_available",
                "home_confirmed_lineup_available",
                "away_confirmed_lineup_available",
                "lineup_confidence_grade",
                "lineup_context_available",
                "lineup_quality_warning",
            ]
        )
        report = {
            "generated_at": _utc_now(),
            "status": "skipped",
            "row_count": 0,
            "grade_counts": {},
            "warnings": warnings,
            "errors": errors,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }
        return output, report

    daily = _latest_by_game(daily_raw)
    savant = _latest_by_game(savant_raw)
    projected = _latest_by_game(projected_raw)

    merged = base.copy()

    if not daily.empty:
        merged = merged.merge(daily, on="game_id", how="left", suffixes=("", "_daily"))

    if not savant.empty:
        merged = merged.merge(savant, on="game_id", how="left", suffixes=("", "_savant"))

    if not projected.empty:
        merged = merged.merge(projected, on="game_id", how="left", suffixes=("", "_projected"))

    rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        home_confirmed = _side_confirmed(row, "home")
        away_confirmed = _side_confirmed(row, "away")
        home_projected = _side_projected(row, "home")
        away_projected = _side_projected(row, "away")
        home_top3_ids = _side_top3_ids(row, "home")
        away_top3_ids = _side_top3_ids(row, "away")
        home_top3_savant = _side_top3_savant(row, "home")
        away_top3_savant = _side_top3_savant(row, "away")

        home_score = _score_side(
            confirmed=home_confirmed,
            projected=home_projected,
            top3_ids=home_top3_ids,
            top3_savant=home_top3_savant,
        )
        away_score = _score_side(
            confirmed=away_confirmed,
            projected=away_projected,
            top3_ids=away_top3_ids,
            top3_savant=away_top3_savant,
        )

        combined_score = min(home_score, away_score)
        grade = _grade(combined_score)

        warning_parts = []
        if not home_confirmed or not away_confirmed:
            warning_parts.append("confirmed lineup missing")
        if not home_projected or not away_projected:
            warning_parts.append("projected lineup incomplete")
        if not home_top3_ids or not away_top3_ids:
            warning_parts.append("top3 hitter ids incomplete")
        if not home_top3_savant or not away_top3_savant:
            warning_parts.append("top3 savant incomplete")

        rows.append(
            {
                "game_id": _normalize_game_id(row.get("game_id")),
                "game_date": str(row.get("game_date") or row.get("game_date_daily") or ""),
                "home_team": str(row.get("home_team") or row.get("home_team_daily") or ""),
                "away_team": str(row.get("away_team") or row.get("away_team_daily") or ""),
                "home_lineup_quality_score": int(home_score),
                "away_lineup_quality_score": int(away_score),
                "lineup_quality_diff": int(home_score - away_score),
                "home_top3_available": bool(home_top3_ids and home_top3_savant),
                "away_top3_available": bool(away_top3_ids and away_top3_savant),
                "home_projected_lineup_available": bool(home_projected),
                "away_projected_lineup_available": bool(away_projected),
                "home_confirmed_lineup_available": bool(home_confirmed),
                "away_confirmed_lineup_available": bool(away_confirmed),
                "lineup_confidence_grade": grade,
                "lineup_context_available": grade in {"A", "B", "C"},
                "lineup_quality_warning": " | ".join(warning_parts),
            }
        )

    output = pd.DataFrame(rows)
    grade_counts = (
        output["lineup_confidence_grade"].value_counts(dropna=False).to_dict()
        if not output.empty and "lineup_confidence_grade" in output.columns
        else {}
    )

    report = {
        "generated_at": _utc_now(),
        "status": "ok" if not errors else "partial",
        "row_count": int(len(output)),
        "grade_counts": grade_counts,
        "context_available_count": int(output["lineup_context_available"].sum()) if not output.empty else 0,
        "average_home_lineup_quality_score": float(output["home_lineup_quality_score"].mean()) if not output.empty else None,
        "average_away_lineup_quality_score": float(output["away_lineup_quality_score"].mean()) if not output.empty else None,
        "warnings": warnings,
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    return output, report


def main() -> None:
    output, report = build_lineup_quality()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV_PATH, index=False)

    _write_json(REPORT_PATH, report)
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
