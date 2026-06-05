from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "home_team",
    "away_team",
    "bullpen_ip_diff",
    "bullpen_availability_diff",
    "home_bullpen_fatigue_score",
    "away_bullpen_fatigue_score",
    "home_bullpen_pitches_last_1d",
    "away_bullpen_pitches_last_1d",
    "home_bullpen_pitches_last_3d",
    "away_bullpen_pitches_last_3d",
    "home_closer_risk_score",
    "away_closer_risk_score",
    "lineup_projection_available_diff",
    "context_bridge_source_status",
    "context_bridge_reason",
    "context_bridge_captured_at",
]

SOURCE_COLUMNS = [
    "home_bullpen_fatigue_score",
    "away_bullpen_fatigue_score",
    "home_bullpen_pitches_last_1d",
    "away_bullpen_pitches_last_1d",
    "home_bullpen_pitches_last_3d",
    "away_bullpen_pitches_last_3d",
    "home_closer_risk_score",
    "away_closer_risk_score",
    "home_projected_lineup_available",
    "away_projected_lineup_available",
]


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return float(number)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False

    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if value == 1:
            return True
        if value == 0:
            return False

    return None


def _empty_output(output_path: Optional[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    return frame


def _clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.where(pd.notnull(frame), "")
    return frame


def _latest_context_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["game_id"] = working["game_id"].astype(str)

    if "captured_at" in working.columns:
        working["captured_at_dt"] = pd.to_datetime(
            working["captured_at"],
            errors="coerce",
            utc=True,
        )
        if working["captured_at_dt"].notna().any():
            return (
                working.sort_values(["game_id", "captured_at_dt"])
                .groupby("game_id", as_index=False)
                .tail(1)
            )

    return working.drop_duplicates(subset=["game_id"], keep="last")


def _bool_to_int(value: Any) -> int:
    parsed = _safe_bool(value)
    if parsed is True:
        return 1
    return 0


def build_context_feature_bridge(
    daily_context_path: str = "data/daily_game_context.csv",
    weather_path: str = "data/weather_context.csv",
    team_form_path: str = "data/team_form_context.csv",
    pitcher_adv_path: str = "data/pitcher_advanced_context.csv",
    output_path: Optional[str] = "data/context_feature_bridge.csv",
) -> pd.DataFrame:
    captured_at = _current_utc_iso()
    context_path = Path(daily_context_path)

    if not context_path.exists():
        return _empty_output(output_path)

    try:
        context_frame = pd.read_csv(context_path)
    except Exception:
        return _empty_output(output_path)

    if context_frame.empty:
        return _empty_output(output_path)

    base_columns = ["game_id", "game_date", "start_time", "captured_at", "home_team", "away_team"]
    for column in base_columns:
        if column not in context_frame.columns:
            context_frame[column] = ""

    missing_source_columns = [
        column for column in SOURCE_COLUMNS if column not in context_frame.columns
    ]

    for column in SOURCE_COLUMNS:
        if column not in context_frame.columns:
            context_frame[column] = ""

    context_frame = context_frame.dropna(subset=["game_id"]).copy()
    if context_frame.empty:
        return _empty_output(output_path)

    latest_frame = _latest_context_rows(context_frame)
    rows: List[dict[str, Any]] = []

    for _, row in latest_frame.iterrows():
        reasons: List[str] = []
        if missing_source_columns:
            reasons.append("missing_source_columns=" + ",".join(missing_source_columns))

        home_pitches_1d = _safe_float(row.get("home_bullpen_pitches_last_1d"))
        away_pitches_1d = _safe_float(row.get("away_bullpen_pitches_last_1d"))
        home_pitches_3d = _safe_float(row.get("home_bullpen_pitches_last_3d"))
        away_pitches_3d = _safe_float(row.get("away_bullpen_pitches_last_3d"))

        home_fatigue = _safe_float(row.get("home_bullpen_fatigue_score"))
        away_fatigue = _safe_float(row.get("away_bullpen_fatigue_score"))

        home_closer_risk = _safe_float(row.get("home_closer_risk_score"))
        away_closer_risk = _safe_float(row.get("away_closer_risk_score"))

        if home_pitches_3d is not None and away_pitches_3d is not None:
            bullpen_ip_diff = round(away_pitches_3d - home_pitches_3d, 4)
        else:
            bullpen_ip_diff = 0.0
            reasons.append("bullpen_pitches_last_3d_missing")

        if home_fatigue is not None and away_fatigue is not None:
            bullpen_availability_diff = round(away_fatigue - home_fatigue, 4)
            availability_source = "fatigue_score"
        elif home_closer_risk is not None and away_closer_risk is not None:
            bullpen_availability_diff = round(away_closer_risk - home_closer_risk, 4)
            availability_source = "closer_risk_score"
            reasons.append("fatigue_missing_used_closer_risk")
        else:
            bullpen_availability_diff = 0.0
            availability_source = "missing"
            reasons.append("bullpen_availability_sources_missing")

        home_lineup_available = _bool_to_int(row.get("home_projected_lineup_available"))
        away_lineup_available = _bool_to_int(row.get("away_projected_lineup_available"))
        lineup_projection_available_diff = home_lineup_available - away_lineup_available

        status = "ok"
        if missing_source_columns or availability_source == "missing":
            status = "partial"

        if availability_source == "fatigue_score":
            reasons.append("bullpen_availability_diff=away_fatigue-home_fatigue")
        elif availability_source == "closer_risk_score":
            reasons.append("bullpen_availability_diff=away_closer_risk-home_closer_risk")

        rows.append(
            {
                "game_id": _safe_str(row.get("game_id")),
                "game_date": _safe_str(row.get("game_date"))[:10],
                "start_time": _safe_str(row.get("start_time")),
                "captured_at": _safe_str(row.get("captured_at")),
                "home_team": _safe_str(row.get("home_team")),
                "away_team": _safe_str(row.get("away_team")),
                "bullpen_ip_diff": bullpen_ip_diff,
                "bullpen_availability_diff": bullpen_availability_diff,
                "home_bullpen_fatigue_score": home_fatigue,
                "away_bullpen_fatigue_score": away_fatigue,
                "home_bullpen_pitches_last_1d": home_pitches_1d,
                "away_bullpen_pitches_last_1d": away_pitches_1d,
                "home_bullpen_pitches_last_3d": home_pitches_3d,
                "away_bullpen_pitches_last_3d": away_pitches_3d,
                "home_closer_risk_score": home_closer_risk,
                "away_closer_risk_score": away_closer_risk,
                "lineup_projection_available_diff": lineup_projection_available_diff,
                "context_bridge_source_status": status,
                "context_bridge_reason": "; ".join(reasons) if reasons else "bridge from daily_context",
                "context_bridge_captured_at": captured_at,
            }
        )

    output_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_frame = _clean_dataframe(output_frame)

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(destination, index=False)

    return output_frame


if __name__ == "__main__":
    df = build_context_feature_bridge()
    status_counts = (
        df["context_bridge_source_status"].value_counts().to_dict()
        if not df.empty and "context_bridge_source_status" in df.columns
        else {}
    )
    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "status_counts": {str(key): int(value) for key, value in status_counts.items()},
                "output_path": "data/context_feature_bridge.csv",
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
