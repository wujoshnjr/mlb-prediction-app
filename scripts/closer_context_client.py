from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE = "closer_context_v1"

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "home_closer_available_known",
    "away_closer_available_known",
    "home_closer_available",
    "away_closer_available",
    "home_closer_status",
    "away_closer_status",
    "home_closer_risk_score",
    "away_closer_risk_score",
    "home_closer_reason",
    "away_closer_reason",
    "closer_context_source",
    "closer_context_captured_at",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, returning None for NaN/inf or unparseable values."""
    try:
        if value is None:
            return None

        numeric_value = float(value)

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None

        return numeric_value

    except (ValueError, TypeError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    """Interpret a value as bool, returning None if indeterminate."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"true", "1", "yes", "y"}:
            return True

        if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
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


def _safe_str(value: Any) -> str:
    """Return a clean string. Null-like values become empty string."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def _current_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read a CSV file into a DataFrame, return (df, error_message)."""
    try:
        frame = pd.read_csv(path)
        return frame, ""

    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


def _get_row_value(row: pd.Series, *columns: str) -> Any:
    """Return the first non-null value from a row using candidate column names."""
    for column in columns:
        if column in row.index:
            value = row.get(column)
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value

    return ""


# ---------------------------------------------------------------------------
# Closer evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_closer_side(
    bullpen_data_available: Any,
    pitches_last_1d: Any,
    pitches_last_3d: Any,
    fatigue_score: Any,
) -> Dict[str, Any]:
    """
    Evaluate closer availability for one team based on bullpen usage data.

    Returns a dict with:
    known, available, status, risk_score, reason.
    """
    data_available = _safe_bool(bullpen_data_available)

    pitches_1d = _safe_float(pitches_last_1d)
    pitches_3d = _safe_float(pitches_last_3d)
    fatigue = _safe_float(fatigue_score)

    if data_available is not True or (
        pitches_1d is None
        and pitches_3d is None
        and fatigue is None
    ):
        return {
            "known": False,
            "available": False,
            "status": "unknown",
            "risk_score": 1.0,
            "reason": "Bullpen usage data unavailable",
        }

    risk = 0.0
    reason_parts = []

    if pitches_1d is not None:
        if pitches_1d >= 45:
            risk += 0.45
        elif pitches_1d >= 30:
            risk += 0.30
        elif pitches_1d >= 20:
            risk += 0.15

        reason_parts.append(f"last_1d={pitches_1d:.0f}")
    else:
        reason_parts.append("last_1d=NA")

    if pitches_3d is not None:
        if pitches_3d >= 150:
            risk += 0.35
        elif pitches_3d >= 100:
            risk += 0.20
        elif pitches_3d >= 70:
            risk += 0.10

        reason_parts.append(f"last_3d={pitches_3d:.0f}")
    else:
        reason_parts.append("last_3d=NA")

    if fatigue is not None:
        if fatigue >= 120:
            risk += 0.35
        elif fatigue >= 90:
            risk += 0.20
        elif fatigue >= 60:
            risk += 0.10

        reason_parts.append(f"fatigue={fatigue:.0f}")
    else:
        reason_parts.append("fatigue=NA")

    risk = round(min(risk, 1.0), 2)

    if risk >= 0.70:
        status = "high_fatigue_risk"
    elif risk >= 0.40:
        status = "fatigue_risk"
    else:
        status = "available"

    available = risk < 0.70

    reason = (
        f"{', '.join(reason_parts)}, "
        f"risk_score={risk:.2f}, "
        f"status={status}"
    )

    return {
        "known": True,
        "available": bool(available),
        "status": status,
        "risk_score": risk,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_closer_context(
    daily_context_path: str = "data/daily_game_context.csv",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build closer availability context from daily game context.

    Parameters
    ----------
    daily_context_path:
        Path to daily_game_context.csv.
    output_path:
        If provided, write the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        DataFrame with closer context columns.
    """
    captured_at = _current_utc_iso()

    context_frame, _error = _safe_read_csv(Path(daily_context_path))
    if context_frame is None or context_frame.empty:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_file, index=False)

        return result

    rows = []

    for _, row in context_frame.iterrows():
        home_result = _evaluate_closer_side(
            bullpen_data_available=_get_row_value(
                row,
                "home_bullpen_data_available",
            ),
            pitches_last_1d=_get_row_value(
                row,
                "home_bullpen_pitches_last_1d",
            ),
            pitches_last_3d=_get_row_value(
                row,
                "home_bullpen_pitches_last_3d",
            ),
            fatigue_score=_get_row_value(
                row,
                "home_bullpen_fatigue_score",
            ),
        )

        away_result = _evaluate_closer_side(
            bullpen_data_available=_get_row_value(
                row,
                "away_bullpen_data_available",
            ),
            pitches_last_1d=_get_row_value(
                row,
                "away_bullpen_pitches_last_1d",
            ),
            pitches_last_3d=_get_row_value(
                row,
                "away_bullpen_pitches_last_3d",
            ),
            fatigue_score=_get_row_value(
                row,
                "away_bullpen_fatigue_score",
            ),
        )

        home_team = _get_row_value(row, "home_team_name", "home_team")
        away_team = _get_row_value(row, "away_team_name", "away_team")

        row_dict = {
            "game_id": _safe_str(_get_row_value(row, "game_id")),
            "game_date": _safe_str(_get_row_value(row, "game_date")),
            "home_team_id": _safe_str(_get_row_value(row, "home_team_id")),
            "away_team_id": _safe_str(_get_row_value(row, "away_team_id")),
            "home_team_name": _safe_str(home_team),
            "away_team_name": _safe_str(away_team),
            "home_closer_available_known": bool(home_result["known"]),
            "away_closer_available_known": bool(away_result["known"]),
            "home_closer_available": bool(home_result["available"]),
            "away_closer_available": bool(away_result["available"]),
            "home_closer_status": _safe_str(home_result["status"]),
            "away_closer_status": _safe_str(away_result["status"]),
            "home_closer_risk_score": float(home_result["risk_score"]),
            "away_closer_risk_score": float(away_result["risk_score"]),
            "home_closer_reason": _safe_str(home_result["reason"]),
            "away_closer_reason": _safe_str(away_result["reason"]),
            "closer_context_source": SOURCE,
            "closer_context_captured_at": captured_at,
        }

        rows.append(row_dict)

    result_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Avoid NaN/inf in CSV output.
    result_frame = result_frame.fillna("")

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(output_file, index=False)

    return result_frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_file = "data/closer_context.csv"

    closer_frame = build_closer_context(
        daily_context_path="data/daily_game_context.csv",
        output_path=output_file,
    )

    if closer_frame.empty:
        summary = {
            "rows": 0,
            "known_home": 0,
            "known_away": 0,
            "available_home": 0,
            "available_away": 0,
            "output_path": output_file,
        }
    else:
        summary = {
            "rows": int(len(closer_frame)),
            "known_home": int(closer_frame["home_closer_available_known"].sum()),
            "known_away": int(closer_frame["away_closer_available_known"].sum()),
            "available_home": int(closer_frame["home_closer_available"].sum()),
            "available_away": int(closer_frame["away_closer_available"].sum()),
            "output_path": output_file,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
