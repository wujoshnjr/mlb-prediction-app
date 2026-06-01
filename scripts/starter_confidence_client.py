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

SOURCE = "starter_confidence_v1"

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "home_team",
    "away_team",
    "home_starter_status",
    "away_starter_status",
    "home_starter_confidence",
    "away_starter_confidence",
    "home_starter_confidence_score",
    "away_starter_confidence_score",
    "home_starter_reason",
    "away_starter_reason",
    "home_probable_pitcher_id",
    "away_probable_pitcher_id",
    "home_probable_pitcher_name",
    "away_probable_pitcher_name",
    "home_starting_pitcher_id",
    "away_starting_pitcher_id",
    "home_starting_pitcher_name",
    "away_starting_pitcher_name",
    "home_starting_pitcher_confirmed",
    "away_starting_pitcher_confirmed",
    "starter_confidence_source",
    "starter_confidence_captured_at",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read CSV into a DataFrame, return (df, error) tuple."""
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


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
    """Interpret value as bool, returning None if ambiguous."""
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
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a string or datetime into a UTC-aware datetime, or None on failure."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    text = _safe_str(value)
    if not text:
        return None

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (ValueError, TypeError):
        return None


def _hours_until_start(captured_at: Any, start_time: Any) -> Optional[float]:
    """Compute hours from captured_at to start_time. Returns None if unparseable."""
    captured_dt = _parse_datetime(captured_at)
    start_dt = _parse_datetime(start_time)

    if captured_dt is None or start_dt is None:
        return None

    delta = start_dt - captured_dt
    return delta.total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Starter confidence evaluation
# ---------------------------------------------------------------------------

def _evaluate_starter_side(
    probable_pitcher_id: Any,
    probable_pitcher_name: Any,
    starting_pitcher_id: Any,
    starting_pitcher_name: Any,
    starting_pitcher_confirmed: Any,
    season_era: Any,
    season_fip: Any,
    game_feed_available: Any,
    captured_at: Any,
    start_time: Any,
) -> Dict[str, Any]:
    """
    Evaluate starter confidence for one team.

    Returns:
        status, confidence, score, reason.
    """
    confirmed = _safe_bool(starting_pitcher_confirmed)

    probable_id = _safe_str(probable_pitcher_id)
    probable_name = _safe_str(probable_pitcher_name)
    starter_id = _safe_str(starting_pitcher_id)
    starter_name = _safe_str(starting_pitcher_name)

    if confirmed is True:
        reason_parts = ["Starting pitcher confirmed by game feed"]

        if starter_id:
            reason_parts.append(f"starter_id={starter_id}")
        if starter_name:
            reason_parts.append(f"starter_name={starter_name}")

        return {
            "status": "confirmed",
            "confidence": True,
            "score": 1.0,
            "reason": "; ".join(reason_parts),
        }

    if probable_id:
        score = 0.55
        reason_parts = [f"probable_id={probable_id}"]

        if probable_name:
            score += 0.10
            reason_parts.append(f"probable_name={probable_name}")
        else:
            reason_parts.append("probable_name_missing")

        era_value = _safe_float(season_era)
        if era_value is not None:
            score += 0.10
            reason_parts.append("ERA_available")
        else:
            reason_parts.append("ERA_missing")

        fip_value = _safe_float(season_fip)
        if fip_value is not None:
            score += 0.10
            reason_parts.append("FIP_available")
        else:
            reason_parts.append("FIP_missing")

        feed_available = _safe_bool(game_feed_available)
        if feed_available is True:
            score += 0.05
            reason_parts.append("game_feed_available")
        else:
            reason_parts.append("game_feed_not_available")

        hours_until_start = _hours_until_start(captured_at, start_time)
        if hours_until_start is not None:
            if hours_until_start <= 3.0:
                score += 0.10
                reason_parts.append(f"captured_within_3h={hours_until_start:.2f}")
            elif hours_until_start <= 6.0:
                score += 0.05
                reason_parts.append(f"captured_within_6h={hours_until_start:.2f}")
            else:
                reason_parts.append(f"hours_until_start={hours_until_start:.2f}")
        else:
            reason_parts.append("hours_until_start_unknown")

        score = round(min(score, 0.95), 2)

        if score >= 0.80:
            status = "high_confidence_probable"
        elif score >= 0.65:
            status = "medium_confidence_probable"
        else:
            status = "low_confidence_probable"

        confidence = score >= 0.80

        reason = (
            f"score={score:.2f}, "
            f"status={status}, "
            f"reasons={', '.join(reason_parts)}"
        )

        return {
            "status": status,
            "confidence": bool(confidence),
            "score": score,
            "reason": reason,
        }

    return {
        "status": "unknown",
        "confidence": False,
        "score": 0.0,
        "reason": "No confirmed or probable starter available",
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_starter_confidence_context(
    daily_context_path: str = "data/daily_game_context.csv",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build starter confidence context from daily game context.

    Parameters
    ----------
    daily_context_path:
        Path to data/daily_game_context.csv.
    output_path:
        If provided, write resulting CSV to this path.

    Returns
    -------
    pd.DataFrame
        Starter confidence context DataFrame.
    """
    starter_confidence_captured_at = _current_utc_iso()

    context_frame, _error = _safe_read_csv(Path(daily_context_path))

    if context_frame is None or context_frame.empty:
        result_frame = pd.DataFrame(columns=OUTPUT_COLUMNS)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result_frame.to_csv(output_file, index=False)

        return result_frame

    rows = []

    for _, row in context_frame.iterrows():
        game_id = _safe_str(row.get("game_id"))
        game_date = _safe_str(row.get("game_date"))
        start_time = row.get("start_time")
        captured_at = row.get("captured_at")
        home_team = _safe_str(row.get("home_team"))
        away_team = _safe_str(row.get("away_team"))
        game_feed_available = row.get("game_feed_available")

        home_result = _evaluate_starter_side(
            probable_pitcher_id=row.get("home_probable_pitcher_id"),
            probable_pitcher_name=row.get("home_probable_pitcher_name"),
            starting_pitcher_id=row.get("home_starting_pitcher_id"),
            starting_pitcher_name=row.get("home_starting_pitcher_name"),
            starting_pitcher_confirmed=row.get("home_starting_pitcher_confirmed"),
            season_era=row.get("home_sp_season_era"),
            season_fip=row.get("home_sp_season_fip"),
            game_feed_available=game_feed_available,
            captured_at=captured_at,
            start_time=start_time,
        )

        away_result = _evaluate_starter_side(
            probable_pitcher_id=row.get("away_probable_pitcher_id"),
            probable_pitcher_name=row.get("away_probable_pitcher_name"),
            starting_pitcher_id=row.get("away_starting_pitcher_id"),
            starting_pitcher_name=row.get("away_starting_pitcher_name"),
            starting_pitcher_confirmed=row.get("away_starting_pitcher_confirmed"),
            season_era=row.get("away_sp_season_era"),
            season_fip=row.get("away_sp_season_fip"),
            game_feed_available=game_feed_available,
            captured_at=captured_at,
            start_time=start_time,
        )

        home_confirmed = _safe_bool(row.get("home_starting_pitcher_confirmed"))
        away_confirmed = _safe_bool(row.get("away_starting_pitcher_confirmed"))

        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "start_time": _safe_str(start_time),
                "captured_at": _safe_str(captured_at),
                "home_team": home_team,
                "away_team": away_team,
                "home_starter_status": _safe_str(home_result["status"]),
                "away_starter_status": _safe_str(away_result["status"]),
                "home_starter_confidence": bool(home_result["confidence"]),
                "away_starter_confidence": bool(away_result["confidence"]),
                "home_starter_confidence_score": float(home_result["score"]),
                "away_starter_confidence_score": float(away_result["score"]),
                "home_starter_reason": _safe_str(home_result["reason"]),
                "away_starter_reason": _safe_str(away_result["reason"]),
                "home_probable_pitcher_id": _safe_str(
                    row.get("home_probable_pitcher_id")
                ),
                "away_probable_pitcher_id": _safe_str(
                    row.get("away_probable_pitcher_id")
                ),
                "home_probable_pitcher_name": _safe_str(
                    row.get("home_probable_pitcher_name")
                ),
                "away_probable_pitcher_name": _safe_str(
                    row.get("away_probable_pitcher_name")
                ),
                "home_starting_pitcher_id": _safe_str(
                    row.get("home_starting_pitcher_id")
                ),
                "away_starting_pitcher_id": _safe_str(
                    row.get("away_starting_pitcher_id")
                ),
                "home_starting_pitcher_name": _safe_str(
                    row.get("home_starting_pitcher_name")
                ),
                "away_starting_pitcher_name": _safe_str(
                    row.get("away_starting_pitcher_name")
                ),
                "home_starting_pitcher_confirmed": (
                    bool(home_confirmed) if home_confirmed is not None else False
                ),
                "away_starting_pitcher_confirmed": (
                    bool(away_confirmed) if away_confirmed is not None else False
                ),
                "starter_confidence_source": SOURCE,
                "starter_confidence_captured_at": starter_confidence_captured_at,
            }
        )

    result_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
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
    output_file = "data/starter_confidence_context.csv"

    starter_frame = build_starter_confidence_context(
        daily_context_path="data/daily_game_context.csv",
        output_path=output_file,
    )

    if starter_frame.empty:
        summary = {
            "rows": 0,
            "home_confirmed": 0,
            "away_confirmed": 0,
            "home_high_confidence": 0,
            "away_high_confidence": 0,
            "output_path": output_file,
        }
    else:
        home_confirmed = int(
            (starter_frame["home_starter_status"] == "confirmed").sum()
        )
        away_confirmed = int(
            (starter_frame["away_starter_status"] == "confirmed").sum()
        )
        home_high = int(
            (
                starter_frame["home_starter_status"]
                == "high_confidence_probable"
            ).sum()
        )
        away_high = int(
            (
                starter_frame["away_starter_status"]
                == "high_confidence_probable"
            ).sum()
        )

        summary = {
            "rows": int(len(starter_frame)),
            "home_confirmed": home_confirmed,
            "away_confirmed": away_confirmed,
            "home_high_confidence": home_high,
            "away_high_confidence": away_high,
            "output_path": output_file,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
