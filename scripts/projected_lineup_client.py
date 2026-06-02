from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE = "projected_lineup_v1"

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_projected_lineup_available",
    "away_projected_lineup_available",
    "home_projected_lineup_status",
    "away_projected_lineup_status",
    "home_projected_player_ids_json",
    "away_projected_player_ids_json",
    "home_projected_player_count",
    "away_projected_player_count",
    "home_projected_top3_player_ids",
    "away_projected_top3_player_ids",
    "home_projected_lineup_reason",
    "away_projected_lineup_reason",
    "projected_lineup_source",
    "projected_lineup_captured_at",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read CSV into DataFrame."""
    try:
        frame = pd.read_csv(path)
        return frame, ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as exc:
        return None, f"Error reading CSV {path}: {exc}"


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


def _safe_int(value: Any) -> Optional[int]:
    """Convert value to int safely."""
    try:
        if value is None:
            return None

        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None

        return int(numeric)

    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    """Return clean string."""
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


def _current_utc_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _json_dumps_list(values: List[int]) -> str:
    """JSON dump list of ints."""
    return json.dumps([int(value) for value in values], ensure_ascii=True)


def _parse_id_token(value: Any) -> Optional[int]:
    """Parse a single player ID token, supporting int, float, and numeric strings."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        numeric = float(text)
        if math.isnan(numeric) or math.isinf(numeric):
            return None

        integer = int(numeric)
        if integer <= 0:
            return None

        return integer

    except (TypeError, ValueError):
        return None


def _parse_id_list(value: Any) -> List[int]:
    """
    Parse player IDs from:
    - JSON list string: "[1, 2, 3]"
    - CSV string: "1,2,3"
    - Python list
    - pandas NaN / blank values
    """
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    raw_items: List[Any] = []

    if isinstance(value, list):
        raw_items = value

    elif isinstance(value, str):
        text = value.strip()

        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = [parsed]
        except json.JSONDecodeError:
            raw_items = [
                item.strip()
                for item in text.replace(";", ",").split(",")
                if item.strip()
            ]

    else:
        raw_items = [value]

    ids: List[int] = []
    seen = set()

    for item in raw_items:
        player_id = _parse_id_token(item)
        if player_id is None:
            continue
        if player_id in seen:
            continue

        ids.append(player_id)
        seen.add(player_id)

    return ids


def _latest_rows_per_game(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep latest row per game_id using captured_at when available."""
    if frame.empty or "game_id" not in frame.columns:
        return frame

    working = frame.copy()

    if "captured_at" not in working.columns:
        return working.drop_duplicates(subset=["game_id"], keep="last")

    working["captured_at_dt"] = pd.to_datetime(
        working["captured_at"],
        errors="coerce",
        utc=True,
    )

    if working["captured_at_dt"].notna().sum() == 0:
        return working.drop_duplicates(subset=["game_id"], keep="last")

    working = working.dropna(subset=["game_id"])
    idx = working.groupby("game_id")["captured_at_dt"].idxmax()
    latest = working.loc[idx].copy()
    latest = latest.drop(columns=["captured_at_dt"], errors="ignore")

    return latest


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _evaluate_projected_lineup_side(
    *,
    lineup_confirmed: Any,
    lineup_player_count: Any,
    lineup_ids_json: Any,
    top3_player_ids: Any,
) -> Dict[str, Any]:
    """Evaluate projected lineup status for one team."""
    confirmed = _safe_bool(lineup_confirmed)
    confirmed_count = _safe_int(lineup_player_count) or 0
    confirmed_ids = _parse_id_list(lineup_ids_json)
    top3_ids = _parse_id_list(top3_player_ids)

    if confirmed is True and confirmed_count >= 9:
        player_ids = confirmed_ids

        if len(player_ids) < confirmed_count:
            reason = (
                "Confirmed lineup flag/count available, but confirmed player IDs are incomplete"
            )
        else:
            reason = "Confirmed lineup already available"

        return {
            "projected_available": True,
            "status": "confirmed_lineup_available",
            "player_ids": player_ids,
            "top3_ids": player_ids[:3],
            "reason": reason,
        }

    if len(top3_ids) >= 3:
        return {
            "projected_available": False,
            "status": "projected_top3_available",
            "player_ids": top3_ids,
            "top3_ids": top3_ids[:3],
            "reason": "Only top3 projected bats available; full projected lineup unavailable",
        }

    return {
        "projected_available": False,
        "status": "unavailable",
        "player_ids": [],
        "top3_ids": [],
        "reason": "No confirmed or projected lineup source available",
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_projected_lineup_context(
    daily_context_path: str = "data/daily_game_context.csv",
    output_path: Optional[str] = "data/projected_lineup_context.csv",
) -> pd.DataFrame:
    """Build conservative projected lineup context from daily game context."""
    captured_at = _current_utc_iso()

    frame, error = _safe_read_csv(Path(daily_context_path))
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    required_columns = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_lineup_confirmed",
        "away_lineup_confirmed",
        "home_lineup_player_count",
        "away_lineup_player_count",
        "home_lineup_player_ids_json",
        "away_lineup_player_ids_json",
        "home_top3_player_ids",
        "away_top3_player_ids",
    ]

    for column in required_columns:
        if column not in frame.columns:
            frame[column] = None

    latest = _latest_rows_per_game(frame)

    rows: List[Dict[str, Any]] = []

    for _, row in latest.iterrows():
        home_eval = _evaluate_projected_lineup_side(
            lineup_confirmed=row.get("home_lineup_confirmed"),
            lineup_player_count=row.get("home_lineup_player_count"),
            lineup_ids_json=row.get("home_lineup_player_ids_json"),
            top3_player_ids=row.get("home_top3_player_ids"),
        )

        away_eval = _evaluate_projected_lineup_side(
            lineup_confirmed=row.get("away_lineup_confirmed"),
            lineup_player_count=row.get("away_lineup_player_count"),
            lineup_ids_json=row.get("away_lineup_player_ids_json"),
            top3_player_ids=row.get("away_top3_player_ids"),
        )

        row_dict = {
            "game_id": _safe_str(row.get("game_id")),
            "game_date": _safe_str(row.get("game_date")),
            "home_team": _safe_str(row.get("home_team")),
            "away_team": _safe_str(row.get("away_team")),
            "home_lineup_confirmed": _safe_bool(row.get("home_lineup_confirmed")) is True,
            "away_lineup_confirmed": _safe_bool(row.get("away_lineup_confirmed")) is True,
            "home_projected_lineup_available": bool(home_eval["projected_available"]),
            "away_projected_lineup_available": bool(away_eval["projected_available"]),
            "home_projected_lineup_status": str(home_eval["status"]),
            "away_projected_lineup_status": str(away_eval["status"]),
            "home_projected_player_ids_json": _json_dumps_list(home_eval["player_ids"]),
            "away_projected_player_ids_json": _json_dumps_list(away_eval["player_ids"]),
            "home_projected_player_count": int(len(home_eval["player_ids"])),
            "away_projected_player_count": int(len(away_eval["player_ids"])),
            "home_projected_top3_player_ids": ",".join(
                str(player_id) for player_id in home_eval["top3_ids"]
            ),
            "away_projected_top3_player_ids": ",".join(
                str(player_id) for player_id in away_eval["top3_ids"]
            ),
            "home_projected_lineup_reason": str(home_eval["reason"]),
            "away_projected_lineup_reason": str(away_eval["reason"]),
            "projected_lineup_source": SOURCE,
            "projected_lineup_captured_at": captured_at,
        }

        rows.append(row_dict)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_file = "data/projected_lineup_context.csv"
    output_frame = build_projected_lineup_context(output_path=output_file)

    summary = {
        "rows": int(len(output_frame)),
        "home_confirmed_available": int(
            (output_frame["home_projected_lineup_status"] == "confirmed_lineup_available").sum()
        ),
        "away_confirmed_available": int(
            (output_frame["away_projected_lineup_status"] == "confirmed_lineup_available").sum()
        ),
        "home_top3_available": int(
            (output_frame["home_projected_lineup_status"] == "projected_top3_available").sum()
        ),
        "away_top3_available": int(
            (output_frame["away_projected_lineup_status"] == "projected_top3_available").sum()
        ),
        "home_full_projected_available": int(
            output_frame["home_projected_lineup_available"].sum()
        ),
        "away_full_projected_available": int(
            output_frame["away_projected_lineup_available"].sum()
        ),
        "output_path": output_file,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
