from __future__ import annotations

import json
import os
from typing import Any

from scripts.savant_top3_context_client import (
    DEFAULT_LOOKBACK_DAYS,
    build_savant_top3_context,
)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return str(value)


def main() -> int:
    """Backfill Savant/top-3 context for all known games.

    The normal daily client used to write only today's rows. That made the
    source audit show zero overlap with finalized training samples. This wrapper
    intentionally runs build_savant_top3_context with as_of_date=None so the
    output contains the latest context row for every game_id present in
    data/daily_game_context.csv.

    For stability, the default historical backfill uses MLB Stats proxy summaries
    after max_unique_players=0. Raising SAVANT_BACKFILL_MAX_UNIQUE_PLAYERS lets a
    run attempt real Baseball Savant CSV for the first N unique batters, but that
    can be slower and more timeout-prone.
    """
    errors: list[str] = []
    lookback_days = _int_env("SAVANT_BACKFILL_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)
    timeout = _int_env("SAVANT_BACKFILL_TIMEOUT", 12)
    sleep_seconds = _float_env("SAVANT_BACKFILL_SLEEP_SECONDS", 0.0)
    max_unique_players = _int_env("SAVANT_BACKFILL_MAX_UNIQUE_PLAYERS", 0)

    summary = build_savant_top3_context(
        daily_context_path="data/daily_game_context.csv",
        output_path="data/savant_top3_context.csv",
        as_of_date=None,
        lookback_days=lookback_days,
        errors=errors,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        max_unique_players=max_unique_players,
    )
    summary["mode"] = "historical_backfill_all_known_games"
    summary["savant_backfill_max_unique_players"] = max_unique_players
    summary["live_betting_allowed"] = False
    summary["automated_wagering_allowed"] = False
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
