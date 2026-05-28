from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_VERSION = "baseline_v2_clean"
MARKET_ODDS_HISTORY_FILE = Path("data/market_odds_history.csv")
SUPPORTED_MARKETS = {"moneyline", "spread", "total"}

COLUMNS = [
    "odds_snapshot_id",
    "pipeline_version",
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "minutes_to_start",
    "home_team",
    "away_team",
    "bookmaker",
    "bookmaker_key",
    "bookmaker_last_update",
    "market",
    "side",
    "line",
    "odds",
    "odds_quality_status",
    "suspicious_odds_reason",
    "starting_pitcher_confirmed",
    "lineup_confirmed",
    "is_pregame",
    "is_opening_snapshot",
    "is_closing_snapshot",
    "settled_at",
    "home_win",
    "home_score",
    "away_score",
]


def parse_utc_datetime(value: Any) -> datetime | None:
    """Parse an ISO timestamp string into a timezone-aware UTC datetime."""
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (TypeError, ValueError):
        return None


def _to_utc_datetime(value: Any) -> datetime | None:
    """Convert a datetime or ISO timestamp string to UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    return parse_utc_datetime(value)


def _utc_iso(value: datetime | None) -> str:
    """Format UTC datetime with a trailing Z."""
    if value is None:
        return ""

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def _safe_decimal_odds(value: Any) -> float | None:
    """Return valid decimal odds greater than 1.0 or None."""
    parsed = _safe_float(value)

    if parsed is None or parsed <= 1.0:
        return None

    return parsed


def _normalise_line(value: Any) -> float | None:
    """Return a finite market line, including zero or negative values."""
    return _safe_float(value)


def _bool_series(series: pd.Series) -> pd.Series:
    """Normalise bool-like CSV values into booleans."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def _format_number(value: float | None) -> str:
    """Format a numeric value consistently for deterministic IDs."""
    if value is None:
        return ""

    return format(float(value), ".12g")


def _make_snapshot_id(
    *,
    game_id: str,
    captured_at: datetime,
    bookmaker_key: str,
    market: str,
    side: str,
    line: float | None,
    odds: float,
) -> str:
    """Create a deterministic snapshot identifier for duplicate prevention."""
    raw_value = "|".join(
        [
            PIPELINE_VERSION,
            game_id,
            _utc_iso(captured_at),
            bookmaker_key,
            market,
            side,
            _format_number(line),
            _format_number(odds),
        ]
    )

    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def ensure_market_odds_store() -> None:
    """Create the canonical history CSV if it does not exist."""
    MARKET_ODDS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not MARKET_ODDS_HISTORY_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(
            MARKET_ODDS_HISTORY_FILE,
            index=False,
            encoding="utf-8",
        )


def _load_market_odds_history() -> pd.DataFrame:
    """Read canonical odds history and normalise required data types."""
    if not MARKET_ODDS_HISTORY_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        frame = pd.read_csv(MARKET_ODDS_HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[COLUMNS].copy()

    if frame.empty:
        return frame

    frame["game_id"] = frame["game_id"].astype(str)

    for column in (
        "is_pregame",
        "is_opening_snapshot",
        "is_closing_snapshot",
    ):
        frame[column] = _bool_series(frame[column])

    for column in (
        "minutes_to_start",
        "line",
        "odds",
        "home_win",
        "home_score",
        "away_score",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _validation_errors(
    *,
    game_id: Any,
    start_time: Any,
    captured_at: Any,
    bookmaker_quotes: Any,
) -> list[str]:
    """Return input validation errors for one game odds capture."""
    errors: list[str] = []

    game_id_text = str(game_id).strip() if game_id is not None else ""
    if not game_id_text:
        errors.append("Missing game_id.")

    if _to_utc_datetime(start_time) is None:
        errors.append("Invalid start_time.")

    if _to_utc_datetime(captured_at) is None:
        errors.append("Invalid captured_at.")

    if not isinstance(bookmaker_quotes, list):
        errors.append("bookmaker_quotes must be a list.")

    return errors


def _make_row(
    *,
    game_id: str,
    game_date: str,
    start_time: datetime,
    captured_at: datetime,
    home_team: str,
    away_team: str,
    bookmaker: str,
    bookmaker_key: str,
    bookmaker_last_update: datetime | None,
    market: str,
    side: str,
    line: float | None,
    odds: float,
    odds_quality_status: str,
    suspicious_odds_reason: str,
    starting_pitcher_confirmed: bool | None,
    lineup_confirmed: bool | None,
) -> dict[str, Any]:
    """Build one canonical long-format market odds row."""
    minutes_to_start = round(
        (start_time - captured_at).total_seconds() / 60.0,
        3,
    )
    is_pregame = captured_at < start_time

    return {
        "odds_snapshot_id": _make_snapshot_id(
            game_id=game_id,
            captured_at=captured_at,
            bookmaker_key=bookmaker_key,
            market=market,
            side=side,
            line=line,
            odds=odds,
        ),
        "pipeline_version": PIPELINE_VERSION,
        "game_id": game_id,
        "game_date": str(game_date),
        "start_time": _utc_iso(start_time),
        "captured_at": _utc_iso(captured_at),
        "minutes_to_start": minutes_to_start,
        "home_team": str(home_team).strip(),
        "away_team": str(away_team).strip(),
        "bookmaker": str(bookmaker).strip(),
        "bookmaker_key": str(bookmaker_key).strip(),
        "bookmaker_last_update": _utc_iso(bookmaker_last_update),
        "market": market,
        "side": side,
        "line": line if line is not None else "",
        "odds": odds,
        "odds_quality_status": str(odds_quality_status).strip() or "UNAVAILABLE",
        "suspicious_odds_reason": str(suspicious_odds_reason).strip(),
        "starting_pitcher_confirmed": starting_pitcher_confirmed,
        "lineup_confirmed": lineup_confirmed,
        "is_pregame": is_pregame,
        "is_opening_snapshot": False,
        "is_closing_snapshot": False,
        "settled_at": "",
        "home_win": "",
        "home_score": "",
        "away_score": "",
    }


def build_market_odds_rows(
    *,
    game_id: Any,
    game_date: str,
    start_time: Any,
    captured_at: Any,
    home_team: str,
    away_team: str,
    bookmaker_quotes: list[dict[str, Any]],
    odds_quality_status: str = "UNAVAILABLE",
    suspicious_odds_reason: str = "",
    starting_pitcher_confirmed: bool | None = None,
    lineup_confirmed: bool | None = None,
) -> list[dict[str, Any]]:
    """Build full-game moneyline, spread and total rows from fetched quotes."""
    validation_errors = _validation_errors(
        game_id=game_id,
        start_time=start_time,
        captured_at=captured_at,
        bookmaker_quotes=bookmaker_quotes,
    )
    if validation_errors:
        return []

    game_id_text = str(game_id).strip()
    start_time_dt = _to_utc_datetime(start_time)
    captured_at_dt = _to_utc_datetime(captured_at)

    if start_time_dt is None or captured_at_dt is None:
        return []

    rows: list[dict[str, Any]] = []

    for quote in bookmaker_quotes:
        if not isinstance(quote, dict):
            continue

        bookmaker = str(quote.get("bookmaker", "")).strip() or "unknown"
        bookmaker_key = (
            str(quote.get("bookmaker_key", "")).strip()
            or bookmaker.lower().replace(" ", "_")
        )
        bookmaker_last_update = _to_utc_datetime(quote.get("last_update"))

        home_moneyline_odds = _safe_decimal_odds(
            quote.get("home_moneyline_odds")
        )
        away_moneyline_odds = _safe_decimal_odds(
            quote.get("away_moneyline_odds")
        )

        if home_moneyline_odds is not None:
            rows.append(
                _make_row(
                    game_id=game_id_text,
                    game_date=game_date,
                    start_time=start_time_dt,
                    captured_at=captured_at_dt,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=bookmaker,
                    bookmaker_key=bookmaker_key,
                    bookmaker_last_update=bookmaker_last_update,
                    market="moneyline",
                    side="home",
                    line=None,
                    odds=home_moneyline_odds,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=starting_pitcher_confirmed,
                    lineup_confirmed=lineup_confirmed,
                )
            )

        if away_moneyline_odds is not None:
            rows.append(
                _make_row(
                    game_id=game_id_text,
                    game_date=game_date,
                    start_time=start_time_dt,
                    captured_at=captured_at_dt,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=bookmaker,
                    bookmaker_key=bookmaker_key,
                    bookmaker_last_update=bookmaker_last_update,
                    market="moneyline",
                    side="away",
                    line=None,
                    odds=away_moneyline_odds,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=starting_pitcher_confirmed,
                    lineup_confirmed=lineup_confirmed,
                )
            )

        home_spread_line = _normalise_line(quote.get("home_spread_line"))
        home_spread_odds = _safe_decimal_odds(quote.get("home_spread_odds"))

        if home_spread_line is not None and home_spread_odds is not None:
            rows.append(
                _make_row(
                    game_id=game_id_text,
                    game_date=game_date,
                    start_time=start_time_dt,
                    captured_at=captured_at_dt,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=bookmaker,
                    bookmaker_key=bookmaker_key,
                    bookmaker_last_update=bookmaker_last_update,
                    market="spread",
                    side="home",
                    line=home_spread_line,
                    odds=home_spread_odds,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=starting_pitcher_confirmed,
                    lineup_confirmed=lineup_confirmed,
                )
            )

        away_spread_line = _normalise_line(quote.get("away_spread_line"))
        away_spread_odds = _safe_decimal_odds(quote.get("away_spread_odds"))

        if away_spread_line is not None and away_spread_odds is not None:
            rows.append(
                _make_row(
                    game_id=game_id_text,
                    game_date=game_date,
                    start_time=start_time_dt,
                    captured_at=captured_at_dt,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=bookmaker,
                    bookmaker_key=bookmaker_key,
                    bookmaker_last_update=bookmaker_last_update,
                    market="spread",
                    side="away",
                    line=away_spread_line,
                    odds=away_spread_odds,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=starting_pitcher_confirmed,
                    lineup_confirmed=lineup_confirmed,
                )
            )

        total_line = _normalise_line(quote.get("total_line"))
        over_odds = _safe_decimal_odds(quote.get("over_odds"))
        under_odds = _safe_decimal_odds(quote.get("under_odds"))

        if total_line is not None and over_odds is not None:
            rows.append(
                _make_row(
                    game_id=game_id_text,
                    game_date=game_date,
                    start_time=start_time_dt,
                    captured_at=captured_at_dt,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=bookmaker,
                    bookmaker_key=bookmaker_key,
                    bookmaker_last_update=bookmaker_last_update,
                    market="total",
                    side="over",
                    line=total_line,
                    odds=over_odds,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=starting_pitcher_confirmed,
                    lineup_confirmed=lineup_confirmed,
                )
            )

        if total_line is not None and under_odds is not None:
            rows.append(
                _make_row(
                    game_id=game_id_text,
                    game_date=game_date,
                    start_time=start_time_dt,
                    captured_at=captured_at_dt,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=bookmaker,
                    bookmaker_key=bookmaker_key,
                    bookmaker_last_update=bookmaker_last_update,
                    market="total",
                    side="under",
                    line=total_line,
                    odds=under_odds,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=starting_pitcher_confirmed,
                    lineup_confirmed=lineup_confirmed,
                )
            )

    return rows


def append_market_odds_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Append new odds rows while preserving existing history."""
    summary: dict[str, Any] = {
        "received": len(rows),
        "inserted": 0,
        "duplicates": 0,
        "stored_rows": 0,
        "errors": [],
    }

    try:
        ensure_market_odds_store()
        existing = _load_market_odds_history()

        if not rows:
            summary["stored_rows"] = int(len(existing))
            return summary

        new_rows = pd.DataFrame(rows)

        for column in COLUMNS:
            if column not in new_rows.columns:
                new_rows[column] = ""

        new_rows = new_rows[COLUMNS]

        existing_ids = set(
            existing["odds_snapshot_id"].dropna().astype(str).tolist()
        )
        incoming_ids: set[str] = set()
        accepted_rows: list[dict[str, Any]] = []

        for _, row in new_rows.iterrows():
            snapshot_id = str(row.get("odds_snapshot_id", "")).strip()

            if not snapshot_id:
                summary["errors"].append("Row missing odds_snapshot_id.")
                continue

            if snapshot_id in existing_ids or snapshot_id in incoming_ids:
                summary["duplicates"] += 1
                continue

            incoming_ids.add(snapshot_id)
            accepted_rows.append(row.to_dict())

        if accepted_rows:
            accepted = pd.DataFrame(accepted_rows, columns=COLUMNS)
            combined = pd.concat([existing, accepted], ignore_index=True)
            combined = combined[COLUMNS]
            combined.to_csv(
                MARKET_ODDS_HISTORY_FILE,
                index=False,
                encoding="utf-8",
            )
            summary["inserted"] = int(len(accepted_rows))
            summary["stored_rows"] = int(len(combined))
        else:
            summary["stored_rows"] = int(len(existing))

    except Exception as exc:
        summary["errors"].append(f"Market odds append failed: {exc}")

    return summary


def append_game_market_snapshot(
    *,
    game_id: Any,
    game_date: str,
    start_time: Any,
    captured_at: Any,
    home_team: str,
    away_team: str,
    bookmaker_quotes: list[dict[str, Any]],
    odds_quality_status: str = "UNAVAILABLE",
    suspicious_odds_reason: str = "",
    starting_pitcher_confirmed: bool | None = None,
    lineup_confirmed: bool | None = None,
) -> dict[str, Any]:
    """Validate one game capture, build rows and append them to history."""
    errors = _validation_errors(
        game_id=game_id,
        start_time=start_time,
        captured_at=captured_at,
        bookmaker_quotes=bookmaker_quotes,
    )

    if errors:
        stored_rows = 0
        try:
            stored_rows = int(len(_load_market_odds_history()))
        except Exception:
            stored_rows = 0

        return {
            "received": 0,
            "inserted": 0,
            "duplicates": 0,
            "stored_rows": stored_rows,
            "errors": errors,
        }

    rows = build_market_odds_rows(
        game_id=game_id,
        game_date=game_date,
        start_time=start_time,
        captured_at=captured_at,
        home_team=home_team,
        away_team=away_team,
        bookmaker_quotes=bookmaker_quotes,
        odds_quality_status=odds_quality_status,
        suspicious_odds_reason=suspicious_odds_reason,
        starting_pitcher_confirmed=starting_pitcher_confirmed,
        lineup_confirmed=lineup_confirmed,
    )

    summary = append_market_odds_rows(rows)

    if not rows and not summary["errors"]:
        summary["errors"].append("No valid market odds rows were built.")

    return summary


def refresh_opening_closing_flags(
    game_ids: list[str] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Mark earliest pregame prices as opening and final pregame prices as closing."""
    summary: dict[str, Any] = {
        "groups_processed": 0,
        "opening_marked": 0,
        "closing_marked": 0,
        "errors": [],
    }

    current_time = _to_utc_datetime(now_utc) if now_utc is not None else None
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    try:
        frame = _load_market_odds_history()
        if frame.empty:
            return summary

        frame["captured_at_parsed"] = pd.to_datetime(
            frame["captured_at"],
            errors="coerce",
            utc=True,
        )
        frame["start_time_parsed"] = pd.to_datetime(
            frame["start_time"],
            errors="coerce",
            utc=True,
        )

        selected_mask = frame["is_pregame"]

        if game_ids is not None:
            target_game_ids = {str(game_id) for game_id in game_ids}
            selected_mask = selected_mask & frame["game_id"].isin(target_game_ids)

        working = frame[selected_mask].copy()
        working = working.dropna(
            subset=["captured_at_parsed", "start_time_parsed"]
        )

        if working.empty:
            return summary

        working["line_group"] = working["line"].fillna("").astype(str)

        group_keys = [
            "game_id",
            "bookmaker_key",
            "market",
            "side",
            "line_group",
        ]

        for _, group in working.groupby(group_keys, dropna=False):
            summary["groups_processed"] += 1
            indices = group.index

            frame.loc[indices, "is_opening_snapshot"] = False
            frame.loc[indices, "is_closing_snapshot"] = False

            opening_index = group["captured_at_parsed"].idxmin()
            frame.loc[opening_index, "is_opening_snapshot"] = True
            summary["opening_marked"] += 1

            start_time = group["start_time_parsed"].iloc[0]
            if pd.notna(start_time) and start_time <= current_time:
                pregame_group = group[
                    group["captured_at_parsed"] < start_time
                ]

                if not pregame_group.empty:
                    closing_index = pregame_group[
                        "captured_at_parsed"
                    ].idxmax()
                    frame.loc[closing_index, "is_closing_snapshot"] = True
                    summary["closing_marked"] += 1

        frame = frame[COLUMNS]
        frame.to_csv(
            MARKET_ODDS_HISTORY_FILE,
            index=False,
            encoding="utf-8",
        )

    except Exception as exc:
        summary["errors"].append(f"Opening closing refresh failed: {exc}")

    return summary


def settle_market_odds_history(
    final_games: list[dict[str, Any]],
) -> dict[str, Any]:
    """Backfill settlement fields for all odds rows matching finalized games."""
    summary: dict[str, Any] = {
        "games_received": len(final_games),
        "games_updated": 0,
        "rows_updated": 0,
        "errors": [],
    }

    try:
        frame = _load_market_odds_history()
        if frame.empty:
            return summary

        settled_at = _utc_iso(datetime.now(timezone.utc))

        for game in final_games:
            game_id = str(game.get("game_id", "")).strip()
            if not game_id:
                continue

            home_score = _safe_float(game.get("home_score"))
            away_score = _safe_float(game.get("away_score"))
            home_win = _safe_float(game.get("home_win"))

            if home_win is None and home_score is not None and away_score is not None:
                if home_score > away_score:
                    home_win = 1.0
                elif home_score < away_score:
                    home_win = 0.0

            unsettled_mask = (
                (frame["game_id"] == game_id)
                & (
                    frame["settled_at"].isna()
                    | (frame["settled_at"].astype(str).str.strip() == "")
                )
            )

            rows_to_update = int(unsettled_mask.sum())
            if rows_to_update == 0:
                continue

            frame.loc[unsettled_mask, "settled_at"] = settled_at
            frame.loc[unsettled_mask, "home_win"] = (
                int(home_win) if home_win in (0.0, 1.0) else ""
            )
            frame.loc[unsettled_mask, "home_score"] = (
                int(home_score) if home_score is not None else ""
            )
            frame.loc[unsettled_mask, "away_score"] = (
                int(away_score) if away_score is not None else ""
            )

            summary["games_updated"] += 1
            summary["rows_updated"] += rows_to_update

        frame = frame[COLUMNS]
        frame.to_csv(
            MARKET_ODDS_HISTORY_FILE,
            index=False,
            encoding="utf-8",
        )

    except Exception as exc:
        summary["errors"].append(f"Market odds settlement failed: {exc}")

    return summary


def compute_moneyline_clv(
    *,
    game_id: Any,
    selected_side: str,
    entry_odds: float | None = None,
    reference_bookmaker_key: str | None = None,
) -> dict[str, Any]:
    """Compute entry-versus-closing CLV or opening-versus-closing market move."""
    selected_side = str(selected_side).strip().lower()

    if selected_side not in {"home", "away"}:
        raise ValueError("selected_side must be home or away.")

    result: dict[str, Any] = {
        "game_id": str(game_id),
        "selected_side": selected_side,
        "entry_odds": None,
        "clv_basis": "",
        "opening_odds": None,
        "closing_odds": None,
        "opening_implied_prob": None,
        "entry_implied_prob": None,
        "closing_implied_prob": None,
        "clv_implied_prob": None,
        "clv_decimal_odds": None,
        "status": "UNAVAILABLE",
        "reason": "",
    }

    try:
        frame = read_market_odds_history(
            game_id=game_id,
            market="moneyline",
            bookmaker_key=reference_bookmaker_key,
        )

        frame = frame[frame["side"] == selected_side].copy()

        if frame.empty:
            result["reason"] = "No moneyline history found."
            return result

        opening_rows = frame[frame["is_opening_snapshot"]]
        closing_rows = frame[frame["is_closing_snapshot"]]

        if not opening_rows.empty:
            opening_values = pd.to_numeric(
                opening_rows["odds"],
                errors="coerce",
            ).dropna()
            if not opening_values.empty:
                result["opening_odds"] = float(opening_values.median())
                result["opening_implied_prob"] = (
                    1.0 / result["opening_odds"]
                )

        if closing_rows.empty:
            result["reason"] = "No closing snapshot available."
            return result

        closing_values = pd.to_numeric(
            closing_rows["odds"],
            errors="coerce",
        ).dropna()

        if closing_values.empty:
            result["reason"] = "Closing odds are invalid."
            return result

        closing_odds = float(closing_values.median())
        result["closing_odds"] = closing_odds
        result["closing_implied_prob"] = 1.0 / closing_odds

        parsed_entry_odds = _safe_decimal_odds(entry_odds)

        if parsed_entry_odds is not None:
            result["entry_odds"] = parsed_entry_odds
            result["entry_implied_prob"] = 1.0 / parsed_entry_odds
            result["clv_basis"] = "ENTRY_VS_CLOSING"
            result["clv_implied_prob"] = (
                result["closing_implied_prob"]
                - result["entry_implied_prob"]
            )
            result["clv_decimal_odds"] = parsed_entry_odds - closing_odds
            result["status"] = "OK"
            return result

        if result["opening_odds"] is None:
            result["reason"] = "No opening snapshot or valid entry odds available."
            return result

        result["clv_basis"] = "OPENING_VS_CLOSING"
        result["clv_implied_prob"] = (
            result["closing_implied_prob"]
            - result["opening_implied_prob"]
        )
        result["clv_decimal_odds"] = (
            result["opening_odds"] - closing_odds
        )
        result["status"] = "MARKET_MOVE_ONLY"
        result["reason"] = "No entry odds provided; result describes market move only."

    except Exception as exc:
        result["status"] = "UNAVAILABLE"
        result["reason"] = f"Moneyline CLV calculation failed: {exc}"

    return result


def read_market_odds_history(
    *,
    game_id: Any | None = None,
    market: str | None = None,
    bookmaker_key: str | None = None,
    pregame_only: bool = False,
    closing_only: bool = False,
) -> pd.DataFrame:
    """Read market odds history with optional filters."""
    if market is not None and market not in SUPPORTED_MARKETS:
        raise ValueError(
            f"Unsupported market: {market}. "
            f"Expected one of {sorted(SUPPORTED_MARKETS)}."
        )

    frame = _load_market_odds_history()

    if frame.empty:
        return frame

    if game_id is not None:
        frame = frame[frame["game_id"] == str(game_id)]

    if market is not None:
        frame = frame[frame["market"] == market]

    if bookmaker_key is not None:
        frame = frame[
            frame["bookmaker_key"].astype(str) == str(bookmaker_key)
        ]

    if pregame_only:
        frame = frame[frame["is_pregame"]]

    if closing_only:
        frame = frame[frame["is_closing_snapshot"]]

    return frame.reset_index(drop=True)
