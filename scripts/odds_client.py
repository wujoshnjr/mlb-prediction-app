"""The Odds API client returning one market summary row per MLB game.

This module keeps the existing one-row-per-game contract while adding
market-quality metadata and bookmaker-level quotes for auditability.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
ODDS_FETCH_DIAGNOSTIC_PATH = Path("report/odds_fetch_diagnostic.json")

MIN_DECIMAL_ODDS = 1.05
MAX_DECIMAL_ODDS = 10.00
MAX_ABS_SPREAD_LINE = 2.5
MIN_TOTAL_LINE = 5.0
MAX_TOTAL_LINE = 14.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    try:
        return str(value)
    except Exception:
        return "unserializable_object"


def safe_json_dump(data: dict[str, Any], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_json_safe(data), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    except Exception:
        return


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(params)
    if "apiKey" in cleaned:
        cleaned["apiKey"] = "***REDACTED***"
    return cleaned


def _headers_snapshot(response: requests.Response | None) -> dict[str, Any]:
    if response is None:
        return {}

    return {
        "x-requests-remaining": response.headers.get("x-requests-remaining"),
        "x-requests-used": response.headers.get("x-requests-used"),
        "x-requests-last": response.headers.get("x-requests-last"),
    }


def _base_diagnostic(api_key_present: bool, date_str: str | None) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "status": "ok",
        "sport_key": "baseball_mlb",
        "endpoint": ODDS_URL,
        "requested_date": date_str,
        "api_key_present": bool(api_key_present),
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
        "attempts": [],
        "selected_attempt": None,
        "final_event_count": 0,
        "final_usable_row_count": 0,
        "errors": [],
        "warnings": [],
        "recommendations": [],
    }


def _write_diagnostic(diagnostic: dict[str, Any]) -> None:
    safe_json_dump(diagnostic, ODDS_FETCH_DIAGNOSTIC_PATH)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _median_or_none(values: list[Any]) -> float | None:
    numeric = [
        parsed
        for value in values
        if (parsed := _finite_float(value)) is not None
    ]
    return float(np.median(numeric)) if numeric else None


def _append_error(errors: list | None, message: str) -> None:
    if errors is not None:
        errors.append(message)


def _quality_status(
    home_odds: float | None,
    away_odds: float | None,
    spread_line: float | None,
    total_line: float | None,
    bookmaker_quotes: list[dict[str, Any]],
) -> tuple[str, str]:
    """Classify clear integrity risks; suspicious markets must not be trusted."""
    if home_odds is None or away_odds is None:
        return "UNAVAILABLE", "Missing two-way moneyline prices."

    reasons: list[str] = []

    for side, odds in (("home", home_odds), ("away", away_odds)):
        if odds < MIN_DECIMAL_ODDS:
            reasons.append(f"{side} moneyline below {MIN_DECIMAL_ODDS:.2f}")
        elif odds > MAX_DECIMAL_ODDS:
            reasons.append(f"{side} moneyline above {MAX_DECIMAL_ODDS:.2f}")

    if spread_line is not None and abs(spread_line) > MAX_ABS_SPREAD_LINE:
        reasons.append(f"spread abs value above {MAX_ABS_SPREAD_LINE:.1f}")

    if total_line is not None and not (
        MIN_TOTAL_LINE <= total_line <= MAX_TOTAL_LINE
    ):
        reasons.append(
            f"total outside {MIN_TOTAL_LINE:.1f}-{MAX_TOTAL_LINE:.1f}"
        )

    home_quotes = [
        quote.get("home_moneyline_odds")
        for quote in bookmaker_quotes
        if quote.get("home_moneyline_odds") is not None
    ]
    away_quotes = [
        quote.get("away_moneyline_odds")
        for quote in bookmaker_quotes
        if quote.get("away_moneyline_odds") is not None
    ]

    for side, quotes in (("home", home_quotes), ("away", away_quotes)):
        numeric = [float(value) for value in quotes]
        if len(numeric) >= 2 and min(numeric) > 0:
            if max(numeric) / min(numeric) > 1.35:
                reasons.append(f"{side} bookmaker prices disagree materially")

    if reasons:
        return "SUSPICIOUS", "; ".join(reasons)

    return "OK", ""


def save_odds_snapshot(
    game_key: str,
    home_odds: float | None,
    away_odds: float | None,
    over_odds: float | None = None,
    under_odds: float | None = None,
    *,
    spread_line: float | None = None,
    total_line: float | None = None,
    odds_source: str = "unknown",
    odds_quality_status: str = "UNAVAILABLE",
    suspicious_odds_reason: str = "",
    bookmaker_quotes: list[dict[str, Any]] | None = None,
) -> None:
    """Save already-fetched quotes only; this does not create an API request."""
    snapshot_dir = "data/odds_snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    filepath = os.path.join(snapshot_dir, f"{game_key}.csv")

    row = {
        "timestamp": datetime.now().isoformat(),
        "home_odds": home_odds,
        "away_odds": away_odds,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "spread_line": spread_line,
        "total_line": total_line,
        "odds_source": odds_source,
        "odds_quality_status": odds_quality_status,
        "suspicious_odds_reason": suspicious_odds_reason,
        "bookmaker_quotes": json.dumps(
            bookmaker_quotes or [],
            ensure_ascii=True,
        ),
    }

    if os.path.exists(filepath):
        frame = pd.read_csv(filepath)
        frame = pd.concat(
            [frame, pd.DataFrame([row])],
            ignore_index=True,
        ).tail(12)
    else:
        frame = pd.DataFrame([row])

    frame.to_csv(filepath, index=False)


def extract_odds_curve_features(game_id: str) -> tuple[float, float, int]:
    filepath = os.path.join("data/odds_snapshots", f"{game_id}.csv")
    if not os.path.exists(filepath):
        return 0.0, 0.0, 0

    frame = pd.read_csv(filepath)
    if len(frame) < 2 or "home_odds" not in frame.columns:
        return 0.0, 0.0, 0

    values = (
        pd.to_numeric(frame["home_odds"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(values) < 2 or np.mean(values) == 0:
        return 0.0, 0.0, 0

    slope = np.polyfit(
        np.arange(len(values)),
        values,
        1,
    )[0] / np.mean(values)
    volatility = np.std(values) / np.mean(values)
    signs = np.sign(np.diff(values))
    reversals = (
        int(np.sum(signs[:-1] != signs[1:]))
        if len(signs) > 1
        else 0
    )

    return float(slope), float(volatility), reversals


def _bookmaker_quote(
    bookmaker: dict[str, Any],
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    quote: dict[str, Any] = {
        "bookmaker": str(bookmaker.get("title", "")).strip() or "unknown",
        "bookmaker_key": str(bookmaker.get("key", "")).strip(),
        "last_update": bookmaker.get("last_update"),
        "home_moneyline_odds": None,
        "away_moneyline_odds": None,
        "home_spread_line": None,
        "home_spread_odds": None,
        "total_line": None,
        "over_odds": None,
        "under_odds": None,
    }

    for market in bookmaker.get("markets", []) or []:
        key = market.get("key")

        for outcome in market.get("outcomes", []) or []:
            name = outcome.get("name")
            price = _finite_float(outcome.get("price"))
            point = _finite_float(outcome.get("point"))

            if key == "h2h":
                if name == home_team:
                    quote["home_moneyline_odds"] = price
                elif name == away_team:
                    quote["away_moneyline_odds"] = price

            elif key == "spreads" and name == home_team:
                quote["home_spread_line"] = point
                quote["home_spread_odds"] = price

            elif key == "totals":
                if name == "Over":
                    quote["total_line"] = point
                    quote["over_odds"] = price
                elif name == "Under":
                    if quote["total_line"] is None:
                        quote["total_line"] = point
                    quote["under_odds"] = price

    return quote


def _parse_events(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []

    return [event for event in payload if isinstance(event, dict)]


def _rows_from_events(
    events: list[dict[str, Any]],
    *,
    date_str: str | None,
    errors: list | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in events:
        home_team = str(event.get("home_team", "")).strip()
        away_team = str(event.get("away_team", "")).strip()

        if not home_team or not away_team:
            continue

        bookmaker_quotes = [
            _bookmaker_quote(bookmaker, home_team, away_team)
            for bookmaker in event.get("bookmakers", []) or []
            if isinstance(bookmaker, dict)
        ]

        bookmaker_quotes = [
            quote
            for quote in bookmaker_quotes
            if any(
                quote.get(field) is not None
                for field in (
                    "home_moneyline_odds",
                    "away_moneyline_odds",
                    "home_spread_line",
                    "total_line",
                    "over_odds",
                    "under_odds",
                )
            )
        ]

        home_odds = _median_or_none(
            [quote.get("home_moneyline_odds") for quote in bookmaker_quotes]
        )
        away_odds = _median_or_none(
            [quote.get("away_moneyline_odds") for quote in bookmaker_quotes]
        )
        spread_line = _median_or_none(
            [quote.get("home_spread_line") for quote in bookmaker_quotes]
        )
        total_line = _median_or_none(
            [quote.get("total_line") for quote in bookmaker_quotes]
        )
        over_odds = _median_or_none(
            [quote.get("over_odds") for quote in bookmaker_quotes]
        )
        under_odds = _median_or_none(
            [quote.get("under_odds") for quote in bookmaker_quotes]
        )

        odds_quality_status, suspicious_odds_reason = _quality_status(
            home_odds=home_odds,
            away_odds=away_odds,
            spread_line=spread_line,
            total_line=total_line,
            bookmaker_quotes=bookmaker_quotes,
        )

        source_titles = sorted(
            {
                str(quote.get("bookmaker", "")).strip()
                for quote in bookmaker_quotes
                if str(quote.get("bookmaker", "")).strip()
            }
        )
        odds_source = ", ".join(source_titles) or "unknown"

        row = {
            "event_id": event.get("id"),
            "commence_time": event.get("commence_time"),
            "requested_date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_moneyline_odds": home_odds,
            "away_moneyline_odds": away_odds,
            "total_line": total_line,
            "spread_line": spread_line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "odds_source": odds_source,
            "odds_quality_status": odds_quality_status,
            "suspicious_odds_reason": suspicious_odds_reason,
            "bookmaker_quotes": bookmaker_quotes,
        }
        rows.append(row)

        if odds_quality_status == "SUSPICIOUS":
            _append_error(
                errors,
                (
                    "Suspicious odds market for "
                    f"{away_team} at {home_team}: {suspicious_odds_reason}"
                ),
            )
        elif odds_quality_status == "UNAVAILABLE":
            _append_error(
                errors,
                (
                    "Unavailable odds market for "
                    f"{away_team} at {home_team}: {suspicious_odds_reason}"
                ),
            )

        if home_odds is not None or away_odds is not None:
            safe_id = (
                f"{home_team}_{away_team}"
                .replace(" ", "_")
                .replace("/", "_")
                .lower()
            )
            save_odds_snapshot(
                game_key=safe_id,
                home_odds=home_odds,
                away_odds=away_odds,
                over_odds=over_odds,
                under_odds=under_odds,
                spread_line=spread_line,
                total_line=total_line,
                odds_source=odds_source,
                odds_quality_status=odds_quality_status,
                suspicious_odds_reason=suspicious_odds_reason,
                bookmaker_quotes=bookmaker_quotes,
            )

    return rows


def _quality_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for row in rows:
        status = str(row.get("odds_quality_status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1

    return counts


def _usable_moneyline_row_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("home_moneyline_odds") is not None
        and row.get("away_moneyline_odds") is not None
    )


def _build_attempts(
    *,
    api_key: str,
    date_str: str | None,
    bookmakers: str,
) -> list[tuple[str, dict[str, Any]]]:
    attempts: list[tuple[str, dict[str, Any]]] = []

    base_with_bookmakers = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
        "bookmakers": bookmakers,
    }

    if date_str:
        dated = dict(base_with_bookmakers)
        dated["commenceTimeFrom"] = f"{date_str}T00:00:00Z"
        dated["commenceTimeTo"] = f"{date_str}T23:59:59Z"
        attempts.append(("configured_bookmakers_full_market_date_window", dated))

    attempts.append(("configured_bookmakers_full_market", base_with_bookmakers))

    attempts.append(
        (
            "region_h2h_only",
            {
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )
    )

    attempts.append(
        (
            "region_full_market_no_bookmaker_filter",
            {
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal",
            },
        )
    )

    attempts.append(
        (
            "us2_h2h_only",
            {
                "apiKey": api_key,
                "regions": "us2",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )
    )

    attempts.append(
        (
            "combined_us_us2_h2h_only",
            {
                "apiKey": api_key,
                "regions": "us,us2",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )
    )

    return attempts


def fetch_odds(
    api_key: str = None,
    date_str: str = None,
    errors: list = None,
) -> pd.DataFrame:
    """Fetch available MLB odds and return one audited row per event.

    Fallback strategy:
    1. configured bookmaker full-market attempt
    2. region-wide h2h attempt without bookmaker filter
    3. region-wide full-market attempt without bookmaker filter
    4. us2 h2h attempt
    5. combined us/us2 h2h attempt
    """
    api_key = (api_key or os.getenv("ODDS_API_KEY", "")).strip()
    diagnostic = _base_diagnostic(api_key_present=bool(api_key), date_str=date_str)

    if not api_key:
        message = "Odds API key missing"
        _append_error(errors, message)
        diagnostic["status"] = "failed"
        diagnostic["errors"].append(message)
        diagnostic["recommendations"].append("Check ODDS_API_KEY secret.")
        _write_diagnostic(diagnostic)
        return pd.DataFrame()

    bookmakers = os.getenv(
        "ODDS_BOOKMAKERS",
        "draftkings,fanduel,betmgm,bet365",
    )

    selected_rows: list[dict[str, Any]] = []
    selected_events: list[dict[str, Any]] = []
    selected_attempt_name: str | None = None
    selected_usable_row_count = 0

    attempts = _build_attempts(
        api_key=api_key,
        date_str=date_str,
        bookmakers=bookmakers,
    )

    for attempt_name, params in attempts:
        attempt_record: dict[str, Any] = {
            "attempt_name": attempt_name,
            "params": _redact_params(params),
            "status_code": None,
            "event_count": 0,
            "usable_row_count": 0,
            "quality_counts": {},
            "request_headers": {},
            "error": "",
        }

        response: requests.Response | None = None
        events: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for retry_index in range(3):
            try:
                response = requests.get(
                    ODDS_URL,
                    params=params,
                    timeout=30,
                )

                attempt_record["status_code"] = response.status_code
                attempt_record["request_headers"] = _headers_snapshot(response)

                if response.status_code == 401:
                    message = "Odds API 401: invalid API key"
                    attempt_record["error"] = message
                    diagnostic["errors"].append(message)
                    _append_error(errors, message)
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "failed"
                    diagnostic["recommendations"].append("Check ODDS_API_KEY secret.")
                    _write_diagnostic(diagnostic)
                    return pd.DataFrame()

                if response.status_code == 429:
                    if retry_index < 2:
                        time.sleep(10)
                        continue

                    message = "Odds API 429: rate limited"
                    attempt_record["error"] = message
                    diagnostic["warnings"].append(message)
                    _append_error(errors, message)
                    break

                if response.status_code == 422:
                    message = (
                        "Odds API 422: requested market, region, or bookmaker is unavailable"
                    )
                    attempt_record["error"] = message
                    diagnostic["warnings"].append(f"{attempt_name}: {message}")
                    _append_error(errors, message)
                    break

                response.raise_for_status()

                payload = response.json()
                if not isinstance(payload, list):
                    message = "Odds API returned an unexpected payload."
                    attempt_record["error"] = message
                    diagnostic["warnings"].append(f"{attempt_name}: {message}")
                    _append_error(errors, message)
                    break

                events = _parse_events(payload)
                rows = _rows_from_events(
                    events,
                    date_str=date_str,
                    errors=errors,
                )
                break

            except Exception as exc:
                if retry_index == 2:
                    message = f"Odds API final failure for {attempt_name}: {exc}"
                    attempt_record["error"] = message
                    diagnostic["warnings"].append(message)
                    _append_error(errors, message)
                else:
                    time.sleep(5)

        usable_row_count = _usable_moneyline_row_count(rows)

        attempt_record["event_count"] = len(events)
        attempt_record["usable_row_count"] = usable_row_count
        attempt_record["quality_counts"] = _quality_counts(rows)
        diagnostic["attempts"].append(attempt_record)

        if len(events) > 0 and usable_row_count > 0:
            selected_events = events
            selected_rows = rows
            selected_attempt_name = attempt_name
            selected_usable_row_count = usable_row_count
            break

    diagnostic["selected_attempt"] = selected_attempt_name
    diagnostic["final_event_count"] = len(selected_events)
    diagnostic["final_usable_row_count"] = selected_usable_row_count

    if selected_rows:
        diagnostic["status"] = "ok"
        _write_diagnostic(diagnostic)

        output = pd.DataFrame(selected_rows)
        status_counts = output["odds_quality_status"].value_counts(dropna=False).to_dict()
        print(
            "Odds data received for "
            f"{len(output)} MLB matchups; quality={status_counts}; "
            f"usable_moneyline_rows={selected_usable_row_count}; "
            f"selected_attempt={selected_attempt_name}"
        )
        return output

    message = "No usable MLB odds markets returned after fallback attempts"
    _append_error(errors, message)
    diagnostic["status"] = "partial"
    diagnostic["errors"].append(message)
    diagnostic["recommendations"].extend(
        [
            "Check ODDS_API_KEY secret.",
            "Check subscription access for baseball_mlb.",
            "Check bookmaker filter; region fallback also returned zero.",
            "Check The Odds API quota headers in odds_fetch_diagnostic attempts.",
            "Check whether commenceTime date window excluded today's MLB slate.",
        ]
    )
    _write_diagnostic(diagnostic)
    return pd.DataFrame()
