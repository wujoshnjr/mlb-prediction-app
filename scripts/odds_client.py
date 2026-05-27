"""The Odds API client returning one market summary row per MLB game.

This module keeps the existing one-row-per-game contract while adding
market-quality metadata and bookmaker-level quotes for auditability.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests


ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

MIN_DECIMAL_ODDS = 1.05
MAX_DECIMAL_ODDS = 10.00
MAX_ABS_SPREAD_LINE = 2.5
MIN_TOTAL_LINE = 5.0
MAX_TOTAL_LINE = 14.5


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
    """Classify only clear integrity risks; suspicious markets are not trusted."""
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
    def fetch_odds(
    api_key: str = None,
    date_str: str = None,
    errors: list = None,
) -> pd.DataFrame:
    """Fetch available MLB odds and return one audited row per event.

    This function does not make extra API calls beyond the existing odds
    request. The requested date is retained for downstream matching; precise
    schedule/start-time matching is completed in model.py.
    """
    api_key = (api_key or os.getenv("ODDS_API_KEY", "")).strip()
    if not api_key:
        _append_error(errors, "Odds API key missing")
        return pd.DataFrame()

    bookmakers = os.getenv(
        "ODDS_BOOKMAKERS",
        "draftkings,fanduel,betmgm,bet365",
    )

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
        "bookmakers": bookmakers,
    }

    events: list[dict[str, Any]] = []

    for attempt in range(3):
        try:
            response = requests.get(
                ODDS_URL,
                params=params,
                timeout=30,
            )

            if response.status_code == 401:
                _append_error(errors, "Odds API 401: invalid API key")
                return pd.DataFrame()

            if response.status_code == 429:
                if attempt < 2:
                    time.sleep(10)
                    continue
                _append_error(errors, "Odds API 429: rate limited")
                return pd.DataFrame()

            if response.status_code == 422:
                _append_error(
                    errors,
                    "Odds API 422: requested market or bookmaker is unavailable",
                )
                return pd.DataFrame()

            response.raise_for_status()

            payload = response.json()
            if isinstance(payload, list):
                events = [
                    event
                    for event in payload
                    if isinstance(event, dict)
                ]
            else:
                _append_error(errors, "Odds API returned an unexpected payload.")
                return pd.DataFrame()

            break

        except Exception as exc:
            if attempt == 2:
                _append_error(errors, f"Odds API final failure: {exc}")
                return pd.DataFrame()
            time.sleep(5)

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
            [
                quote.get("home_moneyline_odds")
                for quote in bookmaker_quotes
            ]
        )
        away_odds = _median_or_none(
            [
                quote.get("away_moneyline_odds")
                for quote in bookmaker_quotes
            ]
        )
        spread_line = _median_or_none(
            [
                quote.get("home_spread_line")
                for quote in bookmaker_quotes
            ]
        )
        total_line = _median_or_none(
            [
                quote.get("total_line")
                for quote in bookmaker_quotes
            ]
        )
        over_odds = _median_or_none(
            [
                quote.get("over_odds")
                for quote in bookmaker_quotes
            ]
        )
        under_odds = _median_or_none(
            [
                quote.get("under_odds")
                for quote in bookmaker_quotes
            ]
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
                    f"{away_team} at {home_team}: "
                    f"{suspicious_odds_reason}"
                ),
            )
        elif odds_quality_status == "UNAVAILABLE":
            _append_error(
                errors,
                (
                    "Unavailable odds market for "
                    f"{away_team} at {home_team}: "
                    f"{suspicious_odds_reason}"
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

    output = pd.DataFrame(rows)

    if output.empty:
        _append_error(errors, "No usable MLB odds markets returned")
        return output

    status_counts = (
        output["odds_quality_status"]
        .value_counts(dropna=False)
        .to_dict()
    )
    print(
        "Odds data received for "
        f"{len(output)} MLB matchups; quality={status_counts}"
    )

    return output
