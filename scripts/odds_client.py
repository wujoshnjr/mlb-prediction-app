"""The Odds API client returning one market summary row per MLB game."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests


ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


def save_odds_snapshot(
    game_id: str,
    home_odds: float | None,
    away_odds: float | None,
    over_odds: float | None = None,
    under_odds: float | None = None,
):
    snapshot_dir = "data/odds_snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    filepath = os.path.join(snapshot_dir, f"{game_id}.csv")
    row = {
        "timestamp": datetime.now().isoformat(),
        "home_odds": home_odds,
        "away_odds": away_odds,
        "over_odds": over_odds,
        "under_odds": under_odds,
    }
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).tail(12)
    else:
        df = pd.DataFrame([row])
    df.to_csv(filepath, index=False)


def extract_odds_curve_features(game_id: str):
    filepath = os.path.join("data/odds_snapshots", f"{game_id}.csv")
    if not os.path.exists(filepath):
        return 0.0, 0.0, 0
    df = pd.read_csv(filepath)
    if len(df) < 2 or "home_odds" not in df.columns:
        return 0.0, 0.0, 0
    values = pd.to_numeric(df["home_odds"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2 or np.mean(values) == 0:
        return 0.0, 0.0, 0
    slope = np.polyfit(np.arange(len(values)), values, 1)[0] / np.mean(values)
    volatility = np.std(values) / np.mean(values)
    signs = np.sign(np.diff(values))
    reversals = int(np.sum(signs[:-1] != signs[1:])) if len(signs) > 1 else 0
    return float(slope), float(volatility), reversals


def _mean_or_none(values):
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(numeric)) if numeric else None


def _median_or_none(values):
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.median(numeric)) if numeric else None


def _append_error(errors: list | None, message: str):
    if errors is not None:
        errors.append(message)


def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    """Fetch h2h, spread and total markets and aggregate per scheduled matchup."""
    api_key = (api_key or os.getenv("ODDS_API_KEY", "")).strip()
    if not api_key:
        _append_error(errors, "Odds API key missing")
        return pd.DataFrame()

    bookmakers = os.getenv("ODDS_BOOKMAKERS", "draftkings,fanduel,betmgm,bet365")
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
        "bookmakers": bookmakers,
    }

    for attempt in range(3):
        try:
            response = requests.get(ODDS_URL, params=params, timeout=30)
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
                _append_error(errors, "Odds API 422: requested market or bookmaker is unavailable")
                return pd.DataFrame()
            response.raise_for_status()
            events = response.json()
            break
        except Exception as exc:
            if attempt == 2:
                _append_error(errors, f"Odds API final failure: {exc}")
                return pd.DataFrame()
            time.sleep(5)
    else:
        return pd.DataFrame()

    rows = []
    for event in events if isinstance(events, list) else []:
        home_team = str(event.get("home_team", "")).strip()
        away_team = str(event.get("away_team", "")).strip()
        if not home_team or not away_team:
            continue

        home_moneyline = []
        away_moneyline = []
        home_spreads = []
        total_lines = []
        over_prices = []
        under_prices = []
        source_titles = []

        for bookmaker in event.get("bookmakers", []):
            source_titles.append(str(bookmaker.get("title", "")).strip())
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if key == "h2h":
                        if name == home_team:
                            home_moneyline.append(price)
                        elif name == away_team:
                            away_moneyline.append(price)
                    elif key == "spreads" and name == home_team:
                        home_spreads.append(point)
                    elif key == "totals":
                        if point is not None:
                            total_lines.append(point)
                        if name == "Over":
                            over_prices.append(price)
                        elif name == "Under":
                            under_prices.append(price)

        home_odds = _mean_or_none(home_moneyline)
        away_odds = _mean_or_none(away_moneyline)
        total_line = _median_or_none(total_lines)
        spread_line = _median_or_none(home_spreads)

        row = {
            "event_id": event.get("id"),
            "commence_time": event.get("commence_time"),
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_moneyline_odds": home_odds,
            "away_moneyline_odds": away_odds,
            "total_line": total_line,
            "spread_line": spread_line,
            "over_odds": _mean_or_none(over_prices),
            "under_odds": _mean_or_none(under_prices),
            "odds_source": ", ".join(sorted({title for title in source_titles if title})) or "unknown",
        }
        rows.append(row)

        if home_odds is not None and away_odds is not None:
            safe_id = f"{home_team}_{away_team}".replace(" ", "_").lower()
            save_odds_snapshot(safe_id, home_odds, away_odds, row["over_odds"], row["under_odds"])

    output = pd.DataFrame(rows)
    if output.empty:
        _append_error(errors, "No usable MLB odds markets returned")
    else:
        print(f"Odds data received for {len(output)} MLB matchups")
    return output
