from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT = 8

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "home_team",
    "away_team",
    "venue_name",
    "weather_source",
    "weather_source_status",
    "weather_captured_at",
    "weather_forecast_time",
    "weather_api_endpoint",
    "weather_lat",
    "weather_lon",
    "weather_temp_f",
    "weather_feels_like_f",
    "weather_wind_speed_mph",
    "weather_wind_deg",
    "weather_precip_probability",
    "weather_rain_3h_in",
    "weather_snow_3h_in",
    "weather_condition",
    "weather_is_dome",
    "temp_effect",
    "wind_effect",
    "precip_effect",
    "weather_reason",
]

# ---------------------------------------------------------------------------
# MLB ballparks
# ---------------------------------------------------------------------------

BALLPARKS: List[Dict[str, Any]] = [
    {
        "name": "Chase Field",
        "city": "Phoenix",
        "lat": 33.4455,
        "lon": -112.0667,
        "is_dome": True,
        "team": "Arizona Diamondbacks",
        "aliases": ["Arizona Diamondbacks", "Diamondbacks", "D-backs", "ARI"],
    },
    {
        "name": "Sutter Health Park",
        "city": "West Sacramento",
        "lat": 38.5804,
        "lon": -121.5134,
        "is_dome": False,
        "team": "Athletics",
        "aliases": ["Athletics", "Oakland Athletics", "A's", "OAK", "ATH"],
    },
    {
        "name": "Truist Park",
        "city": "Atlanta",
        "lat": 33.8908,
        "lon": -84.4678,
        "is_dome": False,
        "team": "Atlanta Braves",
        "aliases": ["Atlanta Braves", "Braves", "ATL"],
    },
    {
        "name": "Oriole Park at Camden Yards",
        "city": "Baltimore",
        "lat": 39.2847,
        "lon": -76.6210,
        "is_dome": False,
        "team": "Baltimore Orioles",
        "aliases": ["Baltimore Orioles", "Orioles", "BAL"],
    },
    {
        "name": "Fenway Park",
        "city": "Boston",
        "lat": 42.3467,
        "lon": -71.0972,
        "is_dome": False,
        "team": "Boston Red Sox",
        "aliases": ["Boston Red Sox", "Red Sox", "BOS"],
    },
    {
        "name": "Wrigley Field",
        "city": "Chicago",
        "lat": 41.9484,
        "lon": -87.6553,
        "is_dome": False,
        "team": "Chicago Cubs",
        "aliases": ["Chicago Cubs", "Cubs", "CHC"],
    },
    {
        "name": "Guaranteed Rate Field",
        "city": "Chicago",
        "lat": 41.8299,
        "lon": -87.6338,
        "is_dome": False,
        "team": "Chicago White Sox",
        "aliases": ["Chicago White Sox", "White Sox", "CWS"],
    },
    {
        "name": "Great American Ball Park",
        "city": "Cincinnati",
        "lat": 39.0979,
        "lon": -84.5082,
        "is_dome": False,
        "team": "Cincinnati Reds",
        "aliases": ["Cincinnati Reds", "Reds", "CIN"],
    },
    {
        "name": "Progressive Field",
        "city": "Cleveland",
        "lat": 41.4962,
        "lon": -81.6852,
        "is_dome": False,
        "team": "Cleveland Guardians",
        "aliases": ["Cleveland Guardians", "Guardians", "CLE"],
    },
    {
        "name": "Coors Field",
        "city": "Denver",
        "lat": 39.7559,
        "lon": -104.9941,
        "is_dome": False,
        "team": "Colorado Rockies",
        "aliases": ["Colorado Rockies", "Rockies", "COL"],
    },
    {
        "name": "Comerica Park",
        "city": "Detroit",
        "lat": 42.3390,
        "lon": -83.0485,
        "is_dome": False,
        "team": "Detroit Tigers",
        "aliases": ["Detroit Tigers", "Tigers", "DET"],
    },
    {
        "name": "Minute Maid Park",
        "city": "Houston",
        "lat": 29.7573,
        "lon": -95.3555,
        "is_dome": True,
        "team": "Houston Astros",
        "aliases": ["Houston Astros", "Astros", "HOU"],
    },
    {
        "name": "Kauffman Stadium",
        "city": "Kansas City",
        "lat": 39.0517,
        "lon": -94.4803,
        "is_dome": False,
        "team": "Kansas City Royals",
        "aliases": ["Kansas City Royals", "Royals", "KC", "KCR"],
    },
    {
        "name": "Angel Stadium",
        "city": "Anaheim",
        "lat": 33.8003,
        "lon": -117.8827,
        "is_dome": False,
        "team": "Los Angeles Angels",
        "aliases": ["Los Angeles Angels", "Angels", "LAA"],
    },
    {
        "name": "Dodger Stadium",
        "city": "Los Angeles",
        "lat": 34.0739,
        "lon": -118.2400,
        "is_dome": False,
        "team": "Los Angeles Dodgers",
        "aliases": ["Los Angeles Dodgers", "Dodgers", "LAD"],
    },
    {
        "name": "loanDepot park",
        "city": "Miami",
        "lat": 25.7781,
        "lon": -80.2196,
        "is_dome": True,
        "team": "Miami Marlins",
        "aliases": ["Miami Marlins", "Marlins", "MIA"],
    },
    {
        "name": "American Family Field",
        "city": "Milwaukee",
        "lat": 43.0280,
        "lon": -87.9712,
        "is_dome": True,
        "team": "Milwaukee Brewers",
        "aliases": ["Milwaukee Brewers", "Brewers", "MIL"],
    },
    {
        "name": "Target Field",
        "city": "Minneapolis",
        "lat": 44.9817,
        "lon": -93.2777,
        "is_dome": False,
        "team": "Minnesota Twins",
        "aliases": ["Minnesota Twins", "Twins", "MIN"],
    },
    {
        "name": "Citi Field",
        "city": "New York",
        "lat": 40.7571,
        "lon": -73.8458,
        "is_dome": False,
        "team": "New York Mets",
        "aliases": ["New York Mets", "Mets", "NYM"],
    },
    {
        "name": "Yankee Stadium",
        "city": "New York",
        "lat": 40.8296,
        "lon": -73.9262,
        "is_dome": False,
        "team": "New York Yankees",
        "aliases": ["New York Yankees", "Yankees", "NYY"],
    },
    {
        "name": "Citizens Bank Park",
        "city": "Philadelphia",
        "lat": 39.9058,
        "lon": -75.1665,
        "is_dome": False,
        "team": "Philadelphia Phillies",
        "aliases": ["Philadelphia Phillies", "Phillies", "PHI"],
    },
    {
        "name": "PNC Park",
        "city": "Pittsburgh",
        "lat": 40.4469,
        "lon": -80.0057,
        "is_dome": False,
        "team": "Pittsburgh Pirates",
        "aliases": ["Pittsburgh Pirates", "Pirates", "PIT"],
    },
    {
        "name": "Petco Park",
        "city": "San Diego",
        "lat": 32.7073,
        "lon": -117.1566,
        "is_dome": False,
        "team": "San Diego Padres",
        "aliases": ["San Diego Padres", "Padres", "SD", "SDP"],
    },
    {
        "name": "Oracle Park",
        "city": "San Francisco",
        "lat": 37.7786,
        "lon": -122.3893,
        "is_dome": False,
        "team": "San Francisco Giants",
        "aliases": ["San Francisco Giants", "Giants", "SF", "SFG"],
    },
    {
        "name": "T-Mobile Park",
        "city": "Seattle",
        "lat": 47.5914,
        "lon": -122.3325,
        "is_dome": True,
        "team": "Seattle Mariners",
        "aliases": ["Seattle Mariners", "Mariners", "SEA"],
    },
    {
        "name": "Busch Stadium",
        "city": "St. Louis",
        "lat": 38.6226,
        "lon": -90.1928,
        "is_dome": False,
        "team": "St. Louis Cardinals",
        "aliases": ["St. Louis Cardinals", "Saint Louis Cardinals", "Cardinals", "STL"],
    },
    {
        "name": "Tropicana Field",
        "city": "St. Petersburg",
        "lat": 27.7682,
        "lon": -82.6534,
        "is_dome": True,
        "team": "Tampa Bay Rays",
        "aliases": ["Tampa Bay Rays", "Rays", "TB", "TBR"],
    },
    {
        "name": "Globe Life Field",
        "city": "Arlington",
        "lat": 32.7473,
        "lon": -97.0840,
        "is_dome": True,
        "team": "Texas Rangers",
        "aliases": ["Texas Rangers", "Rangers", "TEX"],
    },
    {
        "name": "Rogers Centre",
        "city": "Toronto",
        "lat": 43.6414,
        "lon": -79.3894,
        "is_dome": True,
        "team": "Toronto Blue Jays",
        "aliases": ["Toronto Blue Jays", "Blue Jays", "TOR"],
    },
    {
        "name": "Nationals Park",
        "city": "Washington",
        "lat": 38.8730,
        "lon": -77.0074,
        "is_dome": False,
        "team": "Washington Nationals",
        "aliases": ["Washington Nationals", "Nationals", "WSH", "WAS"],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return float(number)
    except (ValueError, TypeError):
        return None


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


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
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
        except (ValueError, TypeError):
            return None

    return None


def _first_nonempty(row: pd.Series, columns: List[str]) -> str:
    for column in columns:
        if column in row.index:
            value = _safe_str(row.get(column))
            if value:
                return value
    return ""


def _mm_to_inches(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / 25.4


def _normalise_lookup_text(value: Any) -> str:
    text = _safe_str(value).lower()
    return (
        text.replace("&", "and")
        .replace(".", "")
        .replace("-", " ")
        .replace("_", " ")
        .replace("'", "")
        .strip()
    )


def _lookup_ballpark(venue_name: str, home_team: str) -> Optional[Dict[str, Any]]:
    venue_norm = _normalise_lookup_text(venue_name)
    team_norm = _normalise_lookup_text(home_team)

    if venue_norm:
        for ballpark in BALLPARKS:
            candidates = [ballpark["name"], ballpark.get("city", "")]
            for candidate in candidates:
                candidate_norm = _normalise_lookup_text(candidate)
                if (
                    venue_norm == candidate_norm
                    or candidate_norm in venue_norm
                    or venue_norm in candidate_norm
                ):
                    return ballpark

    if team_norm:
        for ballpark in BALLPARKS:
            aliases = [ballpark.get("team", ""), *ballpark.get("aliases", [])]
            for alias in aliases:
                alias_norm = _normalise_lookup_text(alias)
                if team_norm == alias_norm:
                    return ballpark

    return None


def _clean_output_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null"}:
        return ""
    return value


def _clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.where(pd.notnull(frame), "")
    try:
        return frame.map(_clean_output_value)
    except AttributeError:
        return frame.applymap(_clean_output_value)


def _empty_weather_fields() -> Dict[str, Any]:
    return {
        "weather_temp_f": None,
        "weather_feels_like_f": None,
        "weather_wind_speed_mph": None,
        "weather_wind_deg": None,
        "weather_precip_probability": None,
        "weather_rain_3h_in": None,
        "weather_snow_3h_in": None,
        "weather_condition": "",
        "weather_forecast_time": "",
        "weather_api_endpoint": "",
    }


# ---------------------------------------------------------------------------
# OpenWeather API
# ---------------------------------------------------------------------------


def _fetch_forecast(
    lat: float,
    lon: float,
    api_key: str,
) -> Tuple[Optional[List[Dict[str, Any]]], str, str]:
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "imperial",
    }

    try:
        response = requests.get(
            FORECAST_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        return None, FORECAST_URL, f"forecast_request_failed: {exc}"

    if response.status_code != 200:
        return None, FORECAST_URL, f"forecast_http_{response.status_code}: {response.text[:120]}"

    try:
        data = response.json()
    except Exception as exc:
        return None, FORECAST_URL, f"forecast_json_error: {exc}"

    forecast_list = data.get("list")
    if not isinstance(forecast_list, list) or not forecast_list:
        return None, FORECAST_URL, "forecast_missing_list"

    return forecast_list, FORECAST_URL, ""


def _fetch_current(
    lat: float,
    lon: float,
    api_key: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "imperial",
    }

    try:
        response = requests.get(
            CURRENT_WEATHER_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        return None, CURRENT_WEATHER_URL, f"current_request_failed: {exc}"

    if response.status_code != 200:
        return None, CURRENT_WEATHER_URL, f"current_http_{response.status_code}: {response.text[:120]}"

    try:
        data = response.json()
    except Exception as exc:
        return None, CURRENT_WEATHER_URL, f"current_json_error: {exc}"

    return data, CURRENT_WEATHER_URL, ""


def _extract_forecast_entry(
    forecast_list: List[Dict[str, Any]],
    target_dt: Optional[datetime],
) -> Optional[Dict[str, Any]]:
    if not forecast_list:
        return None

    if target_dt is None:
        return forecast_list[0]

    target_ts = target_dt.timestamp()
    best_entry = None
    best_diff = float("inf")

    for entry in forecast_list:
        dt_value = _safe_float(entry.get("dt"))
        if dt_value is None:
            continue

        diff = abs(dt_value - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_entry = entry

    return best_entry if best_entry is not None else forecast_list[0]


def _parse_weather_entry(
    entry: Optional[Dict[str, Any]],
    is_dome: bool,
) -> Dict[str, Any]:
    if is_dome or entry is None:
        return _empty_weather_fields()

    main = entry.get("main") if isinstance(entry.get("main"), dict) else {}
    wind = entry.get("wind") if isinstance(entry.get("wind"), dict) else {}
    rain = entry.get("rain") if isinstance(entry.get("rain"), dict) else {}
    snow = entry.get("snow") if isinstance(entry.get("snow"), dict) else {}

    rain_raw = _safe_float(rain.get("3h"))
    if rain_raw is None:
        rain_raw = _safe_float(rain.get("1h"))

    snow_raw = _safe_float(snow.get("3h"))
    if snow_raw is None:
        snow_raw = _safe_float(snow.get("1h"))

    weather_list = entry.get("weather")
    condition = ""
    if isinstance(weather_list, list) and weather_list:
        first_weather = weather_list[0]
        if isinstance(first_weather, dict):
            condition = _safe_str(
                first_weather.get("description") or first_weather.get("main")
            )

    forecast_time = ""
    dt_value = _safe_float(entry.get("dt"))
    if dt_value is not None:
        forecast_time = datetime.fromtimestamp(
            dt_value,
            tz=timezone.utc,
        ).isoformat()

    return {
        "weather_temp_f": _safe_float(main.get("temp")),
        "weather_feels_like_f": _safe_float(main.get("feels_like")),
        "weather_wind_speed_mph": _safe_float(wind.get("speed")),
        "weather_wind_deg": _safe_float(wind.get("deg")),
        "weather_precip_probability": _safe_float(entry.get("pop")),
        "weather_rain_3h_in": _mm_to_inches(rain_raw),
        "weather_snow_3h_in": _mm_to_inches(snow_raw),
        "weather_condition": condition,
        "weather_forecast_time": forecast_time,
        "weather_api_endpoint": "",
    }


def _compute_effects(
    temp_f: Optional[float],
    wind_speed_mph: Optional[float],
    precip_probability: Optional[float],
    rain_3h_in: Optional[float],
    snow_3h_in: Optional[float],
    is_dome: bool,
) -> Tuple[float, float, float]:
    if is_dome:
        return 0.0, 0.0, 0.0

    temp_effect = 0.0
    if temp_f is not None:
        if temp_f < 45:
            temp_effect = -0.02
        elif 45 <= temp_f < 55:
            temp_effect = -0.01
        elif 55 <= temp_f <= 85:
            temp_effect = 0.0
        elif 85 < temp_f <= 95:
            temp_effect = 0.01
        else:
            temp_effect = -0.005

    wind_effect = 0.0
    if wind_speed_mph is not None:
        if wind_speed_mph < 10:
            wind_effect = 0.0
        elif 10 <= wind_speed_mph < 15:
            wind_effect = 0.005
        elif 15 <= wind_speed_mph < 20:
            wind_effect = 0.01
        else:
            wind_effect = 0.015

    precip_effect = 0.0
    has_rain = rain_3h_in is not None and rain_3h_in > 0
    has_snow = snow_3h_in is not None and snow_3h_in > 0

    if has_rain or has_snow:
        precip_effect = -0.015
    elif precip_probability is not None and precip_probability >= 0.5:
        precip_effect = -0.01

    return temp_effect, wind_effect, precip_effect


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def build_weather_context(
    daily_context_path: str = "data/daily_game_context.csv",
    output_path: str = "data/weather_context.csv",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    captured_at = _current_utc_iso()

    if api_key is None:
        api_key = os.environ.get("OPENWEATHER_API_KEY", "")

    api_key_present = bool(_safe_str(api_key))

    context_path = Path(daily_context_path)
    if not context_path.exists():
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
        return result

    try:
        context_frame = pd.read_csv(context_path)
    except Exception:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
        return result

    if context_frame.empty:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
        return result

    for column in ("game_id", "game_date", "start_time", "home_team", "away_team"):
        if column not in context_frame.columns:
            context_frame[column] = ""

    if "captured_at" not in context_frame.columns:
        context_frame["captured_at"] = ""

    context_frame = context_frame.copy()
    context_frame["game_id"] = context_frame["game_id"].astype(str)
    context_frame["captured_at_dt"] = pd.to_datetime(
        context_frame["captured_at"],
        errors="coerce",
        utc=True,
    )

    if context_frame["captured_at_dt"].notna().any():
        context_frame = context_frame.sort_values(["game_id", "captured_at_dt"])
        latest_frame = context_frame.groupby("game_id", as_index=False).tail(1)
    else:
        latest_frame = context_frame.drop_duplicates(subset=["game_id"], keep="last")

    rows: List[Dict[str, Any]] = []

    for _, game_row in latest_frame.iterrows():
        game_id = _safe_str(game_row.get("game_id"))
        game_date = _safe_str(game_row.get("game_date"))
        start_time = _safe_str(game_row.get("start_time"))
        home_team = _safe_str(game_row.get("home_team"))
        away_team = _safe_str(game_row.get("away_team"))
        venue_name_raw = _first_nonempty(
            game_row,
            ["venue_name", "venue", "ballpark", "stadium", "venue_full_name"],
        )

        start_dt = _parse_datetime(start_time)
        ballpark = _lookup_ballpark(venue_name_raw, home_team)

        if ballpark is None:
            row = {column: "" for column in OUTPUT_COLUMNS}
            row.update(
                {
                    "game_id": game_id,
                    "game_date": game_date,
                    "start_time": start_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue_name": venue_name_raw,
                    "weather_source": "fallback_neutral",
                    "weather_source_status": "venue_not_found",
                    "weather_captured_at": captured_at,
                    "weather_api_endpoint": "",
                    "weather_is_dome": False,
                    "temp_effect": 0.0,
                    "wind_effect": 0.0,
                    "precip_effect": 0.0,
                    "weather_reason": (
                        f"Ballpark not found for venue='{venue_name_raw}', "
                        f"home_team='{home_team}'"
                    ),
                }
            )
            rows.append(row)
            continue

        is_dome = bool(ballpark.get("is_dome", False))
        lat = float(ballpark["lat"])
        lon = float(ballpark["lon"])

        weather_data = _empty_weather_fields()
        weather_source = "dome_neutral" if is_dome else "openweather"
        weather_source_status = "dome_neutral" if is_dome else "not_requested"
        weather_reason = ""

        if is_dome:
            weather_data["weather_api_endpoint"] = "none"
        elif not api_key_present:
            weather_source_status = "no_api_key"
            weather_reason = "OPENWEATHER_API_KEY missing"
        else:
            forecast_list, forecast_endpoint, forecast_error = _fetch_forecast(
                lat,
                lon,
                str(api_key),
            )

            if forecast_list is not None:
                forecast_entry = _extract_forecast_entry(forecast_list, start_dt)
                weather_data = _parse_weather_entry(forecast_entry, is_dome=False)
                weather_data["weather_api_endpoint"] = forecast_endpoint
                weather_source_status = "forecast_ok"
            else:
                current_data, current_endpoint, current_error = _fetch_current(
                    lat,
                    lon,
                    str(api_key),
                )

                if current_data is not None:
                    weather_data = _parse_weather_entry(current_data, is_dome=False)
                    weather_data["weather_api_endpoint"] = current_endpoint
                    weather_source_status = "current_ok"
                    weather_reason = f"Forecast fallback used: {forecast_error}"
                else:
                    weather_source_status = "api_error"
                    weather_reason = (
                        f"Forecast: {forecast_error}; Current: {current_error}"
                    )
                    weather_data = _empty_weather_fields()
                    weather_data["weather_api_endpoint"] = (
                        f"{FORECAST_URL}; fallback={CURRENT_WEATHER_URL}"
                    )

        temp_effect, wind_effect, precip_effect = _compute_effects(
            weather_data.get("weather_temp_f"),
            weather_data.get("weather_wind_speed_mph"),
            weather_data.get("weather_precip_probability"),
            weather_data.get("weather_rain_3h_in"),
            weather_data.get("weather_snow_3h_in"),
            is_dome,
        )

        row = {
            "game_id": game_id,
            "game_date": game_date,
            "start_time": start_time,
            "home_team": home_team,
            "away_team": away_team,
            "venue_name": ballpark["name"],
            "weather_source": weather_source,
            "weather_source_status": weather_source_status,
            "weather_captured_at": captured_at,
            "weather_forecast_time": weather_data.get("weather_forecast_time", ""),
            "weather_api_endpoint": weather_data.get("weather_api_endpoint", ""),
            "weather_lat": lat,
            "weather_lon": lon,
            "weather_temp_f": weather_data.get("weather_temp_f"),
            "weather_feels_like_f": weather_data.get("weather_feels_like_f"),
            "weather_wind_speed_mph": weather_data.get("weather_wind_speed_mph"),
            "weather_wind_deg": weather_data.get("weather_wind_deg"),
            "weather_precip_probability": weather_data.get(
                "weather_precip_probability"
            ),
            "weather_rain_3h_in": weather_data.get("weather_rain_3h_in"),
            "weather_snow_3h_in": weather_data.get("weather_snow_3h_in"),
            "weather_condition": weather_data.get("weather_condition", ""),
            "weather_is_dome": is_dome,
            "temp_effect": temp_effect,
            "wind_effect": wind_effect,
            "precip_effect": precip_effect,
            "weather_reason": weather_reason,
        }
        rows.append(row)

    result_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result_frame = _clean_dataframe(result_frame)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(output_file, index=False)

    return result_frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_weather_context()
    api_key_exists = bool(_safe_str(os.environ.get("OPENWEATHER_API_KEY", "")))
    status_counts = (
        df["weather_source_status"].value_counts().to_dict()
        if not df.empty and "weather_source_status" in df.columns
        else {}
    )

    summary = {
        "rows": int(len(df)),
        "api_key_present": api_key_exists,
        "source_status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "output_path": "data/weather_context.csv",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
