from __future__ import annotations

import glob
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15
RECENT_HISTORY_LOOKBACK_DAYS = 45
MAX_SAFE_REST_DAYS = 14
OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "home_team",
    "away_team",
    "home_lag30_winrate",
    "away_lag30_winrate",
    "lag30_winrate_diff",
    "home_lag30_runs_for_avg",
    "away_lag30_runs_for_avg",
    "home_lag30_runs_against_avg",
    "away_lag30_runs_against_avg",
    "lag30_runs_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_back2back",
    "away_back2back",
    "back2back_diff",
    "home_log5_strength",
    "away_log5_strength",
    "log5_prob",
    "home_elo_momentum_7d",
    "away_elo_momentum_7d",
    "elo_momentum_7d",
    "home_elo_momentum_30d",
    "away_elo_momentum_30d",
    "elo_momentum_30d",
    "team_form_source_status",
    "team_form_reason",
    "team_form_captured_at",
]

TEAM_ALIASES: Dict[str, str] = {
    "ari": "Arizona Diamondbacks",
    "diamondbacks": "Arizona Diamondbacks",
    "dbacks": "Arizona Diamondbacks",
    "d-backs": "Arizona Diamondbacks",
    "ath": "Athletics",
    "oak": "Athletics",
    "athletics": "Athletics",
    "oakland athletics": "Athletics",
    "a's": "Athletics",
    "as": "Athletics",
    "atl": "Atlanta Braves",
    "braves": "Atlanta Braves",
    "bal": "Baltimore Orioles",
    "orioles": "Baltimore Orioles",
    "bos": "Boston Red Sox",
    "red sox": "Boston Red Sox",
    "chc": "Chicago Cubs",
    "cubs": "Chicago Cubs",
    "cws": "Chicago White Sox",
    "white sox": "Chicago White Sox",
    "cin": "Cincinnati Reds",
    "reds": "Cincinnati Reds",
    "cle": "Cleveland Guardians",
    "guardians": "Cleveland Guardians",
    "col": "Colorado Rockies",
    "rockies": "Colorado Rockies",
    "det": "Detroit Tigers",
    "tigers": "Detroit Tigers",
    "hou": "Houston Astros",
    "astros": "Houston Astros",
    "kc": "Kansas City Royals",
    "kcr": "Kansas City Royals",
    "royals": "Kansas City Royals",
    "laa": "Los Angeles Angels",
    "angels": "Los Angeles Angels",
    "lad": "Los Angeles Dodgers",
    "dodgers": "Los Angeles Dodgers",
    "mia": "Miami Marlins",
    "marlins": "Miami Marlins",
    "mil": "Milwaukee Brewers",
    "brewers": "Milwaukee Brewers",
    "min": "Minnesota Twins",
    "twins": "Minnesota Twins",
    "nym": "New York Mets",
    "mets": "New York Mets",
    "nyy": "New York Yankees",
    "yankees": "New York Yankees",
    "phi": "Philadelphia Phillies",
    "phillies": "Philadelphia Phillies",
    "pit": "Pittsburgh Pirates",
    "pirates": "Pittsburgh Pirates",
    "sd": "San Diego Padres",
    "sdp": "San Diego Padres",
    "padres": "San Diego Padres",
    "sf": "San Francisco Giants",
    "sfg": "San Francisco Giants",
    "giants": "San Francisco Giants",
    "sea": "Seattle Mariners",
    "mariners": "Seattle Mariners",
    "stl": "St. Louis Cardinals",
    "cardinals": "St. Louis Cardinals",
    "st louis cardinals": "St. Louis Cardinals",
    "tb": "Tampa Bay Rays",
    "tbr": "Tampa Bay Rays",
    "rays": "Tampa Bay Rays",
    "tex": "Texas Rangers",
    "rangers": "Texas Rangers",
    "tor": "Toronto Blue Jays",
    "blue jays": "Toronto Blue Jays",
    "wsh": "Washington Nationals",
    "was": "Washington Nationals",
    "nationals": "Washington Nationals",
}

TEAM_ID_MAP: Dict[str, int] = {
    "Arizona Diamondbacks": 109,
    "Athletics": 133,
    "Atlanta Braves": 144,
    "Baltimore Orioles": 110,
    "Boston Red Sox": 111,
    "Chicago Cubs": 112,
    "Chicago White Sox": 145,
    "Cincinnati Reds": 113,
    "Cleveland Guardians": 114,
    "Colorado Rockies": 115,
    "Detroit Tigers": 116,
    "Houston Astros": 117,
    "Kansas City Royals": 118,
    "Los Angeles Angels": 108,
    "Los Angeles Dodgers": 119,
    "Miami Marlins": 146,
    "Milwaukee Brewers": 158,
    "Minnesota Twins": 142,
    "New York Mets": 121,
    "New York Yankees": 147,
    "Philadelphia Phillies": 143,
    "Pittsburgh Pirates": 134,
    "San Diego Padres": 135,
    "San Francisco Giants": 137,
    "Seattle Mariners": 136,
    "St. Louis Cardinals": 138,
    "Tampa Bay Rays": 139,
    "Texas Rangers": 140,
    "Toronto Blue Jays": 141,
    "Washington Nationals": 120,
}


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


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _normalise_key(value: Any) -> str:
    text = _safe_str(value).lower()
    text = text.replace("&", "and").replace(".", "").replace("-", " ")
    text = text.replace("_", " ").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_team(value: Any) -> str:
    raw = _safe_str(value)
    key = _normalise_key(raw)
    if not key:
        return ""

    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]

    for alias, full in TEAM_ALIASES.items():
        if key == alias or key in alias or alias in key:
            return full

    return raw


def team_id_from_name(value: Any) -> Optional[int]:
    team = normalise_team(value)
    return TEAM_ID_MAP.get(team)


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    text = _safe_str(value)
    if not text:
        return None

    parsed = pd.to_datetime(text[:10], errors="coerce", utc=True)
    if pd.isna(parsed):
        return None

    return parsed.normalize()


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


def _standardize_finalized_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    working = frame.copy()

    date_col = None
    for candidate in ("game_date", "date", "scheduled_date"):
        if candidate in working.columns:
            date_col = candidate
            break

    if date_col is None:
        return pd.DataFrame()

    required = ["home_team", "away_team", "home_score", "away_score"]
    for column in required:
        if column not in working.columns:
            return pd.DataFrame()

    output = pd.DataFrame()
    output["game_date"] = pd.to_datetime(
        working[date_col].astype(str).str[:10],
        errors="coerce",
        utc=True,
    )
    output["home_team"] = working["home_team"].map(normalise_team)
    output["away_team"] = working["away_team"].map(normalise_team)
    output["home_score"] = pd.to_numeric(working["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(working["away_score"], errors="coerce")
    output["source"] = source_name

    if "home_elo" in working.columns:
        output["home_elo"] = pd.to_numeric(working["home_elo"], errors="coerce")
    if "away_elo" in working.columns:
        output["away_elo"] = pd.to_numeric(working["away_elo"], errors="coerce")

    output = output.dropna(
        subset=["game_date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    output["home_score"] = output["home_score"].astype(int)
    output["away_score"] = output["away_score"].astype(int)
    output["home_win"] = output["home_score"] > output["away_score"]

    return output


def _load_history_frames(
    finalized_path: str,
    historical_predictions_path: str,
    historical_glob: str,
) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    notes: List[str] = []

    paths = [(finalized_path, "finalized_games"), (historical_predictions_path, "historical_predictions")]

    for path_text, source_name in paths:
        path = Path(path_text)
        if not path.exists():
            notes.append(f"{source_name}_missing")
            continue
        try:
            raw = pd.read_csv(path)
            standardized = _standardize_finalized_frame(raw, source_name)
            if standardized.empty:
                notes.append(f"{source_name}_no_usable_rows")
            else:
                frames.append(standardized)
        except Exception as exc:
            notes.append(f"{source_name}_read_failed={exc}")

    for path_text in glob.glob(historical_glob):
        path = Path(path_text)
        try:
            raw = pd.read_csv(path)
            standardized = _standardize_finalized_frame(raw, f"historical/{path.name}")
            if not standardized.empty:
                frames.append(standardized)
        except Exception as exc:
            notes.append(f"historical_file_failed={path.name}:{exc}")

    if not frames:
        return pd.DataFrame(), notes

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "home_score", "away_score"],
        keep="last",
    )
    return combined.sort_values("game_date").reset_index(drop=True), notes


def _request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return None, f"request_failed: {exc}"

    if response.status_code != 200:
        return None, f"http_{response.status_code}: {response.text[:160]}"

    try:
        data = response.json()
    except Exception as exc:
        return None, f"json_error: {exc}"

    if not isinstance(data, dict):
        return None, "json_not_object"

    return data, ""


def _fetch_recent_mlb_team_games(
    team_name: str,
    game_date: pd.Timestamp,
    cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, str]],
) -> Tuple[pd.DataFrame, str]:
    team = normalise_team(team_name)
    team_id = team_id_from_name(team)

    if team_id is None:
        return pd.DataFrame(), f"missing_team_id:{team_name}"

    end_date = (game_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (game_date - pd.Timedelta(days=RECENT_HISTORY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    cache_key = (str(team_id), f"{start_date}:{end_date}")

    if cache_key in cache:
        return cache[cache_key]

    data, error = _request_json(
        f"{MLB_BASE_URL}/schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date,
            "endDate": end_date,
        },
    )

    if data is None:
        result = (pd.DataFrame(), f"schedule_api_failed:{error}")
        cache[cache_key] = result
        return result

    rows: List[Dict[str, Any]] = []

    for date_block in data.get("dates", []):
        if not isinstance(date_block, dict):
            continue

        for game in date_block.get("games", []):
            if not isinstance(game, dict):
                continue

            status = game.get("status", {})
            detailed_state = _safe_str(status.get("detailedState")).lower()
            coded_state = _safe_str(status.get("codedGameState")).lower()

            if "final" not in detailed_state and coded_state not in {"f", "fr", "o"}:
                continue

            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})

            home_team = normalise_team(
                home.get("team", {}).get("name")
                if isinstance(home.get("team"), dict)
                else ""
            )
            away_team = normalise_team(
                away.get("team", {}).get("name")
                if isinstance(away.get("team"), dict)
                else ""
            )

            home_score = _safe_int(home.get("score"))
            away_score = _safe_int(away.get("score"))

            official_date = _safe_str(game.get("officialDate"))
            game_dt = _parse_date(official_date)

            if (
                game_dt is None
                or not home_team
                or not away_team
                or home_score is None
                or away_score is None
            ):
                continue

            rows.append(
                {
                    "game_date": game_dt,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "source": "mlb_stats_schedule_recent",
                    "home_win": bool(home_score > away_score),
                }
            )

    frame = pd.DataFrame(rows)
    result = (frame, "ok" if not frame.empty else "schedule_no_recent_final_games")
    cache[cache_key] = result
    return result


def _augment_recent_history_for_game(
    base_history: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    schedule_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, str]],
) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    notes: List[str] = []

    if base_history is not None and not base_history.empty:
        frames.append(base_history)

    for team in (home_team, away_team):
        recent_frame, note = _fetch_recent_mlb_team_games(team, game_date, schedule_cache)
        notes.append(f"{team}:{note}")
        if not recent_frame.empty:
            frames.append(recent_frame)

    if not frames:
        return pd.DataFrame(), notes

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "home_score", "away_score"],
        keep="last",
    )
    combined = combined.sort_values("game_date").reset_index(drop=True)

    return combined, notes


def _team_games(prior: pd.DataFrame, team: str) -> pd.DataFrame:
    return prior[
        (prior["home_team"] == team)
        | (prior["away_team"] == team)
    ].copy()


def _team_window_stats(prior: pd.DataFrame, team: str, game_date: pd.Timestamp, days: int) -> Dict[str, Any]:
    team_all = _team_games(prior, team)

    if team_all.empty:
        return {
            "games": 0,
            "winrate": None,
            "runs_for_avg": None,
            "runs_against_avg": None,
            "rest_days": None,
            "back2back": False,
            "momentum_proxy": 0.0,
        }

    cutoff = game_date - pd.Timedelta(days=int(days))
    window = team_all[team_all["game_date"] >= cutoff].copy()

    if window.empty:
        last_date = team_all["game_date"].max()
        rest_days = int((game_date - last_date).days) if pd.notna(last_date) else None
        if rest_days is not None and rest_days > MAX_SAFE_REST_DAYS:
            rest_days = None
        return {
            "games": 0,
            "winrate": None,
            "runs_for_avg": None,
            "runs_against_avg": None,
            "rest_days": rest_days,
            "back2back": bool(rest_days is not None and rest_days <= 1),
            "momentum_proxy": 0.0,
        }

    wins: List[bool] = []
    runs_for: List[int] = []
    runs_against: List[int] = []

    for _, game in window.iterrows():
        if game["home_team"] == team:
            wins.append(bool(game["home_score"] > game["away_score"]))
            runs_for.append(int(game["home_score"]))
            runs_against.append(int(game["away_score"]))
        elif game["away_team"] == team:
            wins.append(bool(game["away_score"] > game["home_score"]))
            runs_for.append(int(game["away_score"]))
            runs_against.append(int(game["home_score"]))

    winrate = float(sum(wins) / len(wins)) if wins else None
    runs_for_avg = float(sum(runs_for) / len(runs_for)) if runs_for else None
    runs_against_avg = float(sum(runs_against) / len(runs_against)) if runs_against else None

    last_date = team_all["game_date"].max()
    rest_days = int((game_date - last_date).days) if pd.notna(last_date) else None
    if rest_days is not None and rest_days > MAX_SAFE_REST_DAYS:
        rest_days = None
    back2back = bool(rest_days is not None and rest_days <= 1)

    return {
        "games": int(len(window)),
        "winrate": winrate,
        "runs_for_avg": runs_for_avg,
        "runs_against_avg": runs_against_avg,
        "rest_days": rest_days,
        "back2back": back2back,
        "momentum_proxy": (winrate - 0.5) if winrate is not None else 0.0,
    }


def _log5(home_strength: Optional[float], away_strength: Optional[float]) -> float:
    h = 0.5 if home_strength is None else float(home_strength)
    a = 0.5 if away_strength is None else float(away_strength)

    h = max(0.01, min(0.99, h))
    a = max(0.01, min(0.99, a))

    denominator = h * (1.0 - a) + (1.0 - h) * a
    if denominator <= 0:
        return 0.5

    return float((h * (1.0 - a)) / denominator)


def build_team_form_context(
    daily_context_path: str = "data/daily_game_context.csv",
    finalized_path: str = "data/finalized_games.csv",
    historical_predictions_path: str = "data/historical_predictions.csv",
    historical_glob: str = "data/historical/*.csv",
    output_path: Optional[str] = "data/team_form_context.csv",
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

    for column in ("game_id", "game_date", "start_time", "captured_at", "home_team", "away_team"):
        if column not in context_frame.columns:
            context_frame[column] = ""

    context_frame = context_frame.dropna(subset=["game_id"]).copy()
    if context_frame.empty:
        return _empty_output(output_path)

    latest_frame = _latest_context_rows(context_frame)
    history_frame, history_notes = _load_history_frames(
        finalized_path=finalized_path,
        historical_predictions_path=historical_predictions_path,
        historical_glob=historical_glob,
    )

    rows: List[Dict[str, Any]] = []

    for _, row in latest_frame.iterrows():
        game_id = _safe_str(row.get("game_id"))
        game_date_text = _safe_str(row.get("game_date"))[:10]
        game_date = _parse_date(game_date_text)

        home_raw = _safe_str(row.get("home_team"))
        away_raw = _safe_str(row.get("away_team"))
        home_team = normalise_team(home_raw)
        away_team = normalise_team(away_raw)

        output = {column: "" for column in OUTPUT_COLUMNS}
        output.update(
            {
                "game_id": game_id,
                "game_date": game_date_text,
                "start_time": _safe_str(row.get("start_time")),
                "captured_at": _safe_str(row.get("captured_at")),
                "home_team": home_raw,
                "away_team": away_raw,
                "home_lag30_winrate": "",
                "away_lag30_winrate": "",
                "lag30_winrate_diff": 0.0,
                "home_lag30_runs_for_avg": "",
                "away_lag30_runs_for_avg": "",
                "home_lag30_runs_against_avg": "",
                "away_lag30_runs_against_avg": "",
                "lag30_runs_diff": 0.0,
                "home_rest_days": "",
                "away_rest_days": "",
                "rest_diff": 0.0,
                "home_back2back": False,
                "away_back2back": False,
                "back2back_diff": 0,
                "home_log5_strength": 0.5,
                "away_log5_strength": 0.5,
                "log5_prob": 0.5,
                "home_elo_momentum_7d": 0.0,
                "away_elo_momentum_7d": 0.0,
                "elo_momentum_7d": 0.0,
                "home_elo_momentum_30d": 0.0,
                "away_elo_momentum_30d": 0.0,
                "elo_momentum_30d": 0.0,
                "team_form_source_status": "unavailable",
                "team_form_reason": "",
                "team_form_captured_at": captured_at,
            }
        )

        if game_date is None:
            output["team_form_reason"] = "game_date_unparseable"
            rows.append(output)
            continue

        if history_frame.empty:
            output["team_form_source_status"] = "no_history"
            output["team_form_reason"] = "No usable finalized or historical game rows; " + "; ".join(history_notes)
            rows.append(output)
            continue

        prior = history_frame[history_frame["game_date"] < game_date].copy()

        if prior.empty:
            output["team_form_source_status"] = "sparse_history"
            output["team_form_reason"] = "No prior finalized games before game_date"
            rows.append(output)
            continue

        home30 = _team_window_stats(prior, home_team, game_date, 30)
        away30 = _team_window_stats(prior, away_team, game_date, 30)
        home7 = _team_window_stats(prior, home_team, game_date, 7)
        away7 = _team_window_stats(prior, away_team, game_date, 7)

        if home30["winrate"] is not None:
            output["home_lag30_winrate"] = round(float(home30["winrate"]), 4)
            output["home_log5_strength"] = round(float(home30["winrate"]), 4)
        if away30["winrate"] is not None:
            output["away_lag30_winrate"] = round(float(away30["winrate"]), 4)
            output["away_log5_strength"] = round(float(away30["winrate"]), 4)

        if home30["winrate"] is not None and away30["winrate"] is not None:
            output["lag30_winrate_diff"] = round(float(home30["winrate"] - away30["winrate"]), 4)

        if home30["runs_for_avg"] is not None:
            output["home_lag30_runs_for_avg"] = round(float(home30["runs_for_avg"]), 4)
        if away30["runs_for_avg"] is not None:
            output["away_lag30_runs_for_avg"] = round(float(away30["runs_for_avg"]), 4)
        if home30["runs_against_avg"] is not None:
            output["home_lag30_runs_against_avg"] = round(float(home30["runs_against_avg"]), 4)
        if away30["runs_against_avg"] is not None:
            output["away_lag30_runs_against_avg"] = round(float(away30["runs_against_avg"]), 4)

        if (
            home30["runs_for_avg"] is not None
            and home30["runs_against_avg"] is not None
            and away30["runs_for_avg"] is not None
            and away30["runs_against_avg"] is not None
        ):
            output["lag30_runs_diff"] = round(
                float(
                    (home30["runs_for_avg"] - home30["runs_against_avg"])
                    - (away30["runs_for_avg"] - away30["runs_against_avg"])
                ),
                4,
            )

        if home30["rest_days"] is not None:
            output["home_rest_days"] = int(home30["rest_days"])
        if away30["rest_days"] is not None:
            output["away_rest_days"] = int(away30["rest_days"])

        if home30["rest_days"] is not None and away30["rest_days"] is not None:
            output["rest_diff"] = int(home30["rest_days"] - away30["rest_days"])

        output["home_back2back"] = bool(home30["back2back"])
        output["away_back2back"] = bool(away30["back2back"])
        output["back2back_diff"] = int(output["home_back2back"]) - int(output["away_back2back"])

        output["log5_prob"] = round(
            _log5(
                home30["winrate"],
                away30["winrate"],
            ),
            4,
        )

        output["home_elo_momentum_7d"] = round(float(home7["momentum_proxy"]), 4)
        output["away_elo_momentum_7d"] = round(float(away7["momentum_proxy"]), 4)
        output["elo_momentum_7d"] = round(float(home7["momentum_proxy"] - away7["momentum_proxy"]), 4)

        output["home_elo_momentum_30d"] = round(float(home30["momentum_proxy"]), 4)
        output["away_elo_momentum_30d"] = round(float(away30["momentum_proxy"]), 4)
        output["elo_momentum_30d"] = round(float(home30["momentum_proxy"] - away30["momentum_proxy"]), 4)

        reason_parts = []
        if home30["games"] == 0:
            reason_parts.append("home_lag30_missing")
        if away30["games"] == 0:
            reason_parts.append("away_lag30_missing")
        if history_notes:
            reason_parts.append("history_notes=" + ",".join(history_notes[:5]))

        if home30["games"] > 0 and away30["games"] > 0:
            output["team_form_source_status"] = "ok"
        else:
            output["team_form_source_status"] = "sparse_history"

        output["team_form_reason"] = "; ".join(reason_parts) if reason_parts else "shift_by_1_team_form_available"
        rows.append(output)

    output_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_frame = _clean_dataframe(output_frame)

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(destination, index=False)

    return output_frame


if __name__ == "__main__":
    df = build_team_form_context()
    status_counts = (
        df["team_form_source_status"].value_counts().to_dict()
        if not df.empty and "team_form_source_status" in df.columns
        else {}
    )
    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "status_counts": {str(key): int(value) for key, value in status_counts.items()},
                "output_path": "data/team_form_context.csv",
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
