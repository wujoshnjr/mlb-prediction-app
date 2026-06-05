from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "home_team",
    "away_team",
    "home_probable_pitcher_id",
    "away_probable_pitcher_id",
    "home_probable_pitcher_name",
    "away_probable_pitcher_name",
    "home_sp_fip",
    "away_sp_fip",
    "sp_fip_diff",
    "home_sp_k_pct",
    "away_sp_k_pct",
    "k_pct_diff",
    "home_sp_bb_pct",
    "away_sp_bb_pct",
    "bb_pct_diff",
    "home_sp_csw_proxy",
    "away_sp_csw_proxy",
    "sp_csw_diff",
    "home_sp_stuff_plus_proxy",
    "away_sp_stuff_plus_proxy",
    "sp_stuff_plus_diff",
    "pitcher_advanced_source_status",
    "pitcher_advanced_reason",
    "pitcher_advanced_captured_at",
]


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


def _season_from_game_date(value: Any) -> int:
    text = _safe_str(value)
    if text:
        try:
            return int(text[:4])
        except Exception:
            pass
    return datetime.now(timezone.utc).year


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


def parse_innings_pitched(value: Any) -> Optional[float]:
    text = _safe_str(value)
    if not text:
        return None

    if "." not in text:
        whole = _safe_float(text)
        return whole

    whole_text, frac_text = text.split(".", 1)
    whole = _safe_int(whole_text)
    if whole is None:
        return None

    frac_text = frac_text.strip()
    if frac_text == "0":
        return float(whole)
    if frac_text == "1":
        return float(whole) + (1.0 / 3.0)
    if frac_text == "2":
        return float(whole) + (2.0 / 3.0)

    fallback = _safe_float(text)
    return fallback


def _request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
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


def _fetch_pitcher_season_stats(
    player_id: int,
    season: int,
    cache: Dict[Tuple[int, int], Tuple[Optional[Dict[str, Any]], str]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    cache_key = (int(player_id), int(season))
    if cache_key in cache:
        return cache[cache_key]

    url = f"{MLB_BASE_URL}/people/{player_id}/stats"
    params = {
        "stats": "season",
        "group": "pitching",
        "season": int(season),
    }

    data, error = _request_json(url, params=params)
    if data is None:
        result = (None, error)
        cache[cache_key] = result
        return result

    stats_blocks = data.get("stats")
    if not isinstance(stats_blocks, list):
        result = (None, "stats_missing")
        cache[cache_key] = result
        return result

    for block in stats_blocks:
        if not isinstance(block, dict):
            continue
        splits = block.get("splits")
        if isinstance(splits, list) and splits:
            stat = splits[0].get("stat")
            if isinstance(stat, dict):
                result = (stat, "")
                cache[cache_key] = result
                return result

    result = (None, "no_pitching_splits")
    cache[cache_key] = result
    return result


def _compute_fip(
    home_runs: Optional[float],
    walks: Optional[float],
    hit_batters: Optional[float],
    strikeouts: Optional[float],
    innings_pitched: Optional[float],
) -> Optional[float]:
    if innings_pitched is None or innings_pitched <= 0:
        return None

    hr = home_runs or 0.0
    bb = walks or 0.0
    hbp = hit_batters or 0.0
    strikeouts_value = strikeouts or 0.0

    value = ((13.0 * hr) + (3.0 * (bb + hbp)) - (2.0 * strikeouts_value)) / innings_pitched + 3.1
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 4)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    value = float(numerator) / float(denominator)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 4)


def _compute_csw_proxy(k_pct: Optional[float], bb_pct: Optional[float]) -> Optional[float]:
    if k_pct is None or bb_pct is None:
        return None
    value = 0.25 + 0.5 * (float(k_pct) - float(bb_pct))
    return round(max(0.15, min(0.40, value)), 4)


def _compute_stuff_plus_proxy(
    k_pct: Optional[float],
    bb_pct: Optional[float],
    fip: Optional[float],
) -> Optional[float]:
    if k_pct is None or bb_pct is None or fip is None:
        return None

    value = (
        100.0
        + 100.0 * (float(k_pct) - 0.22)
        - 80.0 * (float(bb_pct) - 0.08)
        - 3.0 * (float(fip) - 4.2)
    )
    return round(max(70.0, min(130.0, value)), 2)


def _build_pitcher_side(
    pitcher_id: Optional[int],
    pitcher_name: str,
    season: int,
    stats_cache: Dict[Tuple[int, int], Tuple[Optional[Dict[str, Any]], str]],
) -> Dict[str, Any]:
    reasons = []

    if pitcher_id is None:
        return {
            "fip": None,
            "k_pct": None,
            "bb_pct": None,
            "csw_proxy": None,
            "stuff_plus_proxy": None,
            "status": "missing_pitcher_id",
            "reason": f"Missing probable pitcher id for {pitcher_name or 'unknown'}",
        }

    stats, error = _fetch_pitcher_season_stats(pitcher_id, season, stats_cache)
    if stats is None:
        return {
            "fip": None,
            "k_pct": None,
            "bb_pct": None,
            "csw_proxy": None,
            "stuff_plus_proxy": None,
            "status": "api_error",
            "reason": f"Pitching stats unavailable for pitcher_id={pitcher_id}, season={season}: {error}",
        }

    innings_pitched = parse_innings_pitched(stats.get("inningsPitched"))
    home_runs = _safe_float(stats.get("homeRuns"))
    walks = _safe_float(stats.get("baseOnBalls"))
    hit_batters = _safe_float(stats.get("hitBatsmen"))
    if hit_batters is None:
        hit_batters = _safe_float(stats.get("hitByPitch"))
    strikeouts = _safe_float(stats.get("strikeOuts"))

    batters_faced = _safe_float(stats.get("battersFaced"))
    if batters_faced is None and innings_pitched is not None:
        batters_faced = innings_pitched * 3.0
        reasons.append("battersFaced estimated from inningsPitched*3")

    fip = _compute_fip(home_runs, walks, hit_batters, strikeouts, innings_pitched)
    k_pct = _safe_ratio(strikeouts, batters_faced)
    bb_pct = _safe_ratio(walks, batters_faced)
    csw_proxy = _compute_csw_proxy(k_pct, bb_pct)
    stuff_plus_proxy = _compute_stuff_plus_proxy(k_pct, bb_pct, fip)

    missing_metrics = []
    if fip is None:
        missing_metrics.append("fip")
    if k_pct is None:
        missing_metrics.append("k_pct")
    if bb_pct is None:
        missing_metrics.append("bb_pct")
    if csw_proxy is None:
        missing_metrics.append("csw_proxy")
    if stuff_plus_proxy is None:
        missing_metrics.append("stuff_plus_proxy")

    if missing_metrics:
        reasons.append("missing_metrics=" + ",".join(missing_metrics))
        status = "partial"
    else:
        status = "ok"

    return {
        "fip": fip,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "csw_proxy": csw_proxy,
        "stuff_plus_proxy": stuff_plus_proxy,
        "status": status,
        "reason": "; ".join(reasons) if reasons else "season pitching stats available",
    }


def _diff(home_value: Optional[float], away_value: Optional[float]) -> float:
    if home_value is None or away_value is None:
        return 0.0
    return round(float(home_value) - float(away_value), 6)


def _combined_status(home_status: str, away_status: str) -> str:
    if home_status == "ok" and away_status == "ok":
        return "ok"
    if home_status in {"ok", "partial"} or away_status in {"ok", "partial"}:
        return "partial"
    return "unavailable"


def build_pitcher_advanced_context(
    daily_context_path: str = "data/daily_game_context.csv",
    output_path: Optional[str] = "data/pitcher_advanced_context.csv",
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

    required_columns = [
        "game_id",
        "game_date",
        "start_time",
        "captured_at",
        "home_team",
        "away_team",
        "home_probable_pitcher_id",
        "away_probable_pitcher_id",
        "home_probable_pitcher_name",
        "away_probable_pitcher_name",
    ]

    for column in required_columns:
        if column not in context_frame.columns:
            context_frame[column] = ""

    context_frame = context_frame.dropna(subset=["game_id"]).copy()
    if context_frame.empty:
        return _empty_output(output_path)

    latest_frame = _latest_context_rows(context_frame)
    stats_cache: Dict[Tuple[int, int], Tuple[Optional[Dict[str, Any]], str]] = {}
    rows = []

    for _, row in latest_frame.iterrows():
        game_id = _safe_str(row.get("game_id"))
        game_date = _safe_str(row.get("game_date"))[:10]
        season = _season_from_game_date(game_date)

        home_pid = _safe_int(row.get("home_probable_pitcher_id"))
        away_pid = _safe_int(row.get("away_probable_pitcher_id"))
        home_name = _safe_str(row.get("home_probable_pitcher_name"))
        away_name = _safe_str(row.get("away_probable_pitcher_name"))

        home_result = _build_pitcher_side(home_pid, home_name, season, stats_cache)
        away_result = _build_pitcher_side(away_pid, away_name, season, stats_cache)

        source_status = _combined_status(home_result["status"], away_result["status"])
        reason = (
            f"home={home_result['status']} ({home_result['reason']}); "
            f"away={away_result['status']} ({away_result['reason']})"
        )

        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "start_time": _safe_str(row.get("start_time")),
                "captured_at": _safe_str(row.get("captured_at")),
                "home_team": _safe_str(row.get("home_team")),
                "away_team": _safe_str(row.get("away_team")),
                "home_probable_pitcher_id": home_pid,
                "away_probable_pitcher_id": away_pid,
                "home_probable_pitcher_name": home_name,
                "away_probable_pitcher_name": away_name,
                "home_sp_fip": home_result["fip"],
                "away_sp_fip": away_result["fip"],
                "sp_fip_diff": _diff(home_result["fip"], away_result["fip"]),
                "home_sp_k_pct": home_result["k_pct"],
                "away_sp_k_pct": away_result["k_pct"],
                "k_pct_diff": _diff(home_result["k_pct"], away_result["k_pct"]),
                "home_sp_bb_pct": home_result["bb_pct"],
                "away_sp_bb_pct": away_result["bb_pct"],
                "bb_pct_diff": _diff(home_result["bb_pct"], away_result["bb_pct"]),
                "home_sp_csw_proxy": home_result["csw_proxy"],
                "away_sp_csw_proxy": away_result["csw_proxy"],
                "sp_csw_diff": _diff(home_result["csw_proxy"], away_result["csw_proxy"]),
                "home_sp_stuff_plus_proxy": home_result["stuff_plus_proxy"],
                "away_sp_stuff_plus_proxy": away_result["stuff_plus_proxy"],
                "sp_stuff_plus_diff": _diff(
                    home_result["stuff_plus_proxy"],
                    away_result["stuff_plus_proxy"],
                ),
                "pitcher_advanced_source_status": source_status,
                "pitcher_advanced_reason": reason,
                "pitcher_advanced_captured_at": captured_at,
            }
        )

    output_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_frame = _clean_dataframe(output_frame)

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(destination, index=False)

    return output_frame


if __name__ == "__main__":
    df = build_pitcher_advanced_context()
    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "ok": int((df["pitcher_advanced_source_status"] == "ok").sum()) if not df.empty else 0,
                "partial": int((df["pitcher_advanced_source_status"] == "partial").sum()) if not df.empty else 0,
                "output_path": "data/pitcher_advanced_context.csv",
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
