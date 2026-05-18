"""
球队与球员数据客户端（使用 MLB Stats API 和 Baseball Savant）
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO

def fetch_pybaseball(date_str: str = None, errors: list = None) -> dict:
    if not date_str:
        end = datetime.now()
        start = end - timedelta(days=7)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
    else:
        start_str = date_str
        end_str = date_str

    # Statcast 数据
    try:
        url = (
            "https://baseballsavant.mlb.com/statcast_search/csv?"
            "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
            "hfGT=R%7C&hfC=&hfSea=2026%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&"
            "pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}&"
            "game_date_lt={}&hfFlag=&metric_1=&hfInn=&min_pitches=0&"
            "min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&type=details"
        ).format(start_str, end_str)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        sc = pd.read_csv(StringIO(resp.text)).head(100)
    except Exception as e:
        if errors is not None:
            errors.append(f"PyBaseball Statcast error: {e}")
        sc = pd.DataFrame()

    # 打击/投手排行榜
    try:
        bat_url = "https://statsapi.mlb.com/api/v1/stats"
        bat_params = {"stats": "season", "season": 2026, "group": "hitting", "gameType": "R", "limit": 10}
        bat_resp = requests.get(bat_url, params=bat_params, timeout=15)
        bat_resp.raise_for_status()
        bat_splits = bat_resp.json().get("stats", [])
        bat_rows = []
        if bat_splits:
            for split in bat_splits[0].get("splits", []):
                s = split.get("stat", {})
                bat_rows.append({"name": split.get("player", {}).get("fullName"), "avg": s.get("avg"), "home_runs": s.get("homeRuns"), "ops": s.get("ops")})
        bat = pd.DataFrame(bat_rows)
    except:
        bat = pd.DataFrame()

    try:
        pitch_url = "https://statsapi.mlb.com/api/v1/stats"
        pitch_params = {"stats": "season", "season": 2026, "group": "pitching", "gameType": "R", "limit": 10}
        pitch_resp = requests.get(pitch_url, params=pitch_params, timeout=15)
        pitch_resp.raise_for_status()
        pitch_splits = pitch_resp.json().get("stats", [])
        pitch_rows = []
        if pitch_splits:
            for split in pitch_splits[0].get("splits", []):
                s = split.get("stat", {})
                pitch_rows.append({"name": split.get("player", {}).get("fullName"), "era": s.get("era"), "wins": s.get("wins"), "strikeouts": s.get("strikeOuts")})
        pitch = pd.DataFrame(pitch_rows)
    except:
        pitch = pd.DataFrame()

    return {"statcast_recent": sc, "batting_leaders": bat, "pitching_leaders": pitch}
