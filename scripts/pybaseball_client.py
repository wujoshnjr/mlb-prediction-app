import pandas as pd
import requests

def fetch_pybaseball(date_str=None, errors=None):
    """直接从 Baseball Savant 抓取 Statcast 数据"""
    if not date_str:
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=7)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
    else:
        start_str = date_str
        end_str = date_str

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
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        cols = ["pitch_type", "release_speed", "events", "game_date"]
        if all(c in df.columns for c in cols):
            sc = df[cols].head(100)
        else:
            sc = df.head(100)

        # 从 standings 拿球队数据（已经由 sportsipy 处理过，这里不再重复）
        # 打击和投手排行榜改用 MLB Stats API
        bat_url = "https://statsapi.mlb.com/api/v1/stats"
        bat_params = {
            "stats": "season",
            "season": 2026,
            "group": "hitting",
            "gameType": "R",
            "limit": 10,
        }
        bat_resp = requests.get(bat_url, params=bat_params, timeout=15)
        bat_resp.raise_for_status()
        bat_splits = bat_resp.json().get("stats", [])
        bat_rows = []
        if bat_splits:
            for split in bat_splits[0].get("splits", []):
                s = split.get("stat", {})
                bat_rows.append({
                    "name": split.get("player", {}).get("fullName"),
                    "avg": s.get("avg"),
                    "home_runs": s.get("homeRuns"),
                    "ops": s.get("ops"),
                })
        bat = pd.DataFrame(bat_rows)

        pitch_url = "https://statsapi.mlb.com/api/v1/stats"
        pitch_params = {
            "stats": "season",
            "season": 2026,
            "group": "pitching",
            "gameType": "R",
            "limit": 10,
        }
        pitch_resp = requests.get(pitch_url, params=pitch_params, timeout=15)
        pitch_resp.raise_for_status()
        pitch_splits = pitch_resp.json().get("stats", [])
        pitch_rows = []
        if pitch_splits:
            for split in pitch_splits[0].get("splits", []):
                s = split.get("stat", {})
                pitch_rows.append({
                    "name": split.get("player", {}).get("fullName"),
                    "era": s.get("era"),
                    "wins": s.get("wins"),
                    "strikeouts": s.get("strikeOuts"),
                })
        pitch = pd.DataFrame(pitch_rows)

        return {
            "statcast_recent": sc,
            "batting_leaders": bat,
            "pitching_leaders": pitch,
        }

    except Exception as e:
        if errors is not None:
            errors.append(f"PyBaseball fetch error: {e}")
        return {
            "statcast_recent": pd.DataFrame(),
            "batting_leaders": pd.DataFrame(),
            "pitching_leaders": pd.DataFrame(),
        }
