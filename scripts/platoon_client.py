"""
Platoon 拆分客户端
获取所有球队对左/右投手的打击数据 (OPS)
"""
import requests
import pandas as pd

def fetch_platoon_splits(season=2026, errors=None):
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "season",
        "season": season,
        "group": "hitting",
        "gameType": "R",
        "split": "vsLhp,vsRhp",
        "limit": 60
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if errors is not None:
            errors.append(f"Platoon fetch error: {e}")
        return pd.DataFrame()

    rows = []
    for stat_group in data.get("stats", []):
        split_type = stat_group.get("split", "vsRhp")
        for s in stat_group.get("splits", []):
            team_name = s.get("team", {}).get("name", "")
            stat = s.get("stat", {})
            rows.append({
                "team_name": team_name,
                "split": split_type,
                "ops": stat.get("ops", "0.700"),
                "woba": stat.get("woba", "0.310")
            })
    return pd.DataFrame(rows)
