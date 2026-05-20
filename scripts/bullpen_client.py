import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_bullpen_stats(date_str=None, errors=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # 获取所有球队ID
    try:
        teams_resp = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=10)
        teams_resp.raise_for_status()
        team_ids = [t["id"] for t in teams_resp.json()["teams"]]
    except:
        team_ids = [
            108,109,110,111,112,113,114,115,116,117,118,119,120,121,
            133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,158
        ]

    start_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
    bullpen_data = []

    for tid in team_ids:
        try:
            url = "https://statsapi.mlb.com/api/v1/stats"
            params = {
                "stats": "byDateRange",
                "startDate": start_date,
                "endDate": date_str,
                "teamId": tid,
                "group": "pitching",
                "gameType": "R",
                "limit": 1
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            splits = resp.json().get("stats", [])
            if splits:
                stat = splits[0].get("splits", [{}])[0].get("stat", {})
                bullpen_data.append({
                    "team_id": tid,
                    "bullpen_era": stat.get("era"),
                    "bullpen_whip": stat.get("whip"),
                    "bullpen_pitches": stat.get("numberOfPitches"),
                    "bullpen_innings": stat.get("inningsPitched")
                })
        except Exception as e:
            if errors is not None:
                errors.append(f"Bullpen fetch error for team {tid}: {e}")
    return pd.DataFrame(bullpen_data)
