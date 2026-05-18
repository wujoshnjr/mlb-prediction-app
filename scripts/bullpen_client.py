import requests
import pandas as pd
from datetime import datetime

def fetch_bullpen_stats(date_str: str = None, errors: list = None) -> pd.DataFrame:
    """
    获取各球队近3天牛棚使用情况（投球数、ERA等）
    通过 MLB Stats API 的 stats 端点按球队查询
    """
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

    bullpen_data = []
    for tid in team_ids:
        try:
            # 获取球队投球统计（可筛选最近3天）
            url = "https://statsapi.mlb.com/api/v1/stats"
            params = {
                "stats": "byDateRange",
                "startDate": (datetime.strptime(date_str, "%Y-%m-%d") - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
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
        except:
            pass
    return pd.DataFrame(bullpen_data)
