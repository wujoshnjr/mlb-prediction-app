"""
牛棚数据客户端（增强版：包含背靠背标记，增加超时时间）
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_bullpen_stats(date_str=None, errors=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # 获取所有球队ID
    try:
        teams_resp = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=15)
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
            # 近3天团队投球统计
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
            resp = requests.get(url, params=params, timeout=30)  # 将 timeout 增加到 30 秒
            resp.raise_for_status()
            stat = {}
            if resp.status_code == 200:
                splits = resp.json().get("stats", [])
                if splits:
                    stat = splits[0].get("splits", [{}])[0].get("stat", {})

            # 背靠背判定：昨天是否有比赛
            yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            sched_url = "https://statsapi.mlb.com/api/v1/schedule"
            sched_params = {
                "sportId": 1,
                "teamId": tid,
                "startDate": yesterday,
                "endDate": yesterday,
                "gameTypes": "R"
            }
            sched_resp = requests.get(sched_url, params=sched_params, timeout=30)  # timeout 增加到 30 秒
            back_to_back = 0
            if sched_resp.status_code == 200:
                sched_data = sched_resp.json()
                if sched_data.get("totalGames", 0) > 0:
                    back_to_back = 1

            bullpen_data.append({
                "team_id": tid,
                "bullpen_era": stat.get("era"),
                "bullpen_whip": stat.get("whip"),
                "bullpen_pitches": stat.get("numberOfPitches"),
                "bullpen_innings": stat.get("inningsPitched"),
                "back_to_back": back_to_back
            })
        except Exception as e:
            if errors is not None:
                errors.append(f"Bullpen fetch error for team {tid}: {e}")
    return pd.DataFrame(bullpen_data)
