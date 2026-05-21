"""
牛棚数据客户端（增强版：包含背靠背和近期使用量）
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_bullpen_stats(date_str=None, errors=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # 球队列表
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
                # 获取每日比赛数据以判断背靠背
                schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
                sched_params = {
                    "sportId": 1,
                    "teamId": tid,
                    "startDate": start_date,
                    "endDate": date_str,
                    "gameTypes": "R"
                }
                sched_resp = requests.get(schedule_url, params=sched_params, timeout=10)
                sched_data = sched_resp.json() if sched_resp.status_code == 200 else {}
                game_dates = []
                for d in sched_data.get("dates", []):
                    for g in d.get("games", []):
                        game_dates.append(d["date"])
                # 判断昨天是否有比赛（背靠背）
                yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                back_to_back = 1 if yesterday in game_dates else 0

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
