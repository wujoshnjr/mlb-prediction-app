import requests
import pandas as pd

def fetch_injuries(date_str: str = None, errors: list = None) -> pd.DataFrame:
    # 获取球队列表
    try:
        teams_resp = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=10)
        teams_resp.raise_for_status()
        team_ids = [t["id"] for t in teams_resp.json().get("teams", [])]
    except:
        team_ids = [
            108,109,110,111,112,113,114,115,116,117,118,119,120,121,
            133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,158
        ]
    injury_list = []
    for tid in team_ids:
        try:
            # 修改为正确的 MLB Stats API 伤病端点
            url = f"https://statsapi.mlb.com/api/v1/teams/{tid}/injuries"
            params = {"sportId": 1}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            injuries = resp.json().get("injuries", [])
            for inj in injuries:
                injury_list.append({
                    "team_id": tid,
                    "player_name": inj.get("player", {}).get("fullName"),
                    "injury_type": inj.get("injuryType"),
                    "status": inj.get("status"),
                    "severity": inj.get("severity")
                })
        except Exception as e:
            if errors is not None:
                errors.append(f"Injury fetch error for team {tid}: {e}")
    return pd.DataFrame(injury_list)
