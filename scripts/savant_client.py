import pandas as pd
import requests

def fetch_sportsipy(date_str=None, errors=None):
    year = 2026
    # 球队战绩
    try:
        url = "https://statsapi.mlb.com/api/v1/standings"
        params = {"leagueId": "103,104", "season": year}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        team_list = []
        for record in data.get("records", []):
            for team_record in record.get("teamRecords", []):
                team = team_record.get("team", {})
                team_list.append({
                    "name": team.get("name"),
                    "wins": team_record.get("wins"),
                    "losses": team_record.get("losses"),
                    "win_pct": team_record.get("winningPercentage"),
                    "gb": team_record.get("gamesBack"),
                })
        df_teams = pd.DataFrame(team_list)
    except Exception as e:
        if errors is not None:
            errors.append(f"Sportsipy teams error: {e}")
        df_teams = pd.DataFrame()

    # 球员示例 (静态)
    player_info = {
        "name": "Shohei Ohtani",
        "home_runs": None,
        "avg": None,
        "ops": None,
        "note": "Player stats via Stats API temporarily disabled"
    }
    return {"teams": df_teams, "player_example": player_info}
