import pandas as pd
import requests

def fetch_sportsipy(date_str=None, errors=None):
    year = 2026
    try:
        # 直接从 MLB Stats API 获取球队战绩
        url = "https://statsapi.mlb.com/api/v1/standings"
        params = {"leagueId": "103,104", "season": year}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        standings_data = resp.json()

        team_list = []
        for record in standings_data.get("records", []):
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

    # 球员示例（大谷翔平）
    player_info = {}
    try:
        player_url = "https://statsapi.mlb.com/api/v1/people"
        player_params = {"search": "Shohei Ohtani"}
        player_resp = requests.get(player_url, params=player_params, timeout=15)
        player_resp.raise_for_status()
        people = player_resp.json().get("people", [])
        if people:
            player_id = people[0].get("id")
            stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
            stats_params = {"stats": "season", "season": year, "gameType": "R"}
            stats_resp = requests.get(stats_url, params=stats_params, timeout=15)
            stats_resp.raise_for_status()
            splits = stats_resp.json().get("stats", [])
            if splits:
                stat = splits[0].get("splits", [{}])[0].get("stat", {})
                player_info = {
                    "name": people[0].get("fullName"),
                    "home_runs": stat.get("homeRuns"),
                    "avg": stat.get("avg"),
                    "ops": stat.get("ops"),
                }
            else:
                player_info = {"name": people[0].get("fullName"), "note": "stats not available"}
    except Exception as e:
        if errors is not None:
            errors.append(f"Sportsipy player error: {e}")
        player_info = {"error": str(e)}

    return {"teams": df_teams, "player_example": player_info}
