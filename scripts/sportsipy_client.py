"""
球隊戰績客户端（使用 MLB Stats API standings + team stats）
获取胜率、得分、失分，用于计算 Pythagorean Record
"""
import pandas as pd
import requests

def fetch_sportsipy(date_str=None, errors=None):
    year = 2026
    # 1. 获取 standings（胜率、胜差）
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
            errors.append(f"Sportsipy standings error: {e}")
        df_teams = pd.DataFrame()

    # 2. 获取球队赛季总得分/失分（用于 Pythagorean）
    try:
        stats_url = "https://statsapi.mlb.com/api/v1/stats"
        stats_params = {
            "stats": "season",
            "season": year,
            "group": "hitting",
            "gameType": "R",
            "limit": 30
        }
        resp = requests.get(stats_url, params=stats_params, timeout=15)
        resp.raise_for_status()
        splits = resp.json().get("stats", [])
        runs_scored_map = {}
        if splits:
            for split in splits[0].get("splits", []):
                team_name = split.get("team", {}).get("name", "")
                runs = split.get("stat", {}).get("runs", 0)
                runs_scored_map[team_name] = runs

        # 投球统计获取失分
        pitch_params = {
            "stats": "season",
            "season": year,
            "group": "pitching",
            "gameType": "R",
            "limit": 30
        }
        resp = requests.get(stats_url, params=pitch_params, timeout=15)
        resp.raise_for_status()
        splits = resp.json().get("stats", [])
        runs_allowed_map = {}
        if splits:
            for split in splits[0].get("splits", []):
                team_name = split.get("team", {}).get("name", "")
                runs = split.get("stat", {}).get("runs", 0)
                runs_allowed_map[team_name] = runs

        # 将得分/失分加入 df_teams
        if not df_teams.empty:
            df_teams['runs_scored'] = df_teams['name'].map(runs_scored_map).fillna(400)
            df_teams['runs_allowed'] = df_teams['name'].map(runs_allowed_map).fillna(400)
    except Exception as e:
        if errors is not None:
            errors.append(f"Sportsipy runs error: {e}")
        if not df_teams.empty:
            df_teams['runs_scored'] = 400
            df_teams['runs_allowed'] = 400

    # 球员示例（静态）
    player_info = {
        "name": "Shohei Ohtani",
        "home_runs": None,
        "avg": None,
        "ops": None,
        "note": "Player stats via Stats API temporarily disabled"
    }
    return {"teams": df_teams, "player_example": player_info}
