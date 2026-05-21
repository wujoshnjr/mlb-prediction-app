import pandas as pd
import requests

def fetch_sportsipy(date_str=None, errors=None):
    year = 2026
    # 1. 获取 standings
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

    # 2. 获取团队总得分/失分
    try:
        stats_url = "https://statsapi.mlb.com/api/v1/stats"
        # 进攻
        resp = requests.get(stats_url, params={"stats":"season","season":year,"group":"hitting","gameType":"R","limit":30}, timeout=15)
        runs_scored_map = {}
        so_map = {}
        bb_map = {}
        pa_map = {}
        if resp.status_code == 200:
            for split in resp.json().get("stats",[[]])[0].get("splits",[]):
                team_name = split["team"]["name"]
                stat = split["stat"]
                runs_scored_map[team_name] = stat.get("runs",0)
                so_map[team_name] = stat.get("strikeOuts",0)
                bb_map[team_name] = stat.get("baseOnBalls",0)
                pa_map[team_name] = stat.get("plateAppearances",0)

        # 投球
        resp = requests.get(stats_url, params={"stats":"season","season":year,"group":"pitching","gameType":"R","limit":30}, timeout=15)
        runs_allowed_map = {}
        k_allowed_map = {}
        bb_allowed_map = {}
        bf_map = {}
        if resp.status_code == 200:
            for split in resp.json().get("stats",[[]])[0].get("splits",[]):
                team_name = split["team"]["name"]
                stat = split["stat"]
                runs_allowed_map[team_name] = stat.get("runs",0)
                k_allowed_map[team_name] = stat.get("strikeOuts",0)
                bb_allowed_map[team_name] = stat.get("baseOnBalls",0)
                bf_map[team_name] = stat.get("battersFaced",0)

        if not df_teams.empty:
            df_teams["runs_scored"] = df_teams["name"].map(runs_scored_map).fillna(400)
            df_teams["runs_allowed"] = df_teams["name"].map(runs_allowed_map).fillna(400)
            df_teams["home_k_pct"] = df_teams["name"].apply(lambda x: so_map.get(x,0)/pa_map.get(x,1) if pa_map.get(x,1)>0 else 0.2)
            df_teams["home_bb_pct"] = df_teams["name"].apply(lambda x: bb_map.get(x,0)/pa_map.get(x,1) if pa_map.get(x,1)>0 else 0.08)
            df_teams["away_k_pct"] = df_teams["name"].apply(lambda x: k_allowed_map.get(x,0)/bf_map.get(x,1) if bf_map.get(x,1)>0 else 0.2)
            df_teams["away_bb_pct"] = df_teams["name"].apply(lambda x: bb_allowed_map.get(x,0)/bf_map.get(x,1) if bf_map.get(x,1)>0 else 0.08)
    except Exception as e:
        if errors is not None:
            errors.append(f"Sportsipy runs error: {e}")
        if not df_teams.empty:
            df_teams["runs_scored"] = 400
            df_teams["runs_allowed"] = 400
            df_teams["home_k_pct"] = 0.2
            df_teams["home_bb_pct"] = 0.08
            df_teams["away_k_pct"] = 0.2
            df_teams["away_bb_pct"] = 0.08

    player_info = {"name":"Shohei Ohtani","home_runs":None,"avg":None,"ops":None,"note":"Player stats disabled"}
    return {"teams": df_teams, "player_example": player_info}
