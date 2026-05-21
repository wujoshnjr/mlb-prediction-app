import requests
CATCHER_CACHE = {}
def get_catcher_stats(catcher_id, season=2026):
    if catcher_id in CATCHER_CACHE:
        return CATCHER_CACHE[catcher_id]
    url = f"https://statsapi.mlb.com/api/v1/people/{catcher_id}/stats"
    params = {"stats":"season","season":season,"gameType":"R","group":"fielding"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        splits = resp.json().get("stats",[[]])[0].get("splits",[{}])
        stat = splits[0].get("stat",{}) if splits else {}
        data = {"catcher_era":stat.get("catcherERA"),"stolen_base_pct":stat.get("stolenBasePercentage")}
        CATCHER_CACHE[catcher_id] = data
        return data
    except:
        return {}
def calculate_catcher_effect(home_catcher_id, away_catcher_id, season=2026):
    home = get_catcher_stats(home_catcher_id, season) if home_catcher_id else {}
    away = get_catcher_stats(away_catcher_id, season) if away_catcher_id else {}
    home_era = float(home.get("catcher_era",4.0)) if home.get("catcher_era") else 4.0
    away_era = float(away.get("catcher_era",4.0)) if away.get("catcher_era") else 4.0
    home_cs = float(home.get("stolen_base_pct",0.3)) if home.get("stolen_base_pct") else 0.3
    away_cs = float(away.get("stolen_base_pct",0.3)) if away.get("stolen_base_pct") else 0.3
    return home_era - away_era, home_cs - away_cs
