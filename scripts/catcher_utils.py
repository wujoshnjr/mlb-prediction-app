"""
捕手效应工具函数
获取捕手赛季防守数据，计算 catcher ERA 或 framing 代理
"""
import requests

CATCHER_CACHE = {}

def get_catcher_stats(catcher_id, season=2026):
    """获取捕手赛季防守数据，返回 catcher_era 等"""
    if catcher_id in CATCHER_CACHE:
        return CATCHER_CACHE[catcher_id]

    url = f"https://statsapi.mlb.com/api/v1/people/{catcher_id}/stats"
    params = {"stats": "season", "season": season, "gameType": "R", "group": "fielding"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        if stats:
            splits = stats[0].get("splits", [{}])
            stat = splits[0].get("stat", {}) if splits else {}
            data = {
                "catcher_era": stat.get("catcherERA"),
                "stolen_base_pct": stat.get("stolenBasePercentage"),
                "assists": stat.get("assists"),
                "errors": stat.get("errors")
            }
            CATCHER_CACHE[catcher_id] = data
            return data
    except:
        pass
    return {}

def calculate_catcher_effect(home_catcher_id, away_catcher_id, season=2026):
    """计算主客队捕手效应差值，返回 catcher_era_diff 和 framing_proxy_diff"""
    home_stats = get_catcher_stats(home_catcher_id, season) if home_catcher_id else {}
    away_stats = get_catcher_stats(away_catcher_id, season) if away_catcher_id else {}

    home_era = float(home_stats.get("catcher_era", 4.0)) if home_stats.get("catcher_era") else 4.0
    away_era = float(away_stats.get("catcher_era", 4.0)) if away_stats.get("catcher_era") else 4.0
    catcher_era_diff = home_era - away_era

    # 阻杀率代理
    home_cs = float(home_stats.get("stolen_base_pct", 0.3)) if home_stats.get("stolen_base_pct") else 0.3
    away_cs = float(away_stats.get("stolen_base_pct", 0.3)) if away_stats.get("stolen_base_pct") else 0.3
    cs_diff = home_cs - away_cs

    return catcher_era_diff, cs_diff
