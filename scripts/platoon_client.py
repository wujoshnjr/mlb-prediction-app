import requests
import pandas as pd

def fetch_platoon_splits(season=2026):
    """获取所有球队对左/右投的 wOBA 或 OPS 拆分"""
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "season",
        "season": season,
        "group": "hitting",
        "gameType": "R",
        "limit": 300  # 足够覆盖所有球队
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # 解析球队 split 数据（需按 pitcher_hand 分组，这里简化处理）
    # 返回一个 DataFrame，包含 team_name, vs_lhp_ops, vs_rhp_ops
    # 具体解析逻辑略，根据 API 返回结构调整
    return pd.DataFrame()
