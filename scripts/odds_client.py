"""
Odds-API.io 客户端 (v3 - 直接 requests 调用 + 盘口快照保存 + 曲线特征提取)
"""
import requests
import pandas as pd
import os
import numpy as np
from datetime import datetime

# ---------- 盘口快照保存 ----------
def save_odds_snapshot(game_id: str, home_odds: float, away_odds: float,
                       over_odds: float = None, under_odds: float = None):
    """
    追加一条赔率快照到指定比赛文件。
    game_id: 建议格式 'home_away'（小写、下划线分隔）。
    """
    snapshot_dir = "data/odds_snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    filepath = os.path.join(snapshot_dir, f"{game_id}.csv")

    now = datetime.now().isoformat()
    row = {
        'timestamp': now,
        'home_odds': home_odds,
        'away_odds': away_odds,
        'over_odds': over_odds,
        'under_odds': under_odds
    }
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # 保留最近12条记录，防止文件无限增长
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).tail(12)
    else:
        df = pd.DataFrame([row])
    df.to_csv(filepath, index=False)

# ---------- 盘口曲线特征提取 ----------
def extract_odds_curve_features(game_id: str):
    """
    从历史快照中提取赔率曲线的趋势、波动率和逆转次数。
    返回 (trend, volatility, reversals)
    """
    snapshot_dir = "data/odds_snapshots"
    filepath = os.path.join(snapshot_dir, f"{game_id}.csv")
    if not os.path.exists(filepath):
        return 0.0, 0.0, 0

    df = pd.read_csv(filepath)
    if len(df) < 2 or 'home_odds' not in df.columns:
        return 0.0, 0.0, 0

    home_odds = df['home_odds'].dropna().values.astype(float)
    if len(home_odds) < 2:
        return 0.0, 0.0, 0

    mean_odds = np.mean(home_odds)
    if mean_odds == 0:
        return 0.0, 0.0, 0

    # 趋势：线性回归斜率 / 均值
    x = np.arange(len(home_odds))
    slope = np.polyfit(x, home_odds, 1)[0] / mean_odds

    # 波动率：变异系数
    volatility = np.std(home_odds) / mean_odds

    # 逆转次数：相邻差符号变化次数
    signs = np.sign(np.diff(home_odds))
    reversals = np.sum(signs[:-1] != signs[1:]) if len(signs) > 1 else 0

    return slope, volatility, reversals

# ---------- 主要赔率获取函数（修复认证方式 + 内部保存快照）----------
def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        if errors is not None:
            errors.append("Odds API key missing")
        return pd.DataFrame()

    try:
        # 获取 MLB 赛事列表（统一使用 apiKey 查询参数）
        events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
        events_params = {"apiKey": api_key}
        events_resp = requests.get(events_url, params=events_params, timeout=15)
        if events_resp.status_code == 401:
            if errors is not None:
                errors.append("Odds API 401: Invalid API key")
            return pd.DataFrame()
        events_resp.raise_for_status()
        events_data = events_resp.json()
        events = events_data if isinstance(events_data, list) else events_data.get("data", [])

        if not events:
            if errors is not None:
                errors.append("No MLB events found in Odds API")
            return pd.DataFrame()

        event_ids = [e["id"] for e in events if "id" in e]
        if not event_ids:
            if errors is not None:
                errors.append("No event IDs found")
            return pd.DataFrame()

        # 获取赔率（统一使用 apiKey 查询参数）
        odds_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        odds_params = {
            "apiKey": api_key,
            "eventIds": ",".join(event_ids),
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": "bet365,draftkings"
        }
        odds_resp = requests.get(odds_url, params=odds_params, timeout=30)
        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        # 解析原始数据
        rows = []
        # 用于统计每场比赛的主客赔率（取所有 bookmaker 的平均）
        game_odds = {}  # key: (home_team, away_team) -> {'home': [], 'away': []}

        games = odds_data if isinstance(odds_data, list) else odds_data.get("data", [])
        for game in games:
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            game_key = (home_team, away_team)
            if game_key not in game_odds:
                game_odds[game_key] = {'home': [], 'away': []}

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name")
                            price = outcome.get("price")
                            rows.append({
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": bookmaker.get("title"),
                                "bet_type": "h2h",
                                "team": name,
                                "odds": price
                            })
                            # 收集主客队赔率用于快照
                            if name == home_team:
                                game_odds[game_key]['home'].append(price)
                            elif name == away_team:
                                game_odds[game_key]['away'].append(price)

        odds_df = pd.DataFrame(rows)

        # 内部保存每场比赛的平均赔率快照（用于曲线特征）
        if not odds_df.empty:
            for (home, away), odds_dict in game_odds.items():
                if odds_dict['home'] and odds_dict['away']:
                    avg_home = np.mean(odds_dict['home'])
                    avg_away = np.mean(odds_dict['away'])
                    # 生成安全的 game_id
                    safe_home = home.replace(" ", "_").lower()
                    safe_away = away.replace(" ", "_").lower()
                    game_id = f"{safe_home}_{safe_away}"
                    # 保存快照（大小分赔率暂不获取，可留空）
                    save_odds_snapshot(game_id, avg_home, avg_away)

        return odds_df

    except Exception as e:
        if errors is not None:
            errors.append(f"Odds API fetch error: {e}")
        return pd.DataFrame()
