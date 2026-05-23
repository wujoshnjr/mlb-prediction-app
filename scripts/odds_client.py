"""
Odds-API.io 客户端 (v3 - 直接 requests 调用 + 盘口快照保存 + 曲线特征提取)
"""
import requests
import pandas as pd
import os
import numpy as np
from datetime import datetime

# ---------- 原有 fetch_odds 函数（完全保留）----------
def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        if errors is not None:
            errors.append("Odds API key missing")
        return pd.DataFrame()

    try:
        # 获取 MLB 赛事列表
        events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
        headers = {"apikey": api_key}
        events_resp = requests.get(events_url, headers=headers, timeout=15)
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

        # 获取赔率
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

        # 解析
        rows = []
        games = odds_data if isinstance(odds_data, list) else odds_data.get("data", [])
        for game in games:
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            rows.append({
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": bookmaker.get("title"),
                                "bet_type": "h2h",
                                "team": outcome.get("name"),
                                "odds": outcome.get("price")
                            })

        odds_df = pd.DataFrame(rows)

        # 保存盘口快照（每个比赛单独存一份，方便后续曲线提取）
        if not odds_df.empty:
            snapshot_dir = "data/odds_snapshots"
            os.makedirs(snapshot_dir, exist_ok=True)

            # 为每场比赛保存一个 csv（按比赛 ID 命名）
            for game_id, group in odds_df.groupby(['home_team', 'away_team']):
                home, away = game_id
                # 生成一个唯一比赛 ID（可自定义规则）
                safe_home = home.replace(" ", "_").lower()
                safe_away = away.replace(" ", "_").lower()
                game_key = f"{safe_home}_{safe_away}"
                filepath = os.path.join(snapshot_dir, f"{game_key}.csv")

                # 读取已有快照（若存在）
                if os.path.exists(filepath):
                    old_df = pd.read_csv(filepath)
                    combined = pd.concat([old_df, group], ignore_index=True).tail(24)  # 只保留最近 24 条
                else:
                    combined = group

                combined.to_csv(filepath, index=False)

            # 清理旧的整体快照文件（原来的逻辑可以保留，但改为只清理旧的整体文件）
            # 实际上原来的按时间戳保存的整体文件可以去掉，或者保留但不再用于曲线。
            # 为了兼容，仍然保留一份带时间戳的整体快照。
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            odds_df.to_csv(os.path.join(snapshot_dir, f"odds_{timestamp}.csv"), index=False)
            # 只保留最近24个整体快照
            snapshots = sorted([f for f in os.listdir(snapshot_dir) if f.startswith("odds_") and f.endswith(".csv")])
            if len(snapshots) > 24:
                for old in snapshots[:-24]:
                    os.remove(os.path.join(snapshot_dir, old))

        return odds_df

    except Exception as e:
        if errors is not None:
            errors.append(f"Odds API fetch error: {e}")
        return pd.DataFrame()

# ---------- 新增：盘口快照保存（供其他模块调用，但上述 fetch_odds 内部已处理）----------
def save_odds_snapshot(game_id: str, home_odds: float, away_odds: float, over_odds: float = None, under_odds: float = None):
    """
    追加一条赔率快照到指定比赛文件。
    参数 game_id 建议格式：'home_away' 或使用 fetch_odds 中的 game_key。
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
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).tail(12)  # 保留最近12条
    else:
        df = pd.DataFrame([row])
    df.to_csv(filepath, index=False)

# ---------- 新增：盘口曲线特征提取 ----------
def extract_odds_curve_features(game_id: str):
    """
    从历史快照中提取赔率曲线的趋势、波动率和逆转次数。
    game_id: 比赛标识符，与 save_odds_snapshot 中一致。
    返回 (trend, volatility, reversals)
    """
    snapshot_dir = "data/odds_snapshots"
    filepath = os.path.join(snapshot_dir, f"{game_id}.csv")
    if not os.path.exists(filepath):
        return 0.0, 0.0, 0

    df = pd.read_csv(filepath)
    # 至少需要2条记录
    if len(df) < 2:
        return 0.0, 0.0, 0

    # 使用 home_odds 列
    if 'home_odds' not in df.columns:
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

    # 逆转次数：相邻差符号变化
    signs = np.sign(np.diff(home_odds))
    reversals = np.sum(signs[:-1] != signs[1:]) if len(signs) > 1 else 0

    return slope, volatility, reversals
