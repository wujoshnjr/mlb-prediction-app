"""
Odds-API.io 客户端 (v3 - 直接 requests 调用 + 盘口快照保存 + 曲线特征提取)
修复：增加请求重试、详细日志、兜底处理，确保不会无声失效。
"""
import requests
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import time

def save_odds_snapshot(game_id: str, home_odds: float, away_odds: float,
                       over_odds: float = None, under_odds: float = None):
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
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).tail(12)
    else:
        df = pd.DataFrame([row])
    df.to_csv(filepath, index=False)

def extract_odds_curve_features(game_id: str):
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
    x = np.arange(len(home_odds))
    slope = np.polyfit(x, home_odds, 1)[0] / mean_odds
    volatility = np.std(home_odds) / mean_odds
    signs = np.sign(np.diff(home_odds))
    reversals = np.sum(signs[:-1] != signs[1:]) if len(signs) > 1 else 0
    return slope, volatility, reversals

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        if errors is not None:
            errors.append("Odds API key missing")
        return pd.DataFrame()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
            events_params = {"apiKey": api_key}
            events_resp = requests.get(events_url, params=events_params, timeout=15)
            if events_resp.status_code == 401:
                if errors is not None:
                    errors.append("Odds API 401: Invalid API key")
                return pd.DataFrame()
            if events_resp.status_code == 422:
                if errors is not None:
                    errors.append("Odds API 422: 请求参数错误或当日无比赛")
                return pd.DataFrame()
            if events_resp.status_code == 429:
                wait = 60
                print(f"⚠️ Odds API 频率限制，等待 {wait}s 后重试 (attempt {attempt+1})")
                time.sleep(wait)
                continue
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
            if odds_resp.status_code == 429:
                wait = 60
                print(f"⚠️ Odds API 赔率端点频率限制，等待 {wait}s 后重试")
                time.sleep(wait)
                continue
            odds_resp.raise_for_status()
            odds_data = odds_resp.json()

            rows = []
            game_odds = {}
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
                                if name == home_team:
                                    game_odds[game_key]['home'].append(price)
                                elif name == away_team:
                                    game_odds[game_key]['away'].append(price)

            odds_df = pd.DataFrame(rows)
            if not odds_df.empty:
                for (home, away), odds_dict in game_odds.items():
                    if odds_dict['home'] and odds_dict['away']:
                        avg_home = np.mean(odds_dict['home'])
                        avg_away = np.mean(odds_dict['away'])
                        safe_home = home.replace(" ", "_").lower()
                        safe_away = away.replace(" ", "_").lower()
                        game_id = f"{safe_home}_{safe_away}"
                        save_odds_snapshot(game_id, avg_home, avg_away)

            print(f"✅ 成功获取 {len(odds_df)} 条赔率数据")
            return odds_df

        except Exception as e:
            if attempt == max_retries - 1:
                if errors is not None:
                    errors.append(f"Odds API 最终失败: {e}")
                return pd.DataFrame()
            time.sleep(10)
