"""
Odds-API.io 客户端 (v3 - 直接 requests 调用 + 盘口快照保存)
"""
import requests
import pandas as pd
import os
from datetime import datetime

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

        # 保存盘口快照（供后续计算盘口动量）
        if not odds_df.empty:
            snapshot_dir = "data/odds_snapshots"
            os.makedirs(snapshot_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            odds_df.to_csv(os.path.join(snapshot_dir, f"odds_{timestamp}.csv"), index=False)
            # 只保留最近24个快照（一天的量）
            snapshots = sorted(os.listdir(snapshot_dir))
            if len(snapshots) > 24:
                for old in snapshots[:-24]:
                    os.remove(os.path.join(snapshot_dir, old))

        return odds_df

    except Exception as e:
        if errors is not None:
            errors.append(f"Odds API fetch error: {e}")
        return pd.DataFrame()
