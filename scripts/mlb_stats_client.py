"""
MLB Stats API 客户端（终极诊断版）
硬编码测试日期，超强容错，打印完整返回内容
"""
import requests
import pandas as pd
from datetime import datetime

def fetch_mlb_statsapi(date_str: str = None, errors: list = None) -> pd.DataFrame:
    # ---- 强制使用已知有比赛的日期进行测试 ----
    if date_str is None:
        # 暂时硬编码为昨天，确保有比赛数据
        date_str = "2026-05-17"
        print(f"[DIAG] 未提供日期，默认使用测试日期: {date_str}")
    
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "team,probablePitcher,venue",
        "gameTypes": "R"
    }
    print(f"[DIAG] 请求 URL: {url}")
    print(f"[DIAG] 请求参数: {params}")
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"[DIAG] HTTP 状态码: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] MLB Stats API 请求失败: {e}")
        if errors is not None:
            errors.append(f"MLB Stats API error: {e}")
        return pd.DataFrame()

    # 打印 API 返回的关键顶层字段
    print(f"[DIAG] API 返回顶层键: {list(data.keys())}")
    print(f"[DIAG] totalItems: {data.get('totalItems')}")
    print(f"[DIAG] dates 数组长度: {len(data.get('dates', []))}")

    games = []
    for date_info in data.get("dates", []):
        print(f"[DIAG] 处理日期: {date_info.get('date')}, 该日期比赛数: {date_info.get('totalGames')}")
        for game in date_info.get("games", []):
            game_id = game.get("gamePk")
            game_date = game.get("gameDate")
            status = game.get("status", {}).get("abstractGameState", "Unknown")
            
            # 安全获取球队名称
            try:
                home_name = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "Unknown")
            except:
                home_name = "Unknown"
            try:
                away_name = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "Unknown")
            except:
                away_name = "Unknown"
                
            venue = game.get("venue", {}).get("name", "Unknown")
            
            games.append({
                "game_id": game_id,
                "game_date": game_date,
                "status": status,
                "home_team": home_name,
                "away_team": away_name,
                "venue": venue
            })
            print(f"[DIAG] 成功解析比赛: {home_name} vs {away_name} (Status: {status})")

    df = pd.DataFrame(games)
    print(f"[DIAG] 最终收集到 {len(df)} 场比赛")
    if not df.empty:
        print(f"[DIAG] 数据框列名: {list(df.columns)}")
        print(f"[DIAG] 第一场比赛:\n{df.iloc[0].to_dict()}")
    else:
        print("[DIAG] 警告：未收集到任何比赛！请检查 API 返回的原始数据。")
        # 打印原始数据的前500字符以便排查
        print("[DIAG] API 原始返回内容:", str(data)[:500])
    
    return df
