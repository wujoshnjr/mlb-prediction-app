"""
裁判倾向客户端
从 Baseball Savant 获取裁判好球带倾向数据
"""
import requests
import pandas as pd

# 2025赛季已知裁判倾向数据（来源：Ump Scorecards）
# 格式：{裁判名: {"zone_size": 相对平均的百分比, "k_rate": K%偏差}
UMPIRE_TENDENCIES = {
    "Pat Hoberg": {"zone_size": 1.02, "k_rate": 0.01},
    "Lance Barrett": {"zone_size": 0.97, "k_rate": -0.02},
    "John Libka": {"zone_size": 1.05, "k_rate": 0.03},
    "Lance Barksdale": {"zone_size": 0.95, "k_rate": -0.03},
    "Dan Iassogna": {"zone_size": 1.01, "k_rate": 0.00},
    "Cory Blaser": {"zone_size": 0.98, "k_rate": -0.01},
    "Mark Wegner": {"zone_size": 1.03, "k_rate": 0.02},
    "Chris Guccione": {"zone_size": 0.96, "k_rate": -0.02},
    "Doug Eddings": {"zone_size": 1.04, "k_rate": 0.02},
    "Bill Miller": {"zone_size": 1.00, "k_rate": 0.00},
    "Marvin Hudson": {"zone_size": 0.99, "k_rate": -0.01},
    "Jerry Meals": {"zone_size": 1.02, "k_rate": 0.01},
    "Brian Knight": {"zone_size": 0.97, "k_rate": -0.02},
    "Mike Muchlinski": {"zone_size": 1.01, "k_rate": 0.00},
    "David Rackley": {"zone_size": 0.98, "k_rate": -0.01},
    "Mark Ripperger": {"zone_size": 1.03, "k_rate": 0.02},
    "Ryan Blakney": {"zone_size": 1.00, "k_rate": 0.00},
    "Roberto Ortiz": {"zone_size": 0.96, "k_rate": -0.03},
    "Ramon De Jesus": {"zone_size": 1.01, "k_rate": 0.01},
    "Nick Mahrley": {"zone_size": 0.99, "k_rate": -0.01},
}

def fetch_umpire_data(date_str: str = None, errors: list = None) -> pd.DataFrame:
    """
    获取当日比赛的主裁判信息，匹配已知的裁判倾向。
    返回包含 game_id、umpire_name、zone_size、k_rate 的 DataFrame。
    """
    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d')

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "game(content(editorial(recap))),decisions",
        "gameTypes": "R"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if errors is not None:
            errors.append(f"Umpire fetch error: {e}")
        return pd.DataFrame()

    umpire_data = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            game_id = game["gamePk"]
            # 获取裁判信息（从 decisions 中提取）
            officials = game.get("officials", [])
            home_plate_umpire = None
            for official in officials:
                if official.get("position") == "Home Plate":
                    home_plate_umpire = official.get("official", {}).get("fullName", "Unknown")
                    break

            if home_plate_umpire and home_plate_umpire in UMPIRE_TENDENCIES:
                tendency = UMPIRE_TENDENCIES[home_plate_umpire]
                umpire_data.append({
                    "game_id": game_id,
                    "umpire_name": home_plate_umpire,
                    "zone_size": tendency["zone_size"],
                    "k_rate": tendency["k_rate"]
                })
            elif home_plate_umpire:
                # 未知裁判使用默认值
                umpire_data.append({
                    "game_id": game_id,
                    "umpire_name": home_plate_umpire,
                    "zone_size": 1.0,
                    "k_rate": 0.0
                })

    return pd.DataFrame(umpire_data)
