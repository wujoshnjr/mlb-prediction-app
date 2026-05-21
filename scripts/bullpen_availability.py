"""
牛棚可用性评估器
基于学术研究「Out of gas: quantifying fatigue in MLB relievers」的发现：
- 救援投手投球超过15球后，下一场球速会轻微下降
- 超过20球后，下降幅度进一步放大
- 疲劳效应每天约减半
"""
import pandas as pd

def calculate_bullpen_availability(bullpen_usage_df):
    """
    计算每支球队的牛棚可用性评分。
    返回 {team_id: availability_score} 字典。
    availability_score 范围 0-100，越高表示牛棚越充沛。
    """
    if bullpen_usage_df is None or bullpen_usage_df.empty:
        return {}

    availability_dict = {}
    for _, row in bullpen_usage_df.iterrows():
        team_id = row.get("team_id")
        pitches = float(row.get("bullpen_pitches", 0) or 0)
        innings = float(row.get("bullpen_innings", 0) or 0)
        back_to_back = int(row.get("back_to_back", 0) or 0)

        # 疲劳度计算：每15球扣10分，每局扣5分，背靠背额外扣15分
        fatigue = (pitches / 15) * 10 + innings * 5 + back_to_back * 15

        # 可用性 = 100 - 疲劳度，最低0
        availability = max(0, 100 - fatigue)
        availability_dict[team_id] = round(availability, 1)

    return availability_dict
