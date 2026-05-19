"""
旅行与休息特征
根据赛程计算主客队休息天数、是否跨时区等
"""
import pandas as pd
from datetime import datetime, timedelta

# 简化版：30队所在时区（东部/中部/山地/太平洋）
TEAM_TIMEZONES = {
    "Braves": "Eastern", "Orioles": "Eastern", "Red Sox": "Eastern",
    "Cubs": "Central", "White Sox": "Central", "Reds": "Eastern",
    "Guardians": "Eastern", "Rockies": "Mountain", "Tigers": "Eastern",
    "Astros": "Central", "Royals": "Central", "Angels": "Pacific",
    "Dodgers": "Pacific", "Marlins": "Eastern", "Brewers": "Central",
    "Twins": "Central", "Mets": "Eastern", "Yankees": "Eastern",
    "Athletics": "Pacific", "Phillies": "Eastern", "Pirates": "Eastern",
    "Padres": "Pacific", "Giants": "Pacific", "Mariners": "Pacific",
    "Cardinals": "Central", "Rays": "Eastern", "Rangers": "Central",
    "Blue Jays": "Eastern", "Nationals": "Eastern", "D-backs": "Mountain"
}

def calculate_rest_days(schedule_df):
    """
    根据赛程DataFrame计算每场比赛两队的休息天数
    简化版：假设所有球队前一场比赛在1-3天前
    """
    rest_home = []
    rest_away = []
    for _, game in schedule_df.iterrows():
        # 简化：随机分配1-3天休息（实际应用需查询前一场比赛日期）
        rest_home.append(2)   # 占位，后续可用真实数据替代
        rest_away.append(2)
    schedule_df['rest_home'] = rest_home
    schedule_df['rest_away'] = rest_away
    return schedule_df
