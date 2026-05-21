"""
球员评分系统（基于 Statcast 数据）
对每个投手计算综合评分，返回各队平均评分
"""
import pandas as pd
import numpy as np

def calculate_pitcher_ratings(savant_df):
    """
    基于 Savant 数据计算每个投手的综合评分。
    返回按球队平均的评分。
    """
    if savant_df.empty:
        return {}

    # 需要的关键字段
    required_cols = ['pitch_hand', 'release_speed', 'pfx_x', 'pfx_z', 'barrel', 'hard_hit', 'release_spin_rate']
    if not all(c in savant_df.columns for c in required_cols):
        return {}

    df = savant_df.dropna(subset=required_cols).copy()
    if df.empty:
        return {}

    # 维度计算
    # Velocity: 球速 (以 93 mph 为基准 100)
    df['velocity_score'] = (df['release_speed'] - 93) * 2 + 100
    # Movement: 水平+垂直位移的欧氏距离 (以 10 inches 为基准)
    movement = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_score'] = (movement - 10) * 5 + 100
    # Swing & Miss: 简化为转速代理 (以 2300 rpm 为基准)
    df['swing_miss_score'] = (df['release_spin_rate'] - 2300) * 0.05 + 100
    # Damage: 被扎实击球率 (越低越好)
    df['damage_score'] = 100 - df['barrel'].astype(float) * 50 - df['hard_hit'].astype(float) * 25

    # 综合评分 (4个维度平均)
    df['overall_rating'] = (df['velocity_score'] + df['movement_score'] + df['swing_miss_score'] + df['damage_score']) / 4
    df['overall_rating'] = df['overall_rating'].clip(0, 200)

    # 按球队（假设投手所属球队为主队）平均
    team_ratings = df.groupby('home_team')['overall_rating'].mean().reset_index()
    team_ratings.rename(columns={'home_team': 'team_name', 'overall_rating': 'pitcher_rating'}, inplace=True)
    rating_dict = dict(zip(team_ratings['team_name'], team_ratings['pitcher_rating']))
    return rating_dict
