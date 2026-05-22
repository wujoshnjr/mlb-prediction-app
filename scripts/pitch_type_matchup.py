import requests
import numpy as np

def get_pitcher_pitch_types(pitcher_id):
    """获取投手常用球种及其使用比例（简化版：可用 Savant 数据聚合）"""
    # 此处应从 Savant 或 Stats API 获取数据，为演示返回随机值
    return {"FF": 0.5, "SL": 0.3, "CH": 0.2}  # 示例

def get_batter_vs_pitch_type(batter_ids, pitch_type):
    """返回打者群对特定球种的平均 xwOBA"""
    # 实际应查询 Savant 或使用预计算表
    return 0.320  # 示例

def get_pitch_type_matchup_score(home_pitcher_id, away_pitcher_id, home_lineup=None, away_lineup=None):
    """计算主客队投打对位分数差值"""
    if not home_pitcher_id or not away_pitcher_id:
        return 0.0
    # 简化：随机返回一个小值
    return np.random.normal(0, 0.02)
