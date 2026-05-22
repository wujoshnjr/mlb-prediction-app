import numpy as np

def get_pitch_type_matchup_score(home_pitcher_id, away_pitcher_id, home_lineup=None, away_lineup=None):
    """
    计算主客队投打对位分数差值（简化版，后续可扩展为从Savant获取真实数据）
    """
    if not home_pitcher_id or not away_pitcher_id:
        return 0.0
    # 模拟计算：返回一个均值为0，标准差0.02的随机值
    # 实际实现需查询投手球种分布与打者对该球种的历史xwOBA
    return np.random.normal(0, 0.02)
