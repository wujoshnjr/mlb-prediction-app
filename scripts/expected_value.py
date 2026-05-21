"""
正期望值投注评估器
"""
import numpy as np

def calculate_ev(predicted_prob, odds, commission=0.05):
    """
    计算单次投注的期望值
    predicted_prob: 模型预测的胜率
    odds: 博彩公司赔率 (decimal)
    commission: 博彩公司抽水比例 (默认 5%)
    返回: (ev, edge)
    """
    if odds is None or odds <= 1:
        return 0, 0

    # 市场隐含概率 (扣除抽水)
    market_prob = 1 / (odds * (1 + commission))

    # 期望值
    ev = predicted_prob * (odds - 1) - (1 - predicted_prob)

    # 与市场的差异 (Edge)
    edge = predicted_prob - market_prob

    return ev, edge

def filter_value_bets(predictions, min_ev=0.05, min_edge=0.03):
    """
    筛选正期望投注
    predictions: 列表，每个元素需包含 predicted_home_win_pct, predicted_away_win_pct, home_odds
    返回: 价值投注列表
    """
    value_bets = []
    for p in predictions:
        home_ev, home_edge = calculate_ev(p['predicted_home_win_pct'], p.get('home_odds'))
        # 估算客队赔率（假设市场充分，可以从 home_odds 反推，简化处理）
        away_odds = 1 / (1 - (1 / p['home_odds'])) if p.get('home_odds') and p['home_odds'] > 1 else None
        away_ev, away_edge = calculate_ev(p['predicted_away_win_pct'], away_odds)

        if home_ev > min_ev and home_edge > min_edge:
            value_bets.append({
                'game': f"{p['home_team']} vs {p['away_team']}",
                'type': 'Home',
                'ev': round(home_ev, 4),
                'edge': round(home_edge, 4)
            })
        if away_ev > min_ev and away_edge > min_edge:
            value_bets.append({
                'game': f"{p['home_team']} vs {p['away_team']}",
                'type': 'Away',
                'ev': round(away_ev, 4),
                'edge': round(away_edge, 4)
            })
    return value_bets
