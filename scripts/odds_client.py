# scripts/odds_client.py
"""
赔率客户端：从 The Odds API 获取即时赔率，支持快照保存和曲线特征提取。
"""

import os
import csv
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import requests

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 配置（请替换为实际 Key 或保持免费层）
ODDS_API_KEY = 'your-api-key'
ODDS_BASE_URL = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/'

# 快照存储目录
SNAPSHOT_DIR = 'data/odds_snapshots'

def get_odds(game):
    """获取单场比赛赔率（原有逻辑简化）"""
    # 实际实现可能调用 API 并解析
    # 返回一个 dict，包含 home_implied_prob, change_from_prev, momentum 等
    return {
        'home_implied_prob': 0.55,
        'change_from_prev': 0.02,
        'momentum': 0.01
    }

def save_odds_snapshot(game_id, home_odds, away_odds, over_odds=None, under_odds=None):
    """
    将当前赔率快照追加到比赛 CSV 文件中，保留最近 12 条记录。
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    filepath = os.path.join(SNAPSHOT_DIR, f'{game_id}.csv')
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
    logger.debug(f'快照已保存: {game_id}')

def extract_odds_curve_features(game_id):
    """
    从历史快照中提取赔率曲线的趋势、波动率和逆转次数。
    返回 (trend, volatility, reversals)
    """
    filepath = os.path.join(SNAPSHOT_DIR, f'{game_id}.csv')
    if not os.path.exists(filepath):
        return 0.0, 0.0, 0

    df = pd.read_csv(filepath)
    if len(df) < 2:
        return 0.0, 0.0, 0

    home_odds = df['home_odds'].values.astype(float)
    mean_odds = np.mean(home_odds)
    if mean_odds == 0:
        return 0.0, 0.0, 0

    # 趋势：线性回归斜率标准化
    x = np.arange(len(home_odds))
    slope = np.polyfit(x, home_odds, 1)[0] / mean_odds

    # 波动率：变异系数
    volatility = np.std(home_odds) / mean_odds

    # 逆转次数：符号变化
    signs = np.sign(np.diff(home_odds))
    reversals = np.sum(signs[:-1] != signs[1:]) if len(signs) > 1 else 0

    return slope, volatility, reversals
