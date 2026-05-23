# prediction.py
"""
主预测引擎：生成每场比赛的预测，并写入结果。
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import config
from model import build_game_features
from scripts.database import save_predictions, load_predictions
from scripts.rating_updater import update_ratings, process_event
from scripts.monte_carlo import monte_carlo_simulation
from scripts.expected_value import check_positive_ev

# NRFI 模型
if config.NRFI_USE_ML:
    from scripts.nrf_model import NRFIModel, extract_nrf_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'models/stacking_model.pkl'
NRFI_MODEL_PATH = 'models/nrf_model.pkl'

# 加载主模型
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    logger.warning("主模型文件不存在，将使用手工集成")

# 加载 NRFI 模型
nrfi_model = None
if config.NRFI_USE_ML and os.path.exists(NRFI_MODEL_PATH):
    nrfi_model = NRFIModel(NRFI_MODEL_PATH)
    nrfi_model.load()

def generate_predictions(games):
    """
    为比赛列表生成预测。
    games: list of dict，来自 MLB Stats API 或其他数据源。
    """
    predictions = []
    for game in games:
        # 构建特征
        feats = build_game_features(game)
        X = pd.DataFrame([feats])

        # 主模型预测
        if model is not None:
            try:
                prob_home = model.predict_proba(X)[0, 1]
            except Exception as e:
                logger.error(f"模型预测失败: {e}")
                prob_home = manual_prediction(feats)
        else:
            prob_home = manual_prediction(feats)

        # 蒙特卡洛模拟（可选）
        mc_result = monte_carlo_simulation(game, prob_home)

        # NRFI 预测
        nrf_prob = None
        if config.NRFI_USE_ML and nrfi_model is not None:
            nrf_feats = extract_nrf_features(game)
            X_nrf = pd.DataFrame([nrf_feats])
            try:
                nrf_prob = nrfi_model.predict_proba(X_nrf)[0]
            except Exception:
                nrf_prob = manual_nrfi(game)
        else:
            nrf_prob = manual_nrfi(game)

        # 整合结果
        pred = {
            'game_id': game['game_id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'prob_home_win': prob_home,
            'run_line_over_prob': mc_result.get('rl_over_prob'),
            'total_over_prob': mc_result.get('over_prob'),
            'nrf_prob': nrf_prob,
            'recommendation': check_positive_ev(prob_home, game),
            'timestamp': datetime.now().isoformat()
        }
        predictions.append(pred)

    # 保存预测
    save_predictions(predictions)
    return predictions

def manual_prediction(feats):
    """手工集成预测（当模型不可用时）"""
    win_pct = feats.get('market_prob', 0.5)
    elo_prob = 1 / (1 + 10 ** (-feats.get('elo_diff', 0) / 400))
    market_prob = feats.get('market_prob', 0.5)
    return 0.25 * win_pct + 0.35 * elo_prob + 0.40 * market_prob

def manual_nrfi(game):
    """手工 NRFI 计算"""
    # 简单示例
    return 0.60

def update_results_and_ratings():
    """获取比赛结果并更新评级系统"""
    # 实现获取今日比赛结果
    results = []  # 从 update_results.py 或直接查询 API
    if results:
        update_ratings(results)
    # 如果有伤病事件，可调用 process_event
    # process_event('NYY', 'core_injury')

if __name__ == '__main__':
    # 示例：从赛程获取比赛列表并生成预测
    from scripts.mlb_stats_client import get_today_schedule
    today_games = get_today_schedule()
    if today_games:
        preds = generate_predictions(today_games)
        with open('report/prediction.json', 'w') as f:
            json.dump(preds, f, indent=2)
        logger.info(f"已生成 {len(preds)} 条预测")
