# scripts/backtest.py
"""
回测系统：支持原有赛季切分回测和严格 Walk-Forward 回测。
通过 config.WALKFORWARD_STRICT 切换。
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# 导入项目模块
import config
from scripts.walkforward import walkforward_train_evaluate

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== 原有的赛季切分回测函数（保留） =====================
def backtest_by_season(data, feature_cols, target_col='target', model_builder=None):
    """
    按赛季切分的传统回测。
    """
    # 假设数据有 'season' 列
    seasons = sorted(data['season'].unique())
    results = []
    for i, season in enumerate(seasons[:-1]):
        train = data[data['season'] == season]
        test = data[data['season'] == seasons[i+1]]
        if train.empty or test.empty:
            continue
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]
        model = model_builder()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results.append(pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': proba,
            'date': test['date'].values
        }))
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# ===================== 新增 Walk-Forward 回测 =====================
def backtest_walkforward(data, feature_cols, target_col='target', date_col='date',
                         model_builder=None, gap_days=7):
    """
    使用严格 Walk-Forward 回测。
    """
    X = data[feature_cols]
    y = data[target_col]
    dates = data[date_col]

    y_true, y_pred, dates_out = walkforward_train_evaluate(
        X, y, dates, model_builder,
        gap_days=gap_days,
        test_days=1,
        use_optuna=False
    )
    return pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'date': dates_out})


# ===================== 通用评估函数 =====================
def evaluate_predictions(df_pred):
    """计算回测指标"""
    if df_pred.empty:
        return {}
    y_true = df_pred['y_true'].values
    y_prob = df_pred['y_pred'].values
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    acc = np.mean((y_prob > 0.5) == y_true)
    # 简单 ROI 计算 (假设每次投注等额，胜率>0.5即盈利)
    # 更严谨的可结合赔率
    roi = 2 * acc - 1  # 简化版
    return {
        'brier': brier,
        'log_loss': ll,
        'auc': auc,
        'accuracy': acc,
        'roi': roi,
        'n_samples': len(y_true)
    }


def run_backtest(data, feature_cols, model_builder, date_col='date'):
    """主回测入口，根据配置选择回测方式"""
    if config.WALKFORWARD_STRICT:
        logger.info("使用 Walk-Forward 严格回测")
        df_pred = backtest_walkforward(data, feature_cols, date_col=date_col,
                                       model_builder=model_builder)
    else:
        logger.info("使用赛季切分回测")
        df_pred = backtest_by_season(data, feature_cols, model_builder=model_builder)

    metrics = evaluate_predictions(df_pred)
    logger.info(f"回测结果: {metrics}")
    return df_pred, metrics


# 模型工厂函数，供 walkforward 使用
def create_model():
    from train_ensemble import build_base_estimators, build_meta_learner
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    base = build_base_estimators()
    meta = build_meta_learner()
    return StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=TimeSeriesSplit(n_splits=3),
        stack_method='predict_proba'
    )


if __name__ == '__main__':
    # 加载历史数据
    data_path = 'data/historical/training_features.parquet'
    if not os.path.exists(data_path):
        logger.error("训练数据不存在，无法回测")
        sys.exit(1)
    data = pd.read_parquet(data_path)
    feature_cols = [c for c in data.columns if c not in ['target', 'date', 'season', 'game_id']]
    df_pred, metrics = run_backtest(data, feature_cols, model_builder=create_model)
    print(metrics)
