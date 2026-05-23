# train_ensemble.py
"""
Stacking 集成训练脚本
支持 XGBoost + LightGBM + RandomForest + 可选 MLP
元学习器支持 LogisticRegression 或 ElasticNet (SGDClassifier)
通过 config.py 切换特性
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna

import config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型保存路径
MODEL_PATH = 'models/stacking_model.pkl'


def build_base_estimators():
    """根据配置构建第一层基础学习器列表"""
    estimators = [
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )),
        ('lgb', lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1
        )),
        ('rf', RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ))
    ]

    if config.MODEL_USE_MLP:
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.001,
            early_stopping=True,
            random_state=42
        )
        mlp_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', mlp)
        ])
        estimators.append(('mlp', mlp_pipe))
        logger.info("已添加 MLP 神经网络至基础学习器")

    return estimators


def build_meta_learner():
    """根据配置构建元学习器"""
    if config.MODEL_META == 'elasticnet':
        # 使用 SGDClassifier 实现 ElasticNet 正则化的逻辑回归
        meta = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            l1_ratio=0.5,
            alpha=0.0001,
            random_state=42,
            max_iter=2000,
            tol=1e-3
        )
        logger.info("元学习器: ElasticNet (SGDClassifier)")
    else:
        meta = LogisticRegression(
            random_state=42,
            max_iter=2000,
            solver='lbfgs'
        )
        logger.info("元学习器: LogisticRegression")
    return meta


def train_ensemble(X, y, sample_weights=None, n_splits=5, use_optuna=False, n_trials=30):
    """
    训练 Stacking 集成模型。
    
    参数:
        X: 特征 DataFrame/array
        y: 目标 Series/array
        sample_weights: 样本权重 (可选)
        n_splits: 时间序列交叉验证折数
        use_optuna: 是否使用 Optuna 优化超参 (简化版，可自定义)
        n_trials: Optuna 试验次数
    
    返回:
        trained StackingClassifier
    """
    # 确保是二分类
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError(f"目标变量必须为二分类，当前类别: {unique_labels}")

    # 构建基础学习器和元学习器
    base_estimators = build_base_estimators()
    final_estimator = build_meta_learner()

    # 内层交叉验证 (时序)
    inner_cv = TimeSeriesSplit(n_splits=n_splits)

    # 构建 Stacking 模型
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=inner_cv,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )

    if use_optuna:
        logger.info("启用 Optuna 超参数优化...")
        # 这里可以定义一个目标函数，对基础学习器和元学习器的关键参数进行搜索
        # 为保持简洁，本示例仅展示结构，实际可扩展
        study = optuna.create_study(direction='minimize')
        # 由于参数较多，略去完整 optuna 集成，保留占位
        logger.warning("Optuna 集成需根据实际超参空间定制，当前使用默认参数训练")
    
    # 训练模型
    logger.info(f"开始训练，样本数={len(X)}, 特征数={X.shape[1]}")
    stacking.fit(X, y, sample_weight=sample_weights)
    logger.info("训练完成")

    # 记录特征重要性（仅对树模型）
    try:
        importances = {}
        for name, est in stacking.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                importances[name] = est.feature_importances_
            elif hasattr(est, 'named_steps') and 'mlp' in est.named_steps:
                # MLP 没有特征重要性，跳过
                pass
        if importances:
            # 平均重要性（可选）
            avg_imp = np.mean(list(importances.values()), axis=0)
            importance_df = pd.DataFrame({
                'feature': X.columns if hasattr(X, 'columns') else range(X.shape[1]),
                'avg_importance': avg_imp
            }).sort_values('avg_importance', ascending=False)
            logger.info("Top 10 特征重要性:\n" + importance_df.head(10).to_string(index=False))
    except Exception as e:
        logger.warning(f"特征重要性计算失败: {e}")

    return stacking


def main():
    """主训练流程，可从命令行或 GitHub Actions 调用"""
    # 从数据库或CSV加载特征和目标
    try:
        data = pd.read_csv('data/training_features.csv')
        logger.info(f"加载训练数据: {data.shape}")
    except FileNotFoundError:
        logger.error("训练数据文件不存在: data/training_features.csv")
        return

    # 假设目标列名为 'target' (1=主胜, 0=客胜)
    target_col = 'target'
    if target_col not in data.columns:
        logger.error(f"目标列 '{target_col}' 不存在")
        return

    # 分离特征和日期（如果有）
    date_col = 'date' if 'date' in data.columns else None
    feature_cols = [c for c in data.columns if c not in [target_col, date_col, 'game_id']]

    X = data[feature_cols]
    y = data[target_col]

    # 样本权重：时效性衰减（可选）
    sample_weights = None
    if date_col and date_col in data.columns:
        dates = pd.to_datetime(data[date_col])
        max_date = dates.max()
        # 指数衰减，半衰期约180天
        days_diff = (max_date - dates).dt.days
        sample_weights = np.exp(-days_diff / 260)  # 260天 ≈ 一个赛季
        logger.info("已计算时效性样本权重")

    # 训练模型
    model = train_ensemble(X, y, sample_weights=sample_weights,
                           n_splits=5, use_optuna=False)

    # 保存模型
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"模型已保存至: {MODEL_PATH}")

    # 输出基本性能（使用交叉验证粗略估计）
    tscv = TimeSeriesSplit(n_splits=3)
    cv_preds = cross_val_predict(model, X, y, cv=tscv, method='predict_proba')[:, 1]
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    logger.info(f"CV Brier Score: {brier_score_loss(y, cv_preds):.4f}")
    logger.info(f"CV Log Loss: {log_loss(y, cv_preds):.4f}")
    logger.info(f"CV AUC: {roc_auc_score(y, cv_preds):.4f}")


if __name__ == '__main__':
    main()
