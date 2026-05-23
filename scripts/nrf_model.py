# scripts/nrf_model.py
"""
NRFI 独立分类器 —— 预测 MLB 比赛首局是否无得分 (No Run First Inning)
完全复用主模型的 Stacking 框架，使用独立特征集训练，通过 config.NRFI_USE_ML 开关控制。
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

import config

logger = logging.getLogger(__name__)

class NRFIModel:
    """
    NRFI 模型包装类，负责训练、预测和持久化。
    """

    def __init__(self, model_path='models/nrf_model.pkl'):
        self.model = None
        self.model_path = model_path
        self.feature_cols = [
            'home_sp_first_inning_era',   # 主队先发首局 ERA
            'away_sp_first_inning_era',   # 客队先发首局 ERA
            'home_top3_avg_woba',         # 主队前3棒 wOBA
            'away_top3_avg_woba',         # 客队前3棒 wOBA
            'umpire_k_rate',              # 裁判 K%
            'umpire_zone_size',           # 裁判好球带大小
            'temperature',                # 温度
            'wind_speed',                 # 风速
            'park_hr_factor',             # 球场本垒打因子
            'home_matchup_adv',           # 主队投手对客队打线优势（新增特征）
            'away_matchup_adv',           # 客队投手对主队打线优势
            'is_day_game',                # 是否为日场（可选）
        ]
        self._ensure_model_directory()

    def _ensure_model_directory(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def build_model(self):
        """构建与主模型相同的 Stacking 架构 (可根据 config 开关加入 MLP)"""
        base_estimators = [
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                      use_label_encoder=False, random_state=42)),
            ('lgb', lgb.LGBMClassifier(objective='binary', random_state=42)),
            ('rf', RandomForestClassifier(random_state=42))
        ]
        if config.MODEL_USE_MLP:
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                                alpha=0.001, early_stopping=True, random_state=42)
            mlp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', mlp)
            ])
            base_estimators.append(('mlp', mlp_pipe))

        # 元学习器同样跟随主配置
        if config.MODEL_META == 'elasticnet':
            from sklearn.linear_model import SGDClassifier
            meta = SGDClassifier(loss='log_loss', penalty='elasticnet',
                                 l1_ratio=0.5, alpha=0.0001,
                                 random_state=42, max_iter=1000)
        else:
            meta = LogisticRegression(random_state=42, max_iter=1000)

        self.model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=TimeSeriesSplit(n_splits=3),  # 内层时序交叉验证
            stack_method='predict_proba'
        )

    def fit(self, X_train, y_train):
        """训练 NRFI 模型"""
        if self.model is None:
            self.build_model()
        logger.info(f"训练 NRFI 模型，样本数: {len(X_train)}, 特征数: {X_train.shape[1]}")
        # 注意：MLP 需要标准化，已经在管道内处理
        self.model.fit(X_train, y_train)
        logger.info("NRFI 模型训练完成")

    def predict_proba(self, X):
        """返回首局无得分的概率 (类别 1)"""
        if self.model is None:
            raise RuntimeError("NRFI 模型未训练或未加载")
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, self.model_path)
        logger.info(f"NRFI 模型已保存至 {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info("NRFI 模型加载成功")
        else:
            logger.warning(f"NRFI 模型文件不存在: {self.model_path}，请先训练")

    def train_pipeline(self, historical_data):
        """
        完整训练流程。
        参数:
            historical_data: DataFrame，必须包含所有 feature_cols 和 target 列 'first_inning_runs'
        """
        # 目标变量：首局无得分 = 1
        y = (historical_data['first_inning_runs'] == 0).astype(int)
        X = historical_data[self.feature_cols].copy()

        # 填补缺失值（首局 ERA 等可能缺失）
        # 简单填充为联盟均值或 0，在实际应用中可用更精细的估算
        X = X.fillna({
            'home_sp_first_inning_era': 4.50,
            'away_sp_first_inning_era': 4.50,
            'home_top3_avg_woba': 0.320,
            'away_top3_avg_woba': 0.320,
            'umpire_k_rate': 0.22,
            'umpire_zone_size': 1.0,
            'temperature': 70,
            'wind_speed': 5,
            'park_hr_factor': 1.0,
            'home_matchup_adv': 0.0,
            'away_matchup_adv': 0.0,
            'is_day_game': 0
        })

        self.fit(X, y)
        self.save()
        return self


# 辅助函数：提取单场比赛的 NRFI 特征（供 prediction.py 调用）
def extract_nrf_features(game, matchup_lookup=None):
    """
    从比赛信息字典中提取 NRFI 模型需要的特征。
    可根据实际数据结构调整字段名。
    """
    feats = {}
    # 先发投手首局 ERA
    feats['home_sp_first_inning_era'] = game.get('home_sp_first_inning_era', 4.50)
    feats['away_sp_first_inning_era'] = game.get('away_sp_first_inning_era', 4.50)

    # 前三棒 wOBA
    feats['home_top3_avg_woba'] = game.get('home_top3_avg_woba', 0.320)
    feats['away_top3_avg_woba'] = game.get('away_top3_avg_woba', 0.320)

    # 裁判
    feats['umpire_k_rate'] = game.get('umpire_k_rate', 0.22)
    feats['umpire_zone_size'] = game.get('umpire_zone_size', 1.0)

    # 天气
    feats['temperature'] = game.get('temperature', 70)
    feats['wind_speed'] = game.get('wind_speed', 5)

    # 球场
    feats['park_hr_factor'] = game.get('park_hr_factor', 1.0)

    # 对位特征（如果有）
    feats['home_matchup_adv'] = game.get('home_matchup_adv', 0.0)
    feats['away_matchup_adv'] = game.get('away_matchup_adv', 0.0)

    # 日场
    feats['is_day_game'] = 1 if game.get('is_day_game') else 0

    return feats


# 快速训练入口（独立运行）
if __name__ == '__main__':
    print("NRFI 模型模块已就绪。使用示例:")
    print("from scripts.nrf_model import NRFIModel")
    print("model = NRFIModel()")
    print("model.train_pipeline(historical_data)")
