#!/usr/bin/env python
"""
训练 XGBoost 模型并进行概率校准（健壮版，自动处理缺失列）
"""
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"

# 期望的特征列
EXPECTED_FEATURES = ['elo_diff', 'market_prob', 'sp_era_diff', 'bullpen_era_diff', 'rest_diff']

def prepare_data():
    df = pd.read_csv(HISTORY_FILE)
    # 删除没有结果的记录
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    df['home_win'] = df['home_win'].astype(int)

    if len(df) < 30:
        print(f"数据量不足 ({len(df)} 条)，跳过训练")
        return None, None

    # 确保所有特征列都存在，缺失则填充0
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            print(f"警告：缺少列 {col}，将用 0 填充")
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    X = df[EXPECTED_FEATURES].values
    y = df['home_win'].values
    return X, y

def train():
    X, y = prepare_data()
    if X is None:
        return

    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    calibrated = CalibratedClassifierCV(
        estimator=xgb,
        method='isotonic',
        cv=tscv
    )
    calibrated.fit(X, y)

    joblib.dump(calibrated, MODEL_OUTPUT)
    print(f"模型已保存至 {MODEL_OUTPUT}")

    importances = xgb.feature_importances_
    for name, imp in zip(EXPECTED_FEATURES, importances):
        print(f"{name}: {imp:.4f}")

if __name__ == "__main__":
    train()
