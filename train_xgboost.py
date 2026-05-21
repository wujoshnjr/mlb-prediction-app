#!/usr/bin/env python
"""
训练 XGBoost 模型并进行概率校准（含所有高级特征）
"""
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"

# 特征列表（必须与 prediction.py 中传递给模型的特征顺序完全一致）
EXPECTED_FEATURES = [
    'elo_diff',
    'market_prob',
    'sp_era_diff',
    'sp_fip_diff',
    'bullpen_ip_diff',
    'rest_diff',
    'park_factor',
    'platoon_ops_diff',
    'statcast_launch_speed_diff',
    'statcast_barrel_diff',
    'statcast_hard_hit_diff',
    'statcast_woba_diff',
    'timezone_diff',
    'is_day_game',
    'home_back2back',
    'away_back2back',
    'catcher_era_diff',
    'cs_diff',
    'wind_effect'
]

def prepare_data():
    df = pd.read_csv(HISTORY_FILE)
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    df['home_win'] = df['home_win'].astype(int)

    if len(df) < 30:
        print(f"数据量不足 ({len(df)} 条)，跳过训练")
        return None, None

    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            print(f"警告：缺少列 {col}，将用 0 填充")
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    X = df[EXPECTED_FEATURES].values
    y = df['home_win'].values

    variances = np.var(X, axis=0)
    if np.all(variances < 1e-8):
        print("所有特征方差接近0，无法训练有效模型")
        return None, None

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

    try:
        importances = calibrated.estimator_.feature_importances_
        for name, imp in zip(EXPECTED_FEATURES, importances):
            print(f"{name}: {imp:.4f}")
    except Exception as e:
        print(f"无法获取特征重要性: {e}")

if __name__ == "__main__":
    train()
