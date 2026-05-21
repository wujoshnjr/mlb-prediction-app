#!/usr/bin/env python
"""
训练 XGBoost 模型并进行概率校准（Optuna 调参）
"""
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"

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
    'wind_effect',
    'pythag_diff',
    'log5_prob',
    'lag30_winrate_diff',    # 新增滞后特征
    'lag30_runs_diff'
]

def prepare_data():
    df = pd.read_csv(HISTORY_FILE)
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    df['home_win'] = df['home_win'].astype(int)

    if len(df) < 50:
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

    # 分割训练/验证集（时间序列，保留最后20%作为验证）
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        xgb = XGBClassifier(**params)
        xgb.fit(X_train, y_train)
        preds = xgb.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds)

    print("开始 Optuna 超参数搜索...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best_params = study.best_params
    print("最佳参数:", best_params)

    # 用全部数据重新训练
    tscv = TimeSeriesSplit(n_splits=3)
    final_xgb = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
    calibrated = CalibratedClassifierCV(
        estimator=final_xgb,
        method='isotonic',
        cv=tscv
    )
    calibrated.fit(X, y)

    joblib.dump(calibrated, MODEL_OUTPUT)
    print(f"模型已保存至 {MODEL_OUTPUT}")

    # 特征重要性
    try:
        importances = calibrated.estimator_.feature_importances_
        for name, imp in zip(EXPECTED_FEATURES, importances):
            print(f"{name}: {imp:.4f}")
    except Exception as e:
        print(f"无法获取特征重要性: {e}")

if __name__ == "__main__":
    train()
