import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import joblib

def train_model(data_path="data/historical/training_data.csv"):
    if not os.path.exists(data_path):
        print("训练数据不存在，请先收集历史数据。")
        return

    df = pd.read_csv(data_path)
    # 假设特征列：elo_diff, sp_era_diff, bullpen_era_diff, park_factor, ...
    feature_cols = ["elo_diff", "sp_era_diff", "bullpen_era_diff", "park_factor"]
    X = df[feature_cols]
    y = df["home_win"]

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # 简单训练（后续可加 Optuna 调参）
    model.fit(X, y)
    joblib.dump(model, "data/xgb_model.pkl")
    print("模型已保存到 data/xgb_model.pkl")

    # 特征重要性
    importance = model.feature_importances_
    for name, imp in zip(feature_cols, importance):
        print(f"{name}: {imp:.4f}")

if __name__ == "__main__":
    import os
    train_model()
