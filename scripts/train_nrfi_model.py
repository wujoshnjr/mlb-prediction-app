# scripts/train_nrfi_model.py
"""
訓練 NRFI 模型，使用 historical_predictions.csv 或 data/historical/*.parquet
需包含：home_sp_first_inning_era, away_sp_first_inning_era, home_top3_avg_woba, ...
"""
import pandas as pd
import numpy as np
import os
import glob
from scripts.nrf_model import NRFIModel

def load_historical_features():
    # 優先從 historical_predictions.csv 讀取，因為它含有首局得分
    csv_path = 'data/historical_predictions.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 篩選有首局得分和所需特徵的行
        if 'first_inning_runs' in df.columns:
            return df.dropna(subset=['first_inning_runs'])

    # 否則從 historical/*.parquet 組合
    hist_dir = 'data/historical'
    parquet_files = glob.glob(os.path.join(hist_dir, '*.parquet'))
    if parquet_files:
        df = pd.concat([pd.read_parquet(f) for f in parquet_files])
        # 假設 parquet 中有 game_date, home_team, away_team, first_inning_runs 等
        # 可能需要先處理特徵（簡化，這裡直接使用已有欄位）
        return df.dropna(subset=['first_inning_runs'])

    raise ValueError("No training data found with first_inning_runs.")

if __name__ == '__main__':
    data = load_historical_features()
    print(f"Loaded {len(data)} games with first inning data.")

    # 初始化並訓練
    model = NRFIModel('models/nrf_model.pkl')
    model.train_pipeline(data)   # train_pipeline 內部會處理特徵篩選和擬合
    print("NRFI model saved to models/nrf_model.pkl")
