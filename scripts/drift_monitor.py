# scripts/drift_monitor.py
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

HISTORY_FILE = "data/historical_predictions.csv"
DRIFT_ALERT_FILE = "report/drift_alert.json"
PSI_THRESHOLD = 0.25

EXPECTED_FEATURES = [
    'elo_diff',
    'sp_era_diff', 'sp_fip_diff', 'sp_stuff_plus_diff', 'sp_csw_diff',
    'bullpen_ip_diff', 'rest_diff',
    'dynamic_park_factor',
    'platoon_ops_diff', 'statcast_launch_speed_diff', 'statcast_barrel_diff',
    'statcast_hard_hit_diff', 'statcast_woba_diff',
    'timezone_diff', 'is_day_game', 'back2back_diff',
    'catcher_era_diff', 'cs_diff', 'wind_effect',
    'temp_effect', 'precip_effect', 'injury_diff',
    'dynamic_pythag_diff', 'log5_prob', 'lag30_winrate_diff', 'lag30_runs_diff',
    'pitch_movement_diff',
    'k_pct_diff', 'bb_pct_diff', 'avg_bat_speed_diff',
    'pitcher_rating_diff', 'odds_change', 'odds_momentum',
    'zone_size', 'k_rate', 'bullpen_availability_diff',
    'elo_momentum_7d', 'elo_momentum_30d', 'barrel_pa_diff', 'hardhit_pa_diff',
    'swing_miss_diff', 'csw_diff', 'barrel_bb_pct_diff',
    'sprint_speed_diff', 'pitch_type_matchup_score',
    'top3_woba_diff', 'winrate_diff', 'bt_strength_diff',
    'home_usage_magnitude', 'away_usage_magnitude',
    'home_shift_score', 'away_shift_score',
    'home_delta_FF', 'home_delta_SL', 'home_delta_CH', 'home_delta_CU',
    'home_delta_FC', 'home_delta_SI', 'home_delta_KC', 'home_delta_FS',
    'away_delta_FF', 'away_delta_SL', 'away_delta_CH', 'away_delta_CU',
    'away_delta_FC', 'away_delta_SI', 'away_delta_KC', 'away_delta_FS'
]

def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), bins+1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected) + 1e-10
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual) + 1e-10
    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return psi

def run_drift_monitor():
    if not os.path.exists(HISTORY_FILE):
        print("历史文件不存在")
        return

    df = pd.read_csv(HISTORY_FILE)
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    if len(df) < 200:
        print("数据量不足以计算漂移")
        return

    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if 'game_date' in df.columns:
        df = df.sort_values('game_date')
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    recent = df.iloc[split_idx:]

    psi_total = 0
    psi_details = {}
    for feat in EXPECTED_FEATURES:
        psi_val = calculate_psi(train[feat].values, recent[feat].values)
        psi_total += psi_val
        psi_details[feat] = round(psi_val, 4)

    avg_psi = psi_total / len(EXPECTED_FEATURES)
    print(f"平均 PSI: {avg_psi:.4f}")

    alert_data = {
        "timestamp": datetime.now().isoformat(),
        "avg_psi": round(avg_psi, 4),
        "psi_details": psi_details,
        "alert": avg_psi > PSI_THRESHOLD
    }

    os.makedirs("report", exist_ok=True)
    with open(DRIFT_ALERT_FILE, 'w') as f:
        json.dump(alert_data, f, indent=2)

    if alert_data["alert"]:
        print(f"⚠️ 概念漂移警告！平均 PSI = {avg_psi:.4f} > 阈值 {PSI_THRESHOLD}")
    else:
        print(f"✅ 特征分布稳定，平均 PSI = {avg_psi:.4f}")

if __name__ == "__main__":
    run_drift_monitor()
