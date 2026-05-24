# scripts/batter_vs_pitch_client.py
import pandas as pd
import numpy as np
import os

# 修正：嘗試導入實際存在的函數 fetch_savant_statcast，若無則設為 None
try:
    from scripts.savant_client import fetch_savant_statcast as fetch_statcast_data
except ImportError:
    fetch_statcast_data = None
    print("Warning: fetch_savant_statcast not available, matchup features disabled.")

def get_matchup_lookup():
    return pd.DataFrame(columns=['batter_id', 'pitch_type', 'woba', 'whiff_rate', 'hard_hit_rate', 'avg_run_value'])

def add_matchup_features(features, home_sp_id, away_sp_id, home_top3_ids, away_top3_ids, lookup):
    return features
