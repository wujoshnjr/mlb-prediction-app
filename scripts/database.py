import sqlite3
import os
import pandas as pd

DB_PATH = "data/mlb_predictions.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, game_date TEXT, home_team TEXT, away_team TEXT,
            pred_home_win REAL, home_odds REAL, elo_home REAL, elo_away REAL,
            ml_rec TEXT, spread_rec TEXT, total_rec TEXT, nrfi_rec TEXT, nrfi_prob REAL,
            pred_uncertainty REAL, kelly_fraction REAL, home_win INTEGER,
            manual_no_odds_pred REAL,
            elo_diff REAL, market_prob REAL, sp_era_diff REAL, sp_fip_diff REAL,
            bullpen_ip_diff REAL, rest_diff REAL, park_factor REAL,
            dynamic_park_factor REAL,
            platoon_ops_diff REAL,
            statcast_launch_speed_diff REAL, statcast_barrel_diff REAL,
            statcast_hard_hit_diff REAL, statcast_woba_diff REAL,
            timezone_diff REAL, is_day_game INTEGER,
            home_back2back INTEGER, away_back2back INTEGER,
            catcher_era_diff REAL, cs_diff REAL, wind_effect REAL,
            temp_effect REAL, precip_effect REAL, injury_diff REAL,
            dynamic_pythag_diff REAL, log5_prob REAL,
            lag30_winrate_diff REAL, lag30_runs_diff REAL,
            pitch_movement_diff REAL,
            k_pct_diff REAL, bb_pct_diff REAL, avg_bat_speed_diff REAL,
            pitcher_rating_diff REAL, odds_change REAL,
            zone_size REAL, k_rate REAL, bullpen_availability_diff REAL,
            elo_momentum_7d REAL, elo_momentum_30d REAL, barrel_pa_diff REAL, hardhit_pa_diff REAL,
            swing_miss_diff REAL, csw_diff REAL, barrel_bb_pct_diff REAL,
            sprint_speed_diff REAL, pitch_type_matchup_score REAL,
            home_top3_woba REAL, away_top3_woba REAL,
            closing_odds REAL, top_features TEXT, market_divergence INTEGER, odds_source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_prediction(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    columns = [
        "game_id","game_date","home_team","away_team","pred_home_win","home_odds",
        "elo_home","elo_away","ml_rec","spread_rec","total_rec","nrfi_rec","nrfi_prob",
        "pred_uncertainty","kelly_fraction","home_win","manual_no_odds_pred","elo_diff","market_prob",
        "sp_era_diff","sp_fip_diff","bullpen_ip_diff","rest_diff","park_factor",
        "dynamic_park_factor","platoon_ops_diff","statcast_launch_speed_diff",
        "statcast_barrel_diff","statcast_hard_hit_diff","statcast_woba_diff",
        "timezone_diff","is_day_game","home_back2back","away_back2back",
        "catcher_era_diff","cs_diff","wind_effect","temp_effect","precip_effect",
        "injury_diff","dynamic_pythag_diff","log5_prob","lag30_winrate_diff","lag30_runs_diff",
        "pitch_movement_diff","k_pct_diff","bb_pct_diff","avg_bat_speed_diff",
        "pitcher_rating_diff","odds_change","zone_size","k_rate",
        "bullpen_availability_diff","elo_momentum_7d","elo_momentum_30d",
        "barrel_pa_diff","hardhit_pa_diff","swing_miss_diff","csw_diff","barrel_bb_pct_diff",
        "sprint_speed_diff","pitch_type_matchup_score","home_top3_woba","away_top3_woba",
        "closing_odds","top_features","market_divergence","odds_source"
    ]
    placeholders = ",".join(["?" for _ in columns])
    values = [data.get(col, None) for col in columns]
    cursor.execute(f"INSERT INTO predictions ({','.join(columns)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()

def update_game_result(game_id, home_win):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE predictions SET home_win = ? WHERE game_id = ? AND home_win IS NULL", (home_win, game_id))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

def get_performance_metrics():
    conn = get_connection()
    query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN ml_rec LIKE 'Bet%' THEN 1 ELSE 0 END) as bets,
            SUM(CASE WHEN home_win IS NOT NULL THEN 1 ELSE 0 END) as completed,
            AVG(CASE WHEN ml_rec LIKE 'Bet%' AND home_win = 1 THEN 1.0 ELSE 0.0 END) as win_rate
        FROM predictions
        WHERE home_win IS NOT NULL
    """
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}
