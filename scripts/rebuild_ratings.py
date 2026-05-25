# scripts/rebuild_ratings.py
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import config
from scripts.rating_updater import simple_elo_update, load_glicko2_league, save_glicko2_league, save_elo_ratings
from scripts.glicko2_ratings import Glicko2League

TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "D-backs", "Diamondbacks": "D-backs",
    "Atlanta Braves": "Braves", "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox", "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox", "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians", "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers", "Houston Astros": "Astros",
    "Kansas City Royals": "Royals", "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers", "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers", "Minnesota Twins": "Twins",
    "New York Mets": "Mets", "New York Yankees": "Yankees",
    "Oakland Athletics": "Athletics", "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies", "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres", "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners", "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays", "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "Washington Nationals": "Nationals",
    "D-backs": "D-backs", "Braves": "Braves", "Orioles": "Orioles",
    "Red Sox": "Red Sox", "Cubs": "Cubs", "White Sox": "White Sox",
    "Reds": "Reds", "Guardians": "Guardians", "Rockies": "Rockies",
    "Tigers": "Tigers", "Astros": "Astros", "Royals": "Royals",
    "Angels": "Angels", "Dodgers": "Dodgers", "Marlins": "Marlins",
    "Brewers": "Brewers", "Twins": "Twins", "Mets": "Mets",
    "Yankees": "Yankees", "Athletics": "Athletics", "Phillies": "Phillies",
    "Pirates": "Pirates", "Padres": "Padres", "Giants": "Giants",
    "Mariners": "Mariners", "Cardinals": "Cardinals", "Rays": "Rays",
    "Rangers": "Rangers", "Blue Jays": "Blue Jays", "Nationals": "Nationals"
}

def load_historical_games():
    games = []
    csv_file = "data/historical_predictions.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df = df.dropna(subset=['home_score','away_score'])
            if not df.empty:
                games.append(df[['game_id','game_date','home_team','away_team','home_score','away_score']])
    hist_dir = "data/historical"
    if os.path.exists(hist_dir):
        for f in os.listdir(hist_dir):
            if f.endswith('.parquet'):
                pdf = pd.read_parquet(os.path.join(hist_dir, f))
                if 'home_score' in pdf.columns and 'away_score' in pdf.columns:
                    pdf = pdf.dropna(subset=['home_score','away_score'])
                    if not pdf.empty:
                        games.append(pdf[['game_id','game_date','home_team','away_team','home_score','away_score']])
    if not games:
        return pd.DataFrame()
    all_games = pd.concat(games, ignore_index=True)
    all_games = all_games.drop_duplicates(subset=['game_id'])
    all_games['home_team'] = all_games['home_team'].map(TEAM_NAME_MAP).fillna(all_games['home_team'])
    all_games['away_team'] = all_games['away_team'].map(TEAM_NAME_MAP).fillna(all_games['away_team'])
    if 'game_date' in all_games.columns:
        all_games = all_games.sort_values('game_date')
    return all_games

def rebuild():
    df = load_historical_games()
    if df.empty:
        print("无可用数据"); sys.exit(1)

    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    if len(all_teams) < 30:
        print(f"球队数不足 ({len(all_teams)})"); sys.exit(1)

    elo = {team: 1500.0 for team in all_teams}
    league = Glicko2League()
    for team in all_teams:
        league.add_team(team, 1500.0, 350.0, 0.06)

    rated_ids = set()
    processed = 0
    skipped_missing = 0
    skipped_dup = 0

    for _, row in df.iterrows():
        gid = str(row.get('game_id'))
        if not gid or gid in {'None','nan'}: continue
        if gid in rated_ids: skipped_dup += 1; continue
        home = row['home_team']; away = row['away_team']
        try:
            home_score = int(row['home_score'])
            away_score = int(row['away_score'])
        except:
            skipped_missing += 1; continue

        # Elo
        simple_elo_update(elo, home, away, home_score, away_score)

        # Glicko2
        if home not in league.teams: league.add_team(home)
        if away not in league.teams: league.add_team(away)
        hteam = league.teams[home]; ateam = league.teams[away]
        h_snap = copy.deepcopy(ateam); a_snap = copy.deepcopy(hteam)
        if home_score > away_score:
            hteam.update(h_snap, 1.0); ateam.update(a_snap, 0.0)
        elif home_score < away_score:
            hteam.update(h_snap, 0.0); ateam.update(a_snap, 1.0)
        else:
            hteam.update(h_snap, 0.5); ateam.update(a_snap, 0.5)

        rated_ids.add(gid)
        processed += 1

    save_elo_ratings(elo)
    save_glicko2_league(league)
    with open('data/rated_game_ids.json','w') as f:
        json.dump(list(rated_ids), f)

    elo_vals = list(elo.values())
    glicko_vals = [t.rating for t in league.teams.values()]

    print("===== Rating Rebuild Report =====")
    print(f"Processed: {processed}, Skipped missing: {skipped_missing}, Dup: {skipped_dup}")
    print(f"Teams: {len(all_teams)}")
    print(f"Elo   min:{min(elo_vals):.1f} max:{max(elo_vals):.1f} range:{max(elo_vals)-min(elo_vals):.1f}")
    print(f"Glicko2 min:{min(glicko_vals):.1f} max:{max(glicko_vals):.1f} range:{max(glicko_vals)-min(glicko_vals):.1f}")
    print("================================")

    # 保存报告
    report = {
        "processed": processed,
        "skipped_missing": skipped_missing,
        "skipped_dup": skipped_dup,
        "team_count": len(all_teams),
        "elo_range": max(elo_vals)-min(elo_vals),
        "glicko_range": max(glicko_vals)-min(glicko_vals),
        "teams": sorted(all_teams)
    }
    with open('report/rating_rebuild_report.json','w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    rebuild()
