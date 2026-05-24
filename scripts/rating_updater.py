# scripts/rating_updater.py
"""
統一評級更新器：根據 config 選擇 ELO 或 Glicko2。
新增：基於 game_id 的去重機制，防止同一場比賽重複更新。
"""
import sys, os, json, logging, copy
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.glicko2_ratings import Glicko2League

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ELO_FILE = 'data/elo_ratings.json'
GLICKO_FILE = 'data/glicko2_ratings.json'
RATED_GAMES_FILE = 'data/rated_game_ids.json'
FINAL_RESULTS_FILE = 'data/new_final_results.json'

def load_elo_ratings():
    if os.path.exists(ELO_FILE):
        with open(ELO_FILE) as f:
            return json.load(f)
    return {}

def save_elo_ratings(ratings):
    with open(ELO_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)

def load_glicko2_league():
    if os.path.exists(GLICKO_FILE):
        return Glicko2League.load(GLICKO_FILE)
    else:
        league = Glicko2League()
        elo = load_elo_ratings()
        for team_id, elo_val in elo.items():
            adjusted_elo = elo_val - 24
            league.add_team(team_id, rating=adjusted_elo, rd=350, vol=0.06)
        return league

def save_glicko2_league(league):
    league.save(GLICKO_FILE)

def load_rated_game_ids():
    if os.path.exists(RATED_GAMES_FILE):
        with open(RATED_GAMES_FILE) as f:
            return set(json.load(f))
    return set()

def save_rated_game_ids(game_ids):
    with open(RATED_GAMES_FILE, 'w') as f:
        json.dump(list(game_ids), f)

def simple_elo_update(elo_dict, home_team, away_team, home_score, away_score, K=32, home_adv=24):
    r_home = elo_dict.get(home_team, 1500)
    r_away = elo_dict.get(away_team, 1500)
    expected_home = 1 / (1 + 10 ** ((r_away - (r_home + home_adv)) / 400))
    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5
    elo_dict[home_team] = r_home + K * (actual_home - expected_home)
    elo_dict[away_team] = r_away + K * ((1 - actual_home) - (1 - expected_home))
    return elo_dict

def update_ratings(game_results):
    rated_ids = load_rated_game_ids()
    new_games = []
    skipped = []

    for game in game_results:
        gid = str(game.get('game_id'))
        if not gid:
            continue
        if gid in rated_ids:
            skipped.append(gid)
        else:
            new_games.append(game)

    if skipped:
        logger.info(f"跳过 {len(skipped)} 场已处理比赛: {skipped[:5]}...")

    if not new_games:
        logger.info("没有新的比赛需要更新")
        return

    logger.info(f"开始更新 {len(new_games)} 场新比赛")

    if config.RATINGS_ENGINE == 'elo':
        elo = load_elo_ratings()
        for game in new_games:
            home = game['home_team']
            away = game['away_team']
            if home not in elo: elo[home] = 1500
            if away not in elo: elo[away] = 1500
            simple_elo_update(elo, home, away, game['home_score'], game['away_score'])
        save_elo_ratings(elo)
    elif config.RATINGS_ENGINE == 'glicko2':
        league = load_glicko2_league()
        for game in new_games:
            home = game['home_team']
            away = game['away_team']
            if home not in league.teams: league.add_team(home)
            if away not in league.teams: league.add_team(away)

            home_team = league.teams[home]
            away_team = league.teams[away]
            home_opponent_snapshot = copy.deepcopy(away_team)
            away_opponent_snapshot = copy.deepcopy(home_team)

            if game['home_score'] > game['away_score']:
                home_team.update(home_opponent_snapshot, 1.0)
                away_team.update(away_opponent_snapshot, 0.0)
            elif game['home_score'] < game['away_score']:
                home_team.update(home_opponent_snapshot, 0.0)
                away_team.update(away_opponent_snapshot, 1.0)
            else:
                home_team.update(home_opponent_snapshot, 0.5)
                away_team.update(away_opponent_snapshot, 0.5)
        save_glicko2_league(league)
    else:
        raise ValueError(f"未知的評級引擎: {config.RATINGS_ENGINE}")

    for game in new_games:
        rated_ids.add(str(game['game_id']))
    save_rated_game_ids(rated_ids)

    logger.info(f"新处理 rating 比赛数={len(new_games)}，跳过重复={len(skipped)}，rated_game_ids 总数={len(rated_ids)}")

def main():
    if not os.path.exists(FINAL_RESULTS_FILE):
        logger.error(f"缺少 {FINAL_RESULTS_FILE}，請先執行 update_results.py")
        return
    with open(FINAL_RESULTS_FILE) as f:
        games = json.load(f)
    if not games:
        logger.info("沒有新比賽資料")
        return
    update_ratings(games)

if __name__ == '__main__':
    main()
