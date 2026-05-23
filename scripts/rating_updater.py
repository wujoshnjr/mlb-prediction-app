# scripts/rating_updater.py
"""
统一评级更新器：根据配置选择 ELO 或 Glicko2 引擎。
内置简单 ELO 更新，Glicko2 转换时移除主场优势（24 分）。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from datetime import datetime

import config
from scripts.glicko2_ratings import Glicko2League

logger = logging.getLogger(__name__)

ELO_FILE = 'data/elo_ratings.json'
GLICKO_FILE = 'data/glicko2_ratings.json'

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

def process_event(team_id, event_type):
    if config.RATINGS_ENGINE != 'glicko2':
        return
    league = load_glicko2_league()
    league.process_event(team_id, event_type)
    save_glicko2_league(league)
    logger.info(f"Glicko2 事件处理: 球队 {team_id}, 事件 {event_type}")

def update_ratings(game_results):
    if config.RATINGS_ENGINE == 'elo':
        elo = load_elo_ratings()
        for game in game_results:
            simple_elo_update(elo, game['home_team'], game['away_team'],
                              game['home_score'], game['away_score'])
        save_elo_ratings(elo)
        logger.info(f"ELO 更新完成，共处理 {len(game_results)} 场比赛")
    elif config.RATINGS_ENGINE == 'glicko2':
        league = load_glicko2_league()
        for game in game_results:
            home = game['home_team']
            away = game['away_team']
            if home not in league.teams: league.add_team(home)
            if away not in league.teams: league.add_team(away)
            if game['home_score'] > game['away_score']:
                result_home, result_away = 1.0, 0.0
            elif game['home_score'] < game['away_score']:
                result_home, result_away = 0.0, 1.0
            else:
                result_home, result_away = 0.5, 0.5
            league.teams[home].update(league.teams[away], result_home)
            league.teams[away].update(league.teams[home], result_away)
        save_glicko2_league(league)
        logger.info(f"Glicko2 更新完成，共处理 {len(game_results)} 场比赛")
    else:
        raise ValueError(f"未知的评级引擎: {config.RATINGS_ENGINE}")

if __name__ == '__main__':
    sample_results = [
        {'home_team': 'NYY', 'away_team': 'BOS', 'home_score': 5, 'away_score': 3, 'date': '2026-05-22'},
    ]
    update_ratings(sample_results)
