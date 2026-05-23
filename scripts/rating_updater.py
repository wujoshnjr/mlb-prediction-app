# scripts/rating_updater.py
"""
统一评级更新器：根据配置选择 ELO 或 Glicko2 引擎。
保留原有 ELO 更新逻辑，并通过事件驱动 RD 膨胀 (Glicko2) 支持伤病、先发更换等。
"""

import logging
import os
import json
import math
from datetime import datetime
import pandas as pd

import config
from scripts.elo_updater import update_elo_for_game  # 原有 ELO 更新函数
from scripts.glicko2_ratings import Glicko2League

logger = logging.getLogger(__name__)

# ELO 文件路径（沿用原有路径，便于兼容）
ELO_FILE = 'data/elo_ratings.json'
GLICKO_FILE = 'data/glicko2_ratings.json'


def load_elo_ratings():
    """加载 ELO 评分字典"""
    if os.path.exists(ELO_FILE):
        with open(ELO_FILE) as f:
            return json.load(f)
    return {}


def save_elo_ratings(ratings):
    with open(ELO_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)


def load_glicko2_league():
    """加载 Glicko2 联赛对象"""
    if os.path.exists(GLICKO_FILE):
        return Glicko2League.load(GLICKO_FILE)
    else:
        # 若没有历史文件，尝试从 ELO 转换
        league = Glicko2League()
        elo = load_elo_ratings()
        for team_id, elo_val in elo.items():
            # 简单映射：ELO 1500 -> mu 1500，RD 取默认 350
            league.add_team(team_id, rating=elo_val, rd=350, vol=0.06)
        return league


def save_glicko2_league(league):
    league.save(GLICKO_FILE)


def process_event(team_id, event_type):
    """
    向评级系统注入事件。
    当 config.RATINGS_ENGINE == 'glicko2' 时，增加对应球队的 RD (不确定度)。
    ELO 模式下直接忽略。
    """
    if config.RATINGS_ENGINE != 'glicko2':
        return
    league = load_glicko2_league()
    league.process_event(team_id, event_type)
    save_glicko2_league(league)
    logger.info(f"Glicko2 事件处理: 球队 {team_id}, 事件 {event_type}")


def update_ratings(game_results):
    """
    根据比赛结果更新评级系统。
    game_results: list of dict，每个 dict 包含:
        - home_team: str
        - away_team: str
        - home_score: int
        - away_score: int
        - date: str or datetime (可选)
    """
    if config.RATINGS_ENGINE == 'elo':
        # 使用原有 ELO 更新
        elo = load_elo_ratings()
        for game in game_results:
            home = game['home_team']
            away = game['away_team']
            # 确保球队存在于字典中
            if home not in elo:
                elo[home] = 1500
            if away not in elo:
                elo[away] = 1500
            # 调用原有单场更新函数
            update_elo_for_game(elo, home, away, game['home_score'], game['away_score'])
        save_elo_ratings(elo)
        logger.info(f"ELO 更新完成，共处理 {len(game_results)} 场比赛")

    elif config.RATINGS_ENGINE == 'glicko2':
        league = load_glicko2_league()
        for game in game_results:
            home = game['home_team']
            away = game['away_team']
            # 确保队伍存在
            if home not in league.teams:
                league.add_team(home)
            if away not in league.teams:
                league.add_team(away)
            # 计算得分结果：1.0 代表主胜，0.0 客胜，0.5 平局（极少）
            if game['home_score'] > game['away_score']:
                result_home = 1.0
                result_away = 0.0
            elif game['home_score'] < game['away_score']:
                result_home = 0.0
                result_away = 1.0
            else:
                result_home = 0.5
                result_away = 0.5
            # 获取队伍对象并更新
            team_home = league.teams[home]
            team_away = league.teams[away]
            team_home.update(team_away, result_home)
            team_away.update(team_home, result_away)
        save_glicko2_league(league)
        logger.info(f"Glicko2 更新完成，共处理 {len(game_results)} 场比赛")
    else:
        raise ValueError(f"未知的评级引擎: {config.RATINGS_ENGINE}")


# 如需从命令行独立运行
if __name__ == '__main__':
    # 示例用法
    sample_results = [
        {'home_team': 'NYY', 'away_team': 'BOS', 'home_score': 5, 'away_score': 3, 'date': '2026-05-22'},
        {'home_team': 'LAD', 'away_team': 'SFG', 'home_score': 1, 'away_score': 2, 'date': '2026-05-22'},
    ]
    update_ratings(sample_results)
