"""
Glicko2 评级系统（简化版）
包含事件驱动 RD 膨胀。
"""
import json
import os
import numpy as np

RATING_FILE = "data/glicko2_ratings.json"

class Glicko2System:
    def __init__(self, default_rating=1500, default_rd=80, default_vol=0.06, tau=0.5):
        self.default_rating = default_rating
        self.default_rd = default_rd
        self.default_vol = default_vol
        self.tau = tau
        self.ratings = self._load()

    def _load(self):
        if os.path.exists(RATING_FILE):
            with open(RATING_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        os.makedirs('data', exist_ok=True)
        with open(RATING_FILE, 'w') as f:
            json.dump(self.ratings, f, indent=2)

    def get(self, team):
        if team not in self.ratings:
            self.ratings[team] = {
                'rating': self.default_rating,
                'rd': self.default_rd,
                'vol': self.default_vol
            }
        return self.ratings[team]

    def update(self, home, away, home_win, date_str=None):
        # 简化版：仅更新 RD，不做完整 Glicko2 计算（后续完善）
        home_info = self.get(home)
        away_info = self.get(away)
        # 此处省略完整 Glicko2 公式，仅演示 RD 膨胀逻辑
        self.save()

    def apply_rd_shock(self, team, shock_factor):
        """事件驱动的 RD 膨胀"""
        info = self.get(team)
        info['rd'] = min(info['rd'] * (1 + 0.25 * shock_factor), 100)
        self.save()

    def reset_season_rd(self):
        """新赛季 RD 重置"""
        for team in self.ratings:
            old_rd = self.ratings[team]['rd']
            self.ratings[team]['rd'] = 0.7 * old_rd + 0.3 * self.default_rd
            self.ratings[team]['rd'] = min(self.ratings[team]['rd'], 100)
        self.save()
