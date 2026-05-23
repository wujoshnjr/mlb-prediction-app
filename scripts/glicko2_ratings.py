# scripts/glicko2_ratings.py
"""
Glicko2 评级系统完整实现，包含事件驱动 RD 膨胀。
参考：Mark E. Glickman, "Example of the Glicko-2 system"
"""

import math
import json
import os
from typing import Dict, Optional

# Glicko2 常数：将 rating 和 RD 转换到 Glicko2 尺度
_SCALE = 173.7178
_TAU = 0.5          # 系统常数，控制评分波动性变化速度
_EPSILON = 0.000001  # 收敛精度


def _g(phi: float) -> float:
    """g(φ) 函数"""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """预期得分函数"""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


class Glicko2Team:
    """单支队伍的 Glicko2 评级"""
    def __init__(self, rating: float = 1500.0, rd: float = 350.0, vol: float = 0.06):
        # 原始尺度
        self.rating = rating      # μ
        self.rd = rd              # φ
        self.vol = vol            # σ

    def to_glicko2_scale(self):
        """将 rating 和 RD 转换到 Glicko2 内部尺度"""
        mu = (self.rating - 1500.0) / _SCALE
        phi = self.rd / _SCALE
        return mu, phi, self.vol

    def from_glicko2_scale(self, mu: float, phi: float, sigma: float):
        """从内部尺度恢复到原始尺度"""
        self.rating = mu * _SCALE + 1500.0
        self.rd = phi * _SCALE
        self.vol = sigma
        # 确保 RD 不超过 350
        if self.rd > 350.0:
            self.rd = 350.0
        if self.rd < 30.0:
            self.rd = 30.0

    def update(self, opponent: 'Glicko2Team', score: float) -> None:
        """
        根据对手和比赛结果更新自身评级。
        score: 1.0 表示胜利，0.0 失败，0.5 平局（极少）。
        """
        # 转换为内部尺度
        mu, phi, sigma = self.to_glicko2_scale()
        mu_j, phi_j, _ = opponent.to_glicko2_scale()

        # 1. 计算 v 和 Δ
        g_j = _g(phi_j)
        e_j = _E(mu, mu_j, phi_j)
        v = 1.0 / (g_j**2 * e_j * (1.0 - e_j))
        delta = v * g_j * (score - e_j)

        # 2. 更新波动性 σ'
        sigma_new = self._compute_new_volatility(sigma, phi, v, delta)

        # 3. 更新 RD（加入波动增量）
        phi_star = math.sqrt(phi**2 + sigma_new**2)

        # 4. 更新 RD 和 rating
        phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        mu_new = mu + phi_new**2 * g_j * (score - e_j)

        # 5. 转换回原始尺度
        self.from_glicko2_scale(mu_new, phi_new, sigma_new)

    def _compute_new_volatility(self, sigma: float, phi: float, v: float, delta: float) -> float:
        """使用迭代法计算新的波动性 σ'"""
        a = math.log(sigma**2)
        delta2 = delta**2
        phi2 = phi**2

        def f(x):
            exp_x = math.exp(x)
            return (exp_x * (delta2 - phi2 - v - exp_x) / (2.0 * (phi2 + v + exp_x)**2)
                    - (x - a) / (_TAU**2))

        # 初始区间
        A = a
        if delta2 > phi2 + v:
            B = math.log(delta2 - phi2 - v)
        else:
            k = 1
            while f(a - k * _TAU) < 0:
                k += 1
            B = a - k * _TAU

        # 伊利诺伊算法求根
        fA = f(A)
        fB = f(B)
        while abs(B - A) > _EPSILON:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA /= 2.0
            B = C
            fB = fC
        return math.exp(A / 2.0)


class Glicko2League:
    """管理所有队伍的 Glicko2 评级"""
    def __init__(self):
        self.teams: Dict[str, Glicko2Team] = {}

    def add_team(self, team_id: str, rating: float = 1500.0, rd: float = 350.0, vol: float = 0.06):
        self.teams[team_id] = Glicko2Team(rating, rd, vol)

    def process_event(self, team_id: str, event_type: str) -> None:
        """
        根据事件类型增加球队的 RD (不确定度)。
        事件类型映射表：
            starting_pitcher_change: 先发投手变更
            core_injury: 核心球员伤病
            bullpen_overuse: 牛棚过劳
            playoff: 季后赛强度变化
            trade: 球队交易
        """
        inflation_factors = {
            'starting_pitcher_change': 1.8,
            'core_injury': 1.5,
            'bullpen_overuse': 1.3,
            'playoff': 1.2,
            'trade': 1.7
        }
        if team_id not in self.teams:
            # 若队伍不存在，创建默认队伍
            self.add_team(team_id)
        team = self.teams[team_id]
        factor = inflation_factors.get(event_type, 1.0)
        team.rd = min(350.0, team.rd * factor)

    def get_rating_diff(self, team_a: str, team_b: str) -> tuple:
        """
        返回 (rating_a - rating_b, 总不确定度 sqrt(rd_a^2 + rd_b^2))
        若任一方不存在，返回 (0, 350)
        """
        if team_a not in self.teams or team_b not in self.teams:
            return 0.0, 350.0
        a = self.teams[team_a]
        b = self.teams[team_b]
        return a.rating - b.rating, math.sqrt(a.rd**2 + b.rd**2)

    def save(self, filepath: str) -> None:
        """保存所有队伍到 JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {}
        for tid, team in self.teams.items():
            data[tid] = {
                'mu': team.rating,
                'phi': team.rd,
                'sigma': team.vol
            }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Glicko2League':
        """从 JSON 加载队伍数据"""
        league = cls()
        if not os.path.exists(filepath):
            return league
        with open(filepath) as f:
            data = json.load(f)
        for tid, vals in data.items():
            league.teams[tid] = Glicko2Team(
                rating=vals.get('mu', 1500.0),
                rd=vals.get('phi', 350.0),
                vol=vals.get('sigma', 0.06)
            )
        return league


# 快速测试（可选）
if __name__ == '__main__':
    # 创建联赛
    league = Glicko2League()
    league.add_team('NYY', rating=1550, rd=200)
    league.add_team('BOS', rating=1500, rd=200)

    # 模拟比赛：NYY 胜
    nyy = league.teams['NYY']
    bos = league.teams['BOS']
    nyy.update(bos, 1.0)
    bos.update(nyy, 0.0)

    print(f"NYY: rating={nyy.rating:.1f}, rd={nyy.rd:.1f}")
    print(f"BOS: rating={bos.rating:.1f}, rd={bos.rd:.1f}")

    # 事件测试
    league.process_event('NYY', 'core_injury')
    print(f"事件后 NYY rd={league.teams['NYY'].rd:.1f}")
