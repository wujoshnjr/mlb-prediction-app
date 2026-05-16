"""
蒙地卡羅模擬引擎
用於模擬比賽得分分布，支援讓分盤、大小分盤分析
"""
import numpy as np
from scipy import stats

class MonteCarloSimulator:
    """
    基於球隊進攻/防守數據的比賽模擬器
    """
    def __init__(self, home_runs_avg, away_runs_avg, 
                 home_runs_std=2.2, away_runs_std=2.2,
                 n_simulations=10000):
        self.home_runs_avg = home_runs_avg
        self.away_runs_avg = away_runs_avg
        self.home_runs_std = home_runs_std
        self.away_runs_std = away_runs_std
        self.n_simulations = n_simulations
        self.results = None

    def simulate(self):
        """執行模擬，返回主客隊得分分布"""
        home_runs = np.random.normal(
            self.home_runs_avg, self.home_runs_std, self.n_simulations
        )
        away_runs = np.random.normal(
            self.away_runs_avg, self.away_runs_std, self.n_simulations
        )
        # 得分不能為負數
        home_runs = np.maximum(home_runs, 0)
        away_runs = np.maximum(away_runs, 0)
        self.results = {
            'home_runs': home_runs,
            'away_runs': away_runs,
            'total_runs': home_runs + away_runs,
            'run_diff': home_runs - away_runs
        }
        return self.results

    def moneyline_prob(self):
        """勝負盤：主勝機率"""
        if self.results is None:
            self.simulate()
        home_wins = np.sum(self.results['home_runs'] > self.results['away_runs'])
        return home_wins / self.n_simulations

    def spread_prob(self, spread):
        """
        讓分盤分析
        spread: 主隊讓分（例如 -1.5 表示主隊要贏2分以上才算過盤）
        返回：主隊過盤機率、客隊過盤機率
        """
        if self.results is None:
            self.simulate()
        # 主隊過盤：主隊得分 - 客隊得分 > 讓分絕對值
        home_cover = np.sum(
            self.results['run_diff'] > abs(spread)
        ) if spread < 0 else np.sum(
            self.results['run_diff'] < -abs(spread)
        )
        home_prob = home_cover / self.n_simulations
        return home_prob, 1 - home_prob

    def total_prob(self, total_line):
        """
        大小分盤分析
        total_line: 大小分線（例如 8.5）
        返回：大分機率、小分機率
        """
        if self.results is None:
            self.simulate()
        over = np.sum(self.results['total_runs'] > total_line)
        under = np.sum(self.results['total_runs'] < total_line)
        push = np.sum(self.results['total_runs'] == total_line)
        over_prob = over / self.n_simulations
        under_prob = under / self.n_simulations
        push_prob = push / self.n_simulations
        return over_prob, under_prob, push_prob

    def confidence_interval(self, stat='run_diff', confidence=0.80):
        """計算模擬結果的置信區間"""
        if self.results is None:
            self.simulate()
        data = self.results[stat]
        lower = np.percentile(data, (1 - confidence) / 2 * 100)
        upper = np.percentile(data, (1 + confidence) / 2 * 100)
        return lower, upper
