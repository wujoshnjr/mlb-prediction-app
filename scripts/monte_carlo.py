"""
蒙特卡洛模拟引擎（Negative Binomial 分布版本）
更贴近真实棒球得分分布，解决正态分布的偏态和 heavy tail 问题
"""
import numpy as np
from scipy.stats import nbinom


class MonteCarloSimulator:
    def __init__(self, home_runs_avg, away_runs_avg, 
                 home_runs_std=2.2, away_runs_std=2.2, n_simulations=5000):
        """
        初始化模拟器
        
        参数:
            home_runs_avg: 主队场均得分
            away_runs_avg: 客队场均得分
            home_runs_std: 主队得分的标准差（用于反推负二项参数）
            away_runs_std: 客队得分的标准差
            n_simulations: 模拟次数
        """
        self.home_runs_avg = max(home_runs_avg, 0.5)  # 避免均值为0
        self.away_runs_avg = max(away_runs_avg, 0.5)
        self.home_runs_std = max(home_runs_std, 1.0)
        self.away_runs_std = max(away_runs_std, 1.0)
        self.n_simulations = n_simulations
        self.results = None

    def _get_nbinom_params(self, mu, sigma):
        """
        根据均值 mu 和标准差 sigma 计算负二项分布的参数 r 和 p
        利用公式: r = mu^2 / (sigma^2 - mu) , p = r / (r + mu)
        若 sigma^2 <= mu，则令 r = mu，即退化为泊松近似
        """
        variance = sigma ** 2
        if variance > mu:
            r = mu ** 2 / (variance - mu)
            p = r / (r + mu)
        else:
            # 当方差小于等于均值时，使用较大的 r 值使分布接近对称
            r = mu * 2.0
            p = r / (r + mu)
        return max(r, 0.1), min(max(p, 0.01), 0.99)  # 保证参数在合理范围

    def simulate(self):
        """执行模拟，返回包含得分和结果的字典"""
        r_h, p_h = self._get_nbinom_params(self.home_runs_avg, self.home_runs_std)
        r_a, p_a = self._get_nbinom_params(self.away_runs_avg, self.away_runs_std)

        home_runs = nbinom.rvs(r_h, p_h, size=self.n_simulations)
        away_runs = nbinom.rvs(r_a, p_a, size=self.n_simulations)

        # 得分不能为负（负二项分布本身非负，但以防万一）
        home_runs = np.maximum(home_runs, 0)
        away_runs = np.maximum(away_runs, 0)

        self.results = {
            'home_runs': home_runs,
            'away_runs': away_runs,
            'total_runs': home_runs + away_runs,
            'run_diff': home_runs - away_runs
        }
        return self.results

    def spread_prob(self, spread):
        """
        计算让分盘的过盘概率
        spread: 让分值，通常为负数表示主队让分（如 -1.5）
        返回 (主队过盘概率, 客队过盘概率)
        """
        if self.results is None:
            self.simulate()

        if spread < 0:
            # 主队让分，主队过盘条件：主队得分 - 客队得分 > 让分绝对值
            home_cover = np.sum(self.results['run_diff'] > abs(spread))
        else:
            # 客队让分，主队过盘条件：主队得分 - 客队得分 < -让分值
            home_cover = np.sum(self.results['run_diff'] < -abs(spread))
        
        home_prob = home_cover / self.n_simulations
        away_prob = 1 - home_prob
        return home_prob, away_prob

    def total_prob(self, total_line):
        """
        计算大小分盘的概率
        total_line: 大小分线
        返回 (大分概率, 小分概率, 平盘概率)
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
        """
        计算指定统计量的置信区间
        stat: 'home_runs', 'away_runs', 'total_runs', 'run_diff'
        confidence: 置信水平（如 0.80 表示 80% 置信区间）
        返回 (下界, 上界)
        """
        if self.results is None:
            self.simulate()

        data = self.results[stat]
        lower = np.percentile(data, (1 - confidence) / 2 * 100)
        upper = np.percentile(data, (1 + confidence) / 2 * 100)
        return lower, upper
