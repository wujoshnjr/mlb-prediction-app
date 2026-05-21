import numpy as np
from scipy.stats import nbinom

class MonteCarloSimulator:
    def __init__(self, home_runs_avg, away_runs_avg, home_runs_std=2.2, away_runs_std=2.2, n_simulations=5000):
        self.home_runs_avg = max(home_runs_avg, 0.5)
        self.away_runs_avg = max(away_runs_avg, 0.5)
        self.home_runs_std = max(home_runs_std, 1.0)
        self.away_runs_std = max(away_runs_std, 1.0)
        self.n_simulations = n_simulations
        self.results = None
    def _get_nbinom_params(self, mu, sigma):
        variance = sigma ** 2
        if variance > mu:
            r = mu ** 2 / (variance - mu)
            p = r / (r + mu)
        else:
            r = mu * 2.0
            p = r / (r + mu)
        return max(r, 0.1), min(max(p, 0.01), 0.99)
    def simulate(self):
        r_h, p_h = self._get_nbinom_params(self.home_runs_avg, self.home_runs_std)
        r_a, p_a = self._get_nbinom_params(self.away_runs_avg, self.away_runs_std)
        home_runs = nbinom.rvs(r_h, p_h, size=self.n_simulations)
        away_runs = nbinom.rvs(r_a, p_a, size=self.n_simulations)
        self.results = {
            'home_runs': np.maximum(home_runs, 0),
            'away_runs': np.maximum(away_runs, 0),
            'total_runs': home_runs + away_runs,
            'run_diff': home_runs - away_runs
        }
        return self.results
    def spread_prob(self, spread):
        if self.results is None: self.simulate()
        if spread < 0:
            home_cover = np.sum(self.results['run_diff'] > abs(spread))
        else:
            home_cover = np.sum(self.results['run_diff'] < -abs(spread))
        home_prob = home_cover / self.n_simulations
        return home_prob, 1 - home_prob
    def total_prob(self, total_line):
        if self.results is None: self.simulate()
        over = np.sum(self.results['total_runs'] > total_line)
        under = np.sum(self.results['total_runs'] < total_line)
        push = np.sum(self.results['total_runs'] == total_line)
        return over / self.n_simulations, under / self.n_simulations, push / self.n_simulations
    def confidence_interval(self, stat='run_diff', confidence=0.80):
        if self.results is None: self.simulate()
        data = self.results[stat]
        lower = np.percentile(data, (1 - confidence) / 2 * 100)
        upper = np.percentile(data, (1 + confidence) / 2 * 100)
        return lower, upper
        
