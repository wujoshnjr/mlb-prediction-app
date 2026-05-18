import numpy as np

class MonteCarloSimulator:
    def __init__(self, home_runs_avg, away_runs_avg, 
                 home_runs_std=2.2, away_runs_std=2.2, n_simulations=5000):
        self.home_runs_avg = home_runs_avg
        self.away_runs_avg = away_runs_avg
        self.home_runs_std = home_runs_std
        self.away_runs_std = away_runs_std
        self.n_simulations = n_simulations
        self.results = None

    def simulate(self):
        home_runs = np.random.normal(self.home_runs_avg, self.home_runs_std, self.n_simulations)
        away_runs = np.random.normal(self.away_runs_avg, self.away_runs_std, self.n_simulations)
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
        if self.results is None: self.simulate()
        home_cover = np.sum(self.results['run_diff'] > abs(spread)) if spread < 0 else np.sum(self.results['run_diff'] < -abs(spread))
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
