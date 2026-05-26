"""Monte Carlo run simulation compatible with prediction.py and older callers."""

from __future__ import annotations

import numpy as np


class MonteCarloSimulator:
    def __init__(
        self,
        home_runs_avg=None,
        away_runs_avg=None,
        home_runs_std=2.2,
        away_runs_std=2.2,
        n_simulations=5000,
        random_seed=None,
    ):
        self.home_runs_avg = home_runs_avg
        self.away_runs_avg = away_runs_avg
        self.home_runs_std = float(home_runs_std)
        self.away_runs_std = float(away_runs_std)
        self.n_simulations = int(n_simulations)
        self.rng = np.random.default_rng(random_seed)
        self._last_result = None

    @staticmethod
    def _safe_era(value, default=4.40):
        try:
            parsed = float(value)
            if np.isfinite(parsed) and parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
        return default

    @staticmethod
    def _negative_binomial_params(mean, std):
        mean = max(0.05, float(mean))
        variance = max(float(std) ** 2, mean + 0.05)
        p_value = mean / variance
        p_value = min(0.999, max(0.001, p_value))
        n_value = mean * p_value / (1.0 - p_value)
        return n_value, p_value

    def _set_expected_runs(self, home_sp_era=None, away_sp_era=None, park_factor=1.0):
        if self.home_runs_avg is not None and self.away_runs_avg is not None:
            return

        league_mean = 4.40
        park = float(park_factor) if park_factor is not None else 1.0
        park = float(np.clip(park, 0.80, 1.25))
        home_allowed = self._safe_era(away_sp_era)
        away_allowed = self._safe_era(home_sp_era)

        self.home_runs_avg = float(np.clip(league_mean * park * (home_allowed / league_mean) * 1.03, 2.50, 7.00))
        self.away_runs_avg = float(np.clip(league_mean * park * (away_allowed / league_mean), 2.50, 7.00))

    def _draw_runs(self):
        home_n, home_p = self._negative_binomial_params(self.home_runs_avg, self.home_runs_std)
        away_n, away_p = self._negative_binomial_params(self.away_runs_avg, self.away_runs_std)
        home_runs = self.rng.negative_binomial(home_n, home_p, self.n_simulations)
        away_runs = self.rng.negative_binomial(away_n, away_p, self.n_simulations)
        return {
            "home_runs": home_runs,
            "away_runs": away_runs,
            "total_runs": home_runs + away_runs,
            "run_diff": home_runs - away_runs,
        }

    def simulate(
        self,
        home_team=None,
        away_team=None,
        home_sp_era=None,
        away_sp_era=None,
        park_factor=1.0,
        total_line=None,
        spread_line=None,
    ):
        """Return raw simulation arrays plus probabilities when market lines exist.

        No synthetic market line is invented. If total_line or spread_line is
        unavailable, its betting probability remains None.
        """
        self._set_expected_runs(home_sp_era, away_sp_era, park_factor)
        result = self._draw_runs()
        self._last_result = result

        over_prob = None
        home_cover_prob = None
        away_cover_prob = None

        if total_line is not None:
            try:
                line = float(total_line)
                over_prob = float(np.mean(result["total_runs"] > line))
            except (TypeError, ValueError):
                over_prob = None

        if spread_line is not None:
            try:
                home_spread = float(spread_line)
                home_cover_prob = float(np.mean(result["run_diff"] + home_spread > 0))
                away_cover_prob = float(np.mean(result["run_diff"] + home_spread < 0))
            except (TypeError, ValueError):
                home_cover_prob = None
                away_cover_prob = None

        result.update(
            {
                "over_prob": over_prob,
                "home_cover_prob": home_cover_prob,
                "away_cover_prob": away_cover_prob,
                "expected_home_runs": float(self.home_runs_avg),
                "expected_away_runs": float(self.away_runs_avg),
            }
        )
        return result

    def spread_prob(self, spread):
        if self._last_result is None:
            self._set_expected_runs()
            self._last_result = self._draw_runs()
        line = float(spread)
        home_cover = float(np.mean(self._last_result["run_diff"] + line > 0))
        away_cover = float(np.mean(self._last_result["run_diff"] + line < 0))
        return {"home_cover": home_cover, "away_cover": away_cover}

    def total_prob(self, total_line):
        if self._last_result is None:
            self._set_expected_runs()
            self._last_result = self._draw_runs()
        line = float(total_line)
        over_prob = float(np.mean(self._last_result["total_runs"] > line))
        return {"over": over_prob, "under": float(1.0 - over_prob)}
