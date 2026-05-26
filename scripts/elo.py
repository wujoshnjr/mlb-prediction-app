import json
import os

ELO_FILE = "data/elo_ratings.json"


class MLBElosystem:
    """File-backed Elo ratings for MLB teams.

    This version preserves the existing update behavior and adds get_rating(),
    which prediction.py already expects to call.
    """

    def __init__(self, k_factor=32, home_advantage=24):
        self.k = k_factor
        self.home_adv = home_advantage
        self.elos = self._load()

    def _load(self):
        if os.path.exists(ELO_FILE):
            try:
                with open(ELO_FILE, "r", encoding="utf-8") as file_obj:
                    data = json.load(file_obj)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print(f"Warning: unable to load Elo ratings: {exc}")
        return {}

    def save(self):
        os.makedirs("data", exist_ok=True)
        with open(ELO_FILE, "w", encoding="utf-8") as file_obj:
            json.dump(self.elos, file_obj, indent=2, ensure_ascii=False)

    def get_rating(self, team, default=1500.0):
        """Return one team's Elo without mutating the stored state."""
        try:
            return float(self.elos.get(team, default))
        except (TypeError, ValueError):
            return float(default)

    def expected_score(self, elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, home, away, home_win, date_str=None):
        for team in [home, away]:
            if team not in self.elos:
                self.elos[team] = 1500

        elo_home = float(self.elos[home]) + self.home_adv
        elo_away = float(self.elos[away])
        expected = self.expected_score(elo_home, elo_away)
        actual = 1 if home_win else 0

        self.elos[home] = float(self.elos[home]) + self.k * (actual - expected)
        self.elos[away] = float(self.elos[away]) - self.k * (actual - expected)
        self.save()
