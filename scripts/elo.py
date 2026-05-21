import json
import os

ELO_FILE = "data/elo_ratings.json"

class MLBElosystem:
    def __init__(self, k_factor=32, home_advantage=24):
        self.k = k_factor
        self.home_adv = home_advantage
        self.elos = self._load()

    def _load(self):
        if os.path.exists(ELO_FILE):
            try:
                with open(ELO_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save(self):
        os.makedirs("data", exist_ok=True)
        with open(ELO_FILE, 'w') as f:
            json.dump(self.elos, f, indent=2)

    def expected_score(self, elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, home, away, home_win, date_str=None):
        for team in [home, away]:
            if team not in self.elos:
                self.elos[team] = 1500
        elo_home = self.elos[home] + self.home_adv
        elo_away = self.elos[away]
        expected = self.expected_score(elo_home, elo_away)
        actual = 1 if home_win else 0
        self.elos[home] += self.k * (actual - expected)
        self.elos[away] -= self.k * (actual - expected)
        self.save()
