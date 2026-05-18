class MLBElosystem:
    def __init__(self, k_factor=32, home_advantage=24):
        self.elos = {}
        self.k = k_factor
        self.home_adv = home_advantage

    def expected_score(self, elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, home, away, home_win, date_str):
        for team in [home, away]:
            if team not in self.elos:
                self.elos[team] = 1500
        elo_home = self.elos[home] + self.home_adv
        elo_away = self.elos[away]
        expected = self.expected_score(elo_home, elo_away)
        actual = 1 if home_win else 0
        self.elos[home] += self.k * (actual - expected)
        self.elos[away] -= self.k * (actual - expected)
