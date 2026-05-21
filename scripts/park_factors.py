PARK_FACTORS = {
    "Chase Field": 1.05, "Truist Park": 1.02, "Oriole Park at Camden Yards": 1.08,
    "Fenway Park": 1.12, "Wrigley Field": 1.00, "Guaranteed Rate Field": 1.04,
    "Great American Ball Park": 1.10, "Progressive Field": 1.06, "Coors Field": 1.38,
    "Comerica Park": 0.96, "Minute Maid Park": 1.01, "Kauffman Stadium": 1.03,
    "Angel Stadium": 0.94, "Dodger Stadium": 0.92, "loanDepot park": 0.88,
    "American Family Field": 1.07, "Target Field": 0.98, "Citi Field": 0.95,
    "Yankee Stadium": 1.15, "Oakland Coliseum": 0.90, "Citizens Bank Park": 1.09,
    "PNC Park": 1.01, "Petco Park": 0.87, "T-Mobile Park": 0.93,
    "Busch Stadium": 0.97, "Tropicana Field": 0.91, "Globe Life Field": 1.05,
    "Rogers Centre": 1.06, "Nationals Park": 1.00, "Oracle Park": 0.89
}
def get_park_factor(stadium_name: str) -> float:
    return PARK_FACTORS.get(stadium_name, 1.0)
