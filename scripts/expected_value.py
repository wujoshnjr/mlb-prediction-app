import numpy as np

def calculate_ev(predicted_prob, odds, commission=0.05):
    if odds is None or odds <= 1:
        return 0, 0
    market_prob = 1 / (odds * (1 + commission))
    ev = predicted_prob * (odds - 1) - (1 - predicted_prob)
    edge = predicted_prob - market_prob
    return ev, edge

def filter_value_bets(predictions, min_ev=0.05, min_edge=0.03):
    value_bets = []
    for p in predictions:
        home_ev, home_edge = calculate_ev(p['predicted_home_win_pct'], p.get('home_odds'))
        away_odds = 1 / (1 - (1 / p['home_odds'])) if p.get('home_odds') and p['home_odds'] > 1 else None
        away_ev, away_edge = calculate_ev(p['predicted_away_win_pct'], away_odds)
        if home_ev > min_ev and home_edge > min_edge:
            value_bets.append({'game':f"{p['home_team']} vs {p['away_team']}",'type':'Home','ev':round(home_ev,4),'edge':round(home_edge,4)})
        if away_ev > min_ev and away_edge > min_edge:
            value_bets.append({'game':f"{p['home_team']} vs {p['away_team']}",'type':'Away','ev':round(away_ev,4),'edge':round(away_edge,4)})
    return value_bets
