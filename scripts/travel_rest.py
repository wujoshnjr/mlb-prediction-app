TEAM_TIMEZONES = {
    "Braves": "Eastern", "Orioles": "Eastern", "Red Sox": "Eastern",
    "Cubs": "Central", "White Sox": "Central", "Reds": "Eastern",
    "Guardians": "Eastern", "Rockies": "Mountain", "Tigers": "Eastern",
    "Astros": "Central", "Royals": "Central", "Angels": "Pacific",
    "Dodgers": "Pacific", "Marlins": "Eastern", "Brewers": "Central",
    "Twins": "Central", "Mets": "Eastern", "Yankees": "Eastern",
    "Athletics": "Pacific", "Phillies": "Eastern", "Pirates": "Eastern",
    "Padres": "Pacific", "Giants": "Pacific", "Mariners": "Pacific",
    "Cardinals": "Central", "Rays": "Eastern", "Rangers": "Central",
    "Blue Jays": "Eastern", "Nationals": "Eastern", "D-backs": "Mountain"
}

def calculate_rest_days(schedule_df, last_game_dict=None):
    """
    根据赛程计算每场比赛两队的休息天数
    schedule_df: 包含 home_team, away_team, game_date
    last_game_dict: {team_name: last_game_date_str}，如果没有则默认休息3天
    """
    if last_game_dict is None:
        last_game_dict = {}
    from datetime import datetime, timedelta
    rest_home = []
    rest_away = []
    for _, game in schedule_df.iterrows():
        home = game.get('home_team')
        away = game.get('away_team')
        game_date = game.get('game_date')
        if game_date:
            try:
                game_dt = datetime.strptime(game_date[:10], '%Y-%m-%d')
            except:
                game_dt = datetime.now()
        else:
            game_dt = datetime.now()

        # 主队休息天数
        if home in last_game_dict:
            last = datetime.strptime(last_game_dict[home][:10], '%Y-%m-%d')
            rest = (game_dt - last).days - 1
            rest = max(0, min(rest, 5))  # 限制在0-5天
        else:
            rest = 2  # 默认
        rest_home.append(rest)

        # 客队休息天数
        if away in last_game_dict:
            last = datetime.strptime(last_game_dict[away][:10], '%Y-%m-%d')
            rest = (game_dt - last).days - 1
            rest = max(0, min(rest, 5))
        else:
            rest = 2
        rest_away.append(rest)

    schedule_df = schedule_df.copy()
    schedule_df['rest_home'] = rest_home
    schedule_df['rest_away'] = rest_away
    schedule_df['rest_diff'] = schedule_df['rest_home'] - schedule_df['rest_away']
    return schedule_df
