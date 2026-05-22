"""
ELO动量特征：计算球队短期和长期的ELO变化量
"""
import json
import os
from datetime import datetime, timedelta

ELO_FILE = "data/elo_ratings.json"
ELO_HISTORY_FILE = "data/elo_history.json"  # 保存每日ELO快照

def save_elo_snapshot():
    """保存当前ELO快照到历史文件，用于计算动量"""
    if not os.path.exists(ELO_FILE):
        return
    with open(ELO_FILE, 'r') as f:
        current_elo = json.load(f)
    
    history = {}
    if os.path.exists(ELO_HISTORY_FILE):
        with open(ELO_HISTORY_FILE, 'r') as f:
            history = json.load(f)
    
    today = datetime.now().strftime('%Y-%m-%d')
    history[today] = current_elo
    
    # 只保留最近60天
    keys = sorted(history.keys())
    if len(keys) > 60:
        for old_key in keys[:-60]:
            del history[old_key]
    
    os.makedirs("data", exist_ok=True)
    with open(ELO_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_elo_momentum(team, days=7):
    """计算球队近N天的ELO变化量"""
    if not os.path.exists(ELO_HISTORY_FILE):
        return 0.0
    
    with open(ELO_HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    today = datetime.now().strftime('%Y-%m-%d')
    target_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 查找最接近目标日期的快照
    dates = sorted(history.keys())
    current_elo = history.get(today, {}).get(team, 1500)
    
    # 找到最接近target_date的日期
    closest_date = None
    for d in dates:
        if d <= target_date:
            closest_date = d
        else:
            break
    
    if closest_date:
        past_elo = history[closest_date].get(team, 1500)
    else:
        past_elo = current_elo
    
    return current_elo - past_elo
