import csv
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.elo import MLBElosystem

HISTORY_FILE = "data/historical_predictions.csv"

def update_elo_from_results():
    if not os.path.exists(HISTORY_FILE):
        print("历史预测文件不存在")
        return

    elo = MLBElosystem()
    updated_count = 0

    with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = row.get("home_team", "").strip()
            away = row.get("away_team", "").strip()
            home_win_str = row.get("home_win", "").strip()
            if not home or not away or not home_win_str:
                continue
            home_win = int(home_win_str)
            game_date = row.get("game_date", "")
            elo.update(home, away, home_win, game_date)
            updated_count += 1

    elo.save()
    print(f"ELO 已更新，处理 {updated_count} 场比赛")
    print(f"当前 ELO 前5: {dict(sorted(elo.elos.items(), key=lambda x: x[1], reverse=True)[:5])}")

if __name__ == "__main__":
    update_elo_from_results()
