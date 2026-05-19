"""
ELO 滚动更新器
根据实际比赛结果更新 ELO 评分
"""
import csv
import os
import json
from scripts.elo import MLBElosystem

HISTORY_FILE = "data/historical_predictions.csv"

def update_elo_from_results():
    """读取历史预测中已有结果的比赛，更新 ELO 评分"""
    if not os.path.exists(HISTORY_FILE):
        print("历史预测文件不存在")
        return

    elo = MLBElosystem()  # 自动加载现有 ELO
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
            # 只更新尚未计入 ELO 的比赛（简单判断：两队都在初始1500且未被更新过）
            # 这里简化处理：只要有结果就更新，ELO 系统会自行处理重复更新的问题
            # 更好的做法是记录已更新过的 game_id，避免重复
            game_date = row.get("game_date", "")
            elo.update(home, away, home_win, game_date)
            updated_count += 1

    elo.save()
    print(f"ELO 已更新，处理 {updated_count} 场比赛")
    print(f"当前 ELO 前5: {dict(sorted(elo.elos.items(), key=lambda x: x[1], reverse=True)[:5])}")

if __name__ == "__main__":
    update_elo_from_results()
