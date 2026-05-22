import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.elo import MLBElosystem

HISTORY_FILE = "data/historical_predictions.csv"

def update_elo_from_results():
    if not os.path.exists(HISTORY_FILE):
        print("历史预测文件不存在")
        return

    elo = MLBElosystem()
    updated_count = 0
    skipped_count = 0

    with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 安全获取字段，处理 None 值
            home = (row.get("home_team") or "").strip()
            away = (row.get("away_team") or "").strip()
            home_win_str = (row.get("home_win") or "").strip()

            if not home or not away or not home_win_str:
                skipped_count += 1
                continue

            # 安全转换为主队胜负
            try:
                home_win = int(float(home_win_str) + 0.5)
                if home_win not in (0, 1):
                    skipped_count += 1
                    continue
            except (ValueError, TypeError):
                print(f"跳过无效 home_win: {home_win_str} (比赛 {home} vs {away})")
                skipped_count += 1
                continue

            game_date = (row.get("game_date") or "").strip()
            elo.update(home, away, home_win, game_date)
            updated_count += 1

    elo.save()
    print(f"ELO 更新完成：处理 {updated_count} 场，跳过 {skipped_count} 场")
    if updated_count > 0:
        top5 = sorted(elo.elos.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"当前 ELO 前5: {dict(top5)}")

if __name__ == "__main__":
    update_elo_from_results()
