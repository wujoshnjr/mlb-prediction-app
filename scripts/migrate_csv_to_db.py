"""
一次性将 CSV 迁移到 SQLite
"""
import pandas as pd
from scripts.database import init_database, insert_prediction

HISTORY_FILE = "data/historical_predictions.csv"

def migrate():
    if not os.path.exists(HISTORY_FILE):
        print("CSV 文件不存在，无需迁移")
        return
    init_database()
    df = pd.read_csv(HISTORY_FILE)
    for _, row in df.iterrows():
        data = row.to_dict()
        data['home_win'] = data.get('home_win', None)
        if data['home_win'] == '':
            data['home_win'] = None
        insert_prediction(data)
    print(f"迁移完成，共 {len(df)} 条记录")

if __name__ == "__main__":
    import os
    migrate()
