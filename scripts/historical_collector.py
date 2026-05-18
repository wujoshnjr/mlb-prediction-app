import os
import pandas as pd
from datetime import datetime, timedelta
import time

DATA_DIR = "data/historical"

def collect_date(date_str):
    os.makedirs(DATA_DIR, exist_ok=True)
    # 导入需修改为支持历史日期的版本，这里略过
    pass

def collect_range(start_date, end_date):
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Collecting {date_str}...")
        collect_date(date_str)
        current += timedelta(days=1)
        time.sleep(1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        collect_date(sys.argv[1])
    elif len(sys.argv) == 3:
        collect_range(sys.argv[1], sys.argv[2])
