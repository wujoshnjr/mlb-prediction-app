"""
Retrosheet 客户端
下载指定赛季的 CSV 数据（这里以击球数据为例）
"""
import requests
import pandas as pd
import zipfile
from io import BytesIO

def fetch_retrosheet(date_str: str = None) -> pd.DataFrame:
    """
    下载 Retrosheet 2025 赛季击球数据（batting.zip）
    返回前500行数据作为示例
    """
    season = "2025"
    url = f"https://retrosheet.org/downloads/{season}batting.zip"

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            # 文件名通常为 "2025batting.csv"
            csv_name = f"{season}batting.csv"
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, encoding='latin1')
        return df.head(500)
    except Exception as e:
        print(f"Retrosheet fetch error: {e}")
        return pd.DataFrame()
