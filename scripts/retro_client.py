"""
Retrosheet 客户端（2026 賽季數據可能尚未發布）
"""
import requests
import pandas as pd
import zipfile
from io import BytesIO

def fetch_retrosheet(date_str: str = None, errors: list = None) -> pd.DataFrame:
    season = "2026"
    url = f"https://retrosheet.org/downloads/{season}batting.zip"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            csv_name = f"{season}batting.csv"
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, encoding='latin1')
        return df.head(500)
    except requests.exceptions.HTTPError as e:
        msg = f"Retrosheet {season} 赛季数据尚未发布 (HTTP {e.response.status_code})。通常赛季结束后才会有，目前跳过。"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
    except Exception as e:
        msg = f"Retrosheet 抓取异常: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
