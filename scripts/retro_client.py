import requests
import pandas as pd
import zipfile
from io import BytesIO

def fetch_retrosheet(date_str: str = None, errors: list = None) -> pd.DataFrame:
    season = "2026"  # 当前赛季
    url = f"https://retrosheet.org/downloads/{season}batting.zip"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()  # 如果不是 2XX 状态码会抛出异常
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            csv_name = f"{season}batting.csv"
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, encoding='latin1')
        return df.head(500)
    except requests.exceptions.HTTPError as e:
        # HTTP 错误（如 404）单独处理
        msg = f"Retrosheet {season} data not yet available (HTTP {e.response.status_code}). Will retry when season ends."
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
    except Exception as e:
        msg = f"Retrosheet fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
