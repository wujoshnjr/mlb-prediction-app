"""
伤病数据客户端（ESPN 免费来源，带 fallback）
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_injuries(date_str: str = None, errors: list = None) -> pd.DataFrame:
    """
    尝试从 ESPN MLB Injuries 页面抓取当前伤病名单。
    若失败则返回空 DataFrame，不影响主流程。
    """
    url = "https://www.espn.com/mlb/injuries"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # ESPN 伤病页面结构：表格包含球队和球员信息，class 可能变化，做一个简单的解析
        injury_list = []
        # 查找所有球队区块（以 class "Table__Title" 等为标志，这里仅提供通用解析思路）
        # 实际 ESPN 页面结构可能较复杂，以下为简化示例，若实际解析失败则返回空
        tables = soup.find_all("table", class_="Table")
        for table in tables:
            # 提取球队名（通常在 caption 或 thead 中）
            caption = table.find("caption")
            team_name = caption.get_text(strip=True) if caption else "Unknown"
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    player_name = cols[0].get_text(strip=True)
                    status = cols[1].get_text(strip=True)
                    injury_list.append({
                        "team_name": team_name,
                        "player_name": player_name,
                        "status": status
                    })
        # 如果解析不到，尝试更宽松的解析
        if not injury_list:
            # 备选：直接找所有 class 包含 "injuries" 的 div 等，此处略
            pass
        return pd.DataFrame(injury_list)
    except Exception as e:
        if errors is not None:
            errors.append(f"ESPN injury fetch error: {e}")
        return pd.DataFrame()
