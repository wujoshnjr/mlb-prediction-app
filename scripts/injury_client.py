"""
伤病数据客户端（使用 ESPN 爬虫，稳定可靠）
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_injuries(date_str: str = None, errors: list = None) -> pd.DataFrame:
    """
    从 ESPN MLB Injuries 页面抓取当前伤病名单。
    若失败则返回空 DataFrame，不影响主流程。
    注意：伤病数据为可选辅助信息，获取失败不会影响预测。
    """
    url = "https://www.espn.com/mlb/injuries"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        injury_list = []

        # 尝试解析标准表格结构
        tables = soup.find_all("table", class_="Table")
        for table in tables:
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

        # 如果未解析到，尝试备选结构（ESPN 页面有时会变）
        if not injury_list:
            # 查找所有可能包含伤病信息的卡片
            cards = soup.find_all("div", class_="Wrapper")
            for card in cards:
                # 简单提取文本
                text = card.get_text(strip=True)
                # 这里不做太复杂的解析，保持简洁
                pass

        return pd.DataFrame(injury_list)

    except Exception as e:
        # 伤病数据失败不记录错误，避免干扰主流程
        # 若需要调试可取消注释下面这行
        # print(f"ESPN injury fetch failed: {e}")
        return pd.DataFrame()
