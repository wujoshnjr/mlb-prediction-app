"""
伤病数据客户端（使用 injury-report-monitor 库，失败则用 ESPN 爬虫）
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_injuries(date_str: str = None, errors: list = None) -> pd.DataFrame:
    """
    尝试使用 injury-report-monitor 获取 MLB 伤病名单。
    若不可用或出错，则回退到 ESPN 爬虫。
    """
    try:
        from injury_report_monitor import InjuryMonitor
        monitor = InjuryMonitor()
        report = monitor.get_report(sport="mlb")
        # 解析 report 为 DataFrame
        injury_list = []
        for team_data in report.get("teams", []):
            team_name = team_data.get("team", "Unknown")
            for player in team_data.get("players", []):
                injury_list.append({
                    "team_name": team_name,
                    "player_name": player.get("name", ""),
                    "status": player.get("status", "")
                })
        if injury_list:
            return pd.DataFrame(injury_list)
    except Exception as e:
        if errors is not None:
            errors.append(f"injury-report-monitor 失败: {e}，回退到 ESPN")

    # Fallback：ESPN 爬虫
    url = "https://www.espn.com/mlb/injuries"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        injury_list = []
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
        return pd.DataFrame(injury_list)
    except Exception as e:
        if errors is not None:
            errors.append(f"ESPN injury fetch error: {e}")
        return pd.DataFrame()
