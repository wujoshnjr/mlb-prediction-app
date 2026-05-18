"""
伤病数据客户端（暂不可用 - MLB Stats API 不提供公开伤病端点）
"""
import pandas as pd

def fetch_injuries(date_str: str = None, errors: list = None) -> pd.DataFrame:
    # 暂时返回空 DataFrame，避免 404 错误
    return pd.DataFrame()
