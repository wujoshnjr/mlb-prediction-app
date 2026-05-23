# scripts/walkforward.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def generate_walkforward_splits(df, date_col='date', gap_days=7, train_min_days=180, test_days=1):
    """
    生成滚动时间序列索引，确保训练集截止于测试开始前 gap_days 天。
    :param df: DataFrame，需包含 date 列
    :param date_col: 日期列名
    :param gap_days: 训练集与测试集之间的禁闭天数
    :param train_min_days: 最小训练数据天数
    :param test_days: 每个测试窗口的长度（天数）
    :yield: (train_indices, test_indices)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dates = sorted(df[date_col].unique())
    # 至少需要 train_min_days 天的历史数据
    start_idx = max(train_min_days, 1)
    for i in range(start_idx, len(dates) - test_days + 1):
        train_end_date = dates[i] - pd.Timedelta(days=gap_days)
        test_start_date = dates[i]
        test_end_date = dates[i + test_days - 1]
        train_mask = df[date_col] <= train_end_date
        test_mask = (df[date_col] >= test_start_date) & (df[date_col] <= test_end_date)
        train_idx = df.index[train_mask].tolist()
        test_idx = df.index[test_mask].tolist()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def walkforward_train_evaluate(
    X, y, dates, model_builder, gap_days=7, test_days=1, train_min_days=180,
    optuna_trials=20, use_optuna=True
):
    """
    严格 Walk-Forward 训练 + 预测。
    :param X: 特征 DataFrame
    :param y: 目标 Series
    :param dates: 日期 Series (与 X、y 对齐)
    :param model_builder: 可调用对象，返回一个未训练的 sklearn 风格模型实例
    :param gap_days: 禁闭天数
    :param test_days: 每次预测未来几天（通常为 1）
    :param train_min_days: 最少训练天数
    :param optuna_trials: Optuna 优化次数（如果 use_optuna=True）
    :param use_optuna: 是否在每个折内做超参数搜索
    :return: (y_true_all, y_pred_all, dates_all)
    """
    y_true_all, y_pred_all, dates_all = [], [], []
    df = pd.DataFrame({'date': dates})
    splits = generate_walkforward_splits(df, date_col='date', gap_days=gap_days,
                                         train_min_days=train_min_days, test_days=test_days)

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if use_optuna:
            # 此处可以集成 Optuna 搜索，为简化演示直接使用默认参数
            # 实际使用时可以替换 model_builder 为带超参的函数
            model = model_builder()
            # 这里省略 optuna 集成，只展示结构，如需完整 Optuna 请告知
        else:
            model = model_builder()

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        y_true_all.extend(y_test.values)
        y_pred_all.extend(preds)
        dates_all.extend(dates.iloc[test_idx].values)

    return np.array(y_true_all), np.array(y_pred_all), dates_all
