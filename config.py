# config.py
# ========== 全局功能开关 ==========

# 评级系统：'elo' 或 'glicko2'
RATINGS_ENGINE = 'glicko2'   # 切换为 glicko2 后，所有评级差值特征会自动改变

# 模型元学习器：'lr' 或 'elasticnet'
MODEL_META = 'elasticnet'

# 是否在第一层集成 MLP 神经网络
MODEL_USE_MLP = True

# 是否使用严格的 Walk-Forward 回测（扩张窗口+禁闭期）
WALKFORWARD_STRICT = True

# 是否使用打者-投手球种对位特征
FEATURE_USE_PITCH_MATCHUP = True

# NRFI 是否使用独立机器学习模型（否则回退手工算法）
NRFI_USE_ML = True

# 是否使用盘口时间曲线特征（趋势/波动率/逆转次数）
ODDS_USE_CURVE_FEATURES = True
