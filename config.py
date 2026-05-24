# config.py
RATINGS_ENGINE = 'elo'                     # 暂时使用 ELO，Glicko2 需重建后切换
MODEL_META = 'lr'                          # 或 'elasticnet'
MODEL_USE_MLP = False                      # MLP 未真正接通，暂关
WALKFORWARD_STRICT = False                 # 严格回测暂未在 CI 启用
FEATURE_USE_PITCH_MATCHUP = False          # 数据未就绪
NRFI_USE_ML = False                        # 模型未训练，关闭
ODDS_USE_CURVE_FEATURES = False            # 快照不足
