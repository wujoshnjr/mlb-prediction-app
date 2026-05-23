# scripts/check_player_data.py
import os
import pandas as pd

hist_dir = "data/historical"
if not os.path.exists(hist_dir):
    print("无历史数据目录")
    exit(1)

files = [f for f in os.listdir(hist_dir) if f.endswith(".parquet")]
if not files:
    print("无 parquet 文件")
    exit(1)

sample_file = os.path.join(hist_dir, files[0])
df = pd.read_parquet(sample_file)
print(f"检查文件: {files[0]}")
print(f"行数: {len(df)}, 列数: {len(df.columns)}")
print("所有列名:")
for col in df.columns:
    print(f"  {col}")

key_columns = ['batter', 'pitcher', 'batter_id', 'pitcher_id', 'player_name', 'pitch_type']
found = [c for c in key_columns if c in df.columns]
missing = [c for c in key_columns if c not in df.columns]
print(f"\n找到的关键列: {found}")
print(f"缺失的关键列: {missing}")

if found:
    print("\n✅ 数据可用，可以构建球员级特征")
    print(df[found].head(2).to_string())
else:
    print("\n❌ 数据缺少球员标识，无法直接构建球员特征")
    print("需要调整 savant_client 抓取逻辑，确保包含 batter/pitcher 列")
