import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv("cleaned_wine_data.csv")

# 创建一个新的列来存储样品ID
df["sample_id"] = 0

# 初始化变量
current_sample_id = 0
current_freq = None
prev_freq = None

# 遍历每一行数据
for idx, row in df.iterrows():
    current_freq = row["frquency"]

    # 如果是第一行或者频率不是单调递增，创建新的样品ID
    if prev_freq is None or current_freq <= prev_freq:
        current_sample_id += 1

    # 分配样品ID
    df.at[idx, "sample_id"] = current_sample_id
    prev_freq = current_freq

# 保存处理后的数据
df.to_csv("按连续频率把样本分组到sample.csv", index=False)

print(f"处理完成！共创建了 {current_sample_id} 个样品ID。")
