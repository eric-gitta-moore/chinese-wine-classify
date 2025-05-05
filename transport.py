# 把频率数据转成宽表

import pandas as pd

# 读取数据
df = pd.read_csv("插值+平均补齐到120Hz.csv")

# 将数据转换为宽表格式
wide_df = df.pivot(
    index=["wine_id", "wine_name", "类别", "类别编码", "sample_id"],
    columns="frquency",
    values="impedance",
)

# 重置索引，使所有列都成为普通列
wide_df = wide_df.reset_index()

# 保存结果
wide_df.to_csv("频率数据宽表.csv", index=False)

print("数据转换完成，已保存为'频率数据宽表.csv'")
