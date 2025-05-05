import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def process_data(df):
    # 获取所有可能的频率值（20-120Hz）
    all_frequencies = range(20, 121)

    # 按wine_id分组
    wine_groups = df.groupby("wine_id")

    # 存储处理后的数据
    processed_data = []

    for wine_id, wine_group in wine_groups:
        # 获取该wine_id下所有sample_id
        sample_ids = wine_group["sample_id"].unique()

        # 找出相对完整的sample_id（频率数量接近完整）
        complete_samples = []
        for sample_id in sample_ids:
            sample_data = wine_group[wine_group["sample_id"] == sample_id]
            frequencies = sample_data["frquency"].unique()
            # 如果频率数量接近完整（比如只缺失几个点），则进行插值
            if sample_data["frquency"].iloc[-1] == 120:  # 如果要有120Hz的数据就插值
                print(f"线性插值样本 {sample_id} 中")
                # 对数据进行排序
                sample_data = sample_data.sort_values("frquency")
                # 创建插值函数
                f = interp1d(
                    sample_data["frquency"],
                    sample_data["impedance"],
                    kind="linear",
                    fill_value="extrapolate",
                )
                # 生成所有频率点的阻抗值
                interpolated_impedance = f(all_frequencies)
                # 创建新的数据行
                for freq, imp in zip(all_frequencies, interpolated_impedance):
                    new_row = sample_data.iloc[0].copy()
                    new_row["frquency"] = freq
                    new_row["impedance"] = imp
                    processed_data.append(new_row)
                complete_samples.append(sample_id)
            else:
                # 对于不完整的数据，使用原始数据
                processed_data.extend(
                    [sample_data.iloc[i] for i in range(len(sample_data))]
                )

        if not complete_samples:
            print(f"警告：wine_id {wine_id} 没有相对完整的数据样本")
            continue

        # 计算完整样本的平均值
        complete_data = pd.DataFrame(processed_data)
        mean_values = complete_data.groupby("frquency")["impedance"].mean()

        # 处理每个sample_id
        for sample_id in sample_ids:
            if sample_id not in complete_samples:
                print(f"平均插值处理样本 {sample_id} 中")
                sample_data = wine_group[wine_group["sample_id"] == sample_id]
                frequencies = sample_data["frquency"].unique()

                # 如果数据不完整
                if len(frequencies) < len(all_frequencies):
                    missing_frequencies = set(all_frequencies) - set(frequencies)

                    # 为缺失的频率创建新行
                    for freq in missing_frequencies:
                        new_row = sample_data.iloc[0].copy()
                        new_row["frquency"] = freq
                        new_row["impedance"] = mean_values[freq]
                        processed_data.append(new_row)

                # # 添加原始数据
                # processed_data.extend(sample_data.to_dict('records'))
                # processed_data.extend([sample_data.iloc[i] for i in range(len(sample_data))])

    # 创建新的DataFrame
    result_df = pd.DataFrame(processed_data)

    # 按 wine_id, frequency, sample_id 排序
    result_df = result_df.sort_values(["wine_id", "sample_id", "frquency"])

    return result_df


# 读取数据
df = pd.read_csv("按连续频率把样本分组到sample.csv")

# 处理数据
processed_df = process_data(df)

# 保存处理后的数据
processed_df.to_csv("插值+平均补齐到120Hz.csv", index=False)
