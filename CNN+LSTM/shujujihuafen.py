import pandas as pd

# 假设你的数据在 CSV 文件中
data_path = "E:/GCN+transformer_1/Data/data.csv"
df = pd.read_csv(data_path, header=None)

# 只取前 50 列（频带）
df_selected = df.iloc[:, :100]

# 查看结果
print(df_selected.shape)  # 应该是 (2880, 50)

# 如果需要保存处理后的数据
df_selected.to_csv("E:/GCN+transformer_1/Data/data_selected_100_columns.csv", index=False, header=False)