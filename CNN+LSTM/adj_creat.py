import pandas as pd

# Calculate adjacency matrix through correlation coefficient
df = pd.read_csv('E:/GCN+transformer_1/Data/data_selected_100_columns.csv',header=None)

print(df.shape)  # 确认有多少列

corr_matrix = df.corr(method='spearman',min_periods=1)

# 将相关系数大于 0.9 的元素标记为 True，小于等于 0.9 的元素标记为 False。
adj_matrix = (corr_matrix > 0.8).astype(int)

adj_matrix.to_csv('E:/GCN+transformer_1/Data/adj_matrix_100.csv', index=False)

print(adj_matrix)
print(adj_matrix.shape)