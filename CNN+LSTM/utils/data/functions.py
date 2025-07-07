import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat

def load_adjacency_matrix(adj_path, dtype=np.float32):
    # 加载邻接矩阵并跳过第一行
    adj_df = pd.read_csv(adj_path, header=None, skiprows=1)  # 跳过第一行
    # print(adj_df)
    adj = np.array(adj_df, dtype=dtype)
    # print(adj)
    return adj

def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
        # print(data)
        train_size = int(time_len * split_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:time_len]
        train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
        for i in range(len(train_data) - seq_len - pre_len):
            train_X.append(np.array(train_data[i: i + seq_len]))
            train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
        for i in range(len(test_data) - seq_len - pre_len):
            test_X.append(np.array(test_data[i: i + seq_len]))
            test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))
        return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)



def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    # TensorDataset 是 PyTorch 提供的一个简单的数据集封装类，它将数据和标签（或目标）打包在一起，使得它们可以很方便地配合 DataLoader 使用，用于训练和验证。
    # 每次从 TensorDataset 中取出一条数据时，会返回对应的输入数据和目标数据。
    # torch.FloatTensor 是 PyTorch 中的数据类型之一，它将 NumPy 数组或其他数据转换为 PyTorch 张量（Tensor）。
    # PyTorch 模型的输入通常需要以张量的形式传递，因此需要将数据转换为 torch.Tensor。
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    # print('train_dataset:',train_dataset)
    return train_dataset, test_dataset