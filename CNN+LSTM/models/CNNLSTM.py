import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden_dim, lstm_layers, num_nodes):
        super(CNNLSTM, self).__init__()
        self.cnn_channels=cnn_channels
        self.num_nodes = num_nodes
        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1)
        # LSTM 部分
        self.lstm = nn.LSTM(cnn_channels, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True)
        # 输出层
        self.fc = nn.Linear(lstm_hidden_dim, num_nodes)


    def forward(self, X):
        # X: (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = X.size()
        # 将输入调整为 CNN 的输入格式
        X = X.permute(0, 2, 1)  # (batch_size, num_nodes, seq_len)
        X = X.reshape(batch_size * num_nodes, 1, seq_len)  # (batch_size * num_nodes, 1, seq_len)
        # 通过 CNN
        X = F.relu(self.conv1(X))  # (batch_size * num_nodes, cnn_channels, seq_len)
        # X = F.relu(self.conv2(X))  # (batch_size * num_nodes, cnn_channels, seq_len)
        # 恢复形状
        X = X.view(batch_size, num_nodes, self.cnn_channels, seq_len)  # (batch_size, num_nodes, cnn_channels, seq_len)
        # print(X.shape) # [64, 100, 64, 16]
        # 调整维度，准备输入 LSTM
        X = X.permute(0, 3, 1, 2)  # (batch_size, seq_len, num_nodes, cnn_channels)
        # 将 num_nodes 和时间步分开进行 LSTM 处理
        lstm_out = []
        for node in range(num_nodes):
            # 对每个节点的时间序列进行 LSTM 处理
            node_features = X[:, :, node, :]  # (batch_size, time_steps, cnn_channels)
            # print(node_features.shape) # [64, 16, 64]
            lstm_node_out, _ = self.lstm(node_features)  # (batch_size, time_steps, lstm_hidden_dim)
            # print(lstm_node_out.shape) # [64, 16, 128]
            lstm_out.append(lstm_node_out[:, -1, :].unsqueeze(1))  # 取最后一个时间步的输出, (batch_size, 1, lstm_hidden_dim)
        # 合并所有节点的 LSTM 输出
        lstm_out = torch.cat(lstm_out, dim=1)  # (batch_size, num_nodes, lstm_hidden_dim)
        # print(lstm_out.shape)
        # 对每个节点的特征进行预测.
        output = self.fc(lstm_out)  # (batch_size, num_nodes, output_dim) [64, 151, 1]
        # print(output.shape)
        # print(output)
        return output



    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--cnn_channels", type=int, default=16)
        parser.add_argument("--lstm_hidden_dim", type=int, default=32)
        parser.add_argument("--lstm_layers", type=int, default=1)
        return parser