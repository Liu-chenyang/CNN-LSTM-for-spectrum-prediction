import pytorch_lightning as pl
import utils.data.functions
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader




# LightningDataModule 是 PyTorch Lightning 提供的标准数据模块类，用来封装数据处理的逻辑。
# 包括：数据的下载、准备、清理等。训练集、验证集、测试集的数据加载逻辑。数据的预处理（例如归一化、标准化、数据增强等）。以及每个阶段的数据加载器的创建。
class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
            self,
            feat_path: str,
            adj_path: str,
            batch_size: int,
            seq_len: int,
            pre_len: int,
            split_ratio,
            normalize: bool,
            **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        # print(self._feat)
        # print(self._feat.shape)
        self._feat_max_val = np.max(self._feat)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)


    # Python 中的一个装饰器，用于定义 静态方法。静态方法是属于类的，但它不依赖于类实例，也不访问实例属性或实例方法。
    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--seq_len", type=int, default=32)
        parser.add_argument("--pre_len", type=int, default=1)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        print('数据集划分')
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        # DataLoader 是 PyTorch 中的一个类，专门用于将数据集打包为可迭代的批次，用于模型训练和验证。
        print('数据集打包为可迭代的批次，用于模型训练')
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        print("将数据集用于验证")
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

