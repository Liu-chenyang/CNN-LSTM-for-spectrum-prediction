import pytorch_lightning as pl
import torch.nn as nn
import argparse
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import utils.metrics

from torch.utils.tensorboard import SummaryWriter
import os
import shutil

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss="mse",
            feat_max_val: float = 1.0,
            **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        # print(self.hparams)
        self.model = model
        self._loss = loss
        self.feat_max_val = feat_max_val
        # print(self.feat_max_val)
        log_dir = "tb_logs/my_model"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)  # 删除文件夹及其所有内容
        self.writer = SummaryWriter(log_dir="tb_logs/my_model")

    def forward(self, x):
        batch_size, _, num_nodes = x.size()
        predictions = self.model(x) # [64, 151, 1]
        # print(hidden.shape)
        return predictions


    def shared_step(self, batch, batch_idx):
        x, y = batch
        # x.shape [64, 64, 151] [样本数，时间步长，节点数]
        num_nodes = x.size(2)
        predictions = self(x) # [64, 151, 1]
        # print(predictions.shape)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes)) # [64, 151]
        y = y.reshape((-1, y.size(2))) # [64, 151]
        # print(predictions.shape,y.shape)
        return predictions, y

    def loss(self, predictions, true):
        if self._loss == "mse":
            return F.mse_loss(predictions, true)
        raise NameError("Loss not supported:", self._loss)


    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss) # 将train_loss储存在tb_logs/my_model下
        # 记录损失值到 TensorBoard
        self.writer.add_scalar('train_loss', loss.item(), self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        # print('在每个epoch后进行检验')
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val
        predictions = torch.abs(predictions)  # 取预测结果的绝对值。
        predictions = torch.round(predictions * 10) / 10  # 取小数点后一位。
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        # 记录验证损失到 TensorBoard
        self.writer.add_scalar('val_loss', loss.item(), self.current_epoch)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        mape = utils.metrics.MAPE(y, predictions)
        mae = utils.metrics.MAE(y, predictions)
        rmse = utils.metrics.RMSE(y, predictions)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
            "MAPE": mape,
        }
        self.log_dict(metrics)
        self.writer.add_scalar('RMSE', rmse.item(), self.current_epoch)
        self.writer.add_scalar('accuracy', accuracy.item(), self.current_epoch)
        self.writer.add_scalar('R2', r2.item(), self.current_epoch)
        return predictions, y # 这里返回的predictions和y是传给了on_validation_batch_end中的outputs

    # 关闭 writer 在训练结束时
    def on_train_end(self):
        self.writer.close()

    # # 这段代码定义了一个 configure_optimizers 方法，用于设置优化器和学习率调度器。
    # def configure_optimizers(self):
    #     print('设置优化器和学习率调度器')
    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.hparams.learning_rate,
    #         weight_decay=self.hparams.weight_decay,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=5e-6),
    #             "monitor": "train_loss",
    #         },
    #     }
    def configure_optimizers(self):
        print('设置优化器和学习率调度器')
        # 使用 AdamW 优化器
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  # 从超参数中获取学习率
            weight_decay=self.hparams.weight_decay  # 从超参数中获取权重衰减
        )

        # 在 configure_optimizers 中直接获取 num_training_steps
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.1 * num_training_steps)
        print(f"总训练步数: {num_training_steps}, 预热步数: {num_warmup_steps}")

        # 设置学习率调度器
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            ),
            "interval": "step",
            "frequency": 1
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }


    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=5e-5)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1e-4)
        parser.add_argument("--loss", type=str, default="mse")
        return parser