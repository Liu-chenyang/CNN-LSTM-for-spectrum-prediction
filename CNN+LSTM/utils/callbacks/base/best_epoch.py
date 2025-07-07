from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import copy




# 这段代码定义了一个自定义的回调类 BestEpochCallback，继承自 PyTorch Lightning 的 Callback 类。
# BestEpochCallback 的主要功能是监控某个指标（如 train_loss 或 val_loss），并记录该指标在训练过程中表现最好的 epoch（轮次）。
# 它能够帮助你追踪训练过程中最优的模型状态。
class BestEpochCallback(Callback):
    # 定义了两个变量 TORCH_INF 和 torch_inf，它们都被赋值为 torch.tensor(np.Inf)。这意味着它们都是代表无穷大的 PyTorch 张量 (torch.Tensor)。
    TORCH_INF = torch_inf = torch.tensor(np.Inf)
    MODE_DICT = {
        "min": (torch_inf, "min"),
        "max": (-torch_inf, "max"),
    }
    # 如果你选择 "min" 模式，它会使用 torch.lt 来比较当前值是否小于最优值。
    # 如果你选择 "max" 模式，它会使用 torch.gt 来比较当前值是否大于最优值。
    MONITOR_OP_DICT = {"min": torch.lt, "max": torch.gt}

    def __init__(self, monitor="", mode="min"):
        super(BestEpochCallback, self).__init__()
        self.monitor = monitor
        self.__init_monitor_mode(monitor, mode)
        self.best_epoch = 0

    def __init_monitor_mode(self, monitor, mode):
        self.best_value, self.mode = self.MODE_DICT[mode]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch != 0:
            return
        monitor_op = self.MONITOR_OP_DICT[self.mode]
        # trainer.callback_metrics：这个是 PyTorch Lightning 中存储模型评估指标的字典，包含了当前 epoch 结束时的所有度量（例如验证集上的损失或准确率）。
        metrics_dict = copy.copy(trainer.callback_metrics)
        monitor_value = metrics_dict.get(self.monitor, self.best_value)
        if monitor_op(monitor_value.type_as(self.best_value), self.best_value):
            self.best_value = monitor_value
            self.best_epoch = trainer.current_epoch
