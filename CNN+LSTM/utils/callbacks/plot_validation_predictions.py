from utils.callbacks.base import BestEpochCallback
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

# 这段代码定义了一个用于在模型训练和验证过程中绘制验证集预测结果的回调类 PlotValidationPredictionsCallback。
class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []
        log_dir = "tb_logs/predictions"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)  # 删除文件夹及其所有内容
        self.writer = SummaryWriter(log_dir="tb_logs/predictions")  # 初始化 SummaryWriter

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    # 这是一个在每个验证批次结束后触发的回调函数。
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.current_epoch != self.best_epoch:
            return
        self.ground_truths.clear()
        self.predictions.clear()
        predictions, y = outputs
        # print('predictions.shape:', predictions.shape)
        # print('y.shape:', y.shape)
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y)
        self.predictions.append(predictions)


    # 只有当所有的训练和验证过程（包括多个 epoch）都完成后，on_fit_end 才会被调用。
    def on_fit_end(self, trainer, pl_module):
        ground_truth = np.concatenate(self.ground_truths, 0)
        # print(ground_truth[:, 1])
        predictions = np.concatenate(self.predictions, 0)
        # print(predictions[:, 1])
        # print(ground_truth.shape)
        # if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
        #     tensorboard = trainer.logger.experiment
        #     # 继续后续操作
        # else:
        #     print("TensorBoard logger is not initialized or not available.")  # 获取 TensorBoardLogger 的实验属性
        # 这段代码的作用是在模型训练结束后，将模型在验证集上的预测结果与真实值进行可视化，并将生成的图表记录到 TensorBoard 中进行分析和展示
        for node_idx in range(ground_truth.shape[1]):
            plt.clf() # 清除当前的绘图，以便开始新的图表。
            plt.rcParams["font.family"] = "Times New Roman" # 设置字体为 Times New Roman，以美化绘图的字体样式。
            fig = plt.figure(figsize=(7, 2), dpi=300)
            plt.plot(
                ground_truth[:, node_idx],
                color="yellow",
                linestyle="-",
                label="Ground truth",
                linewidth=0.5
            )
            plt.plot(
                predictions[:, node_idx],
                color="deepskyblue",
                linestyle="-",
                label="Predictions",
                linewidth=0.5
            )
            plt.legend(loc="best", fontsize=10)
            plt.xlabel("Time(45Min)")
            plt.ylabel("PSD(dB/Hz)")
            # 使用 SummaryWriter 手动添加图表
            self.writer.add_figure(
                "Prediction result of node " + str(node_idx),
                fig,
                global_step=trainer.current_epoch,
            )
        # 清除缓存并关闭 writer
        self.writer.flush()
        self.writer.close()

