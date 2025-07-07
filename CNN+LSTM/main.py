import argparse     #argparse：用于从命令行传递参数
import utils.data
import models
import tasks
import pytorch_lightning as pl
import utils.logging
import traceback    #traceback：用于捕获和打印异常堆栈信息
from pytorch_lightning.utilities import rank_zero_info
import utils.callbacks
import os
import shutil
# from pytorch_lightning.loggers import TensorBoardLogger


DATA_PATHS = {
    "my": {"feat": "E:/CNN+LSTM/Data/data_selected_50_columns.csv",
           "adj": "E:/CNN+LSTM/Data/adj_matrix_50.csv"},
}


def get_model(args, dm):
    model = None
    if args.model_name == "CNNLSTM":
        model = models.CNNLSTM(
            input_dim=1,
            cnn_channels=args.cnn_channels,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_layers=args.lstm_layers,
            num_nodes=1
        )
    return model

def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm._feat_max_val, **vars(args)
    )
    return task

def get_callbacks(args):
    # 通过这个回调函数，可以确保在训练过程中自动保存最优模型，这样即使中途训练中断，也可以从保存的模型检查点继续训练，或在验证时使用最优的模型。
    # 使用 ModelCheckpoint 回调时，PyTorch Lightning 会默认创建一个名为 checkpoints 的子文件夹，并将模型检查点文件（例如 .ckpt 文件）保存在这个文件夹内。
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min",dirpath="tb_logs/my_model",filename="best_model")
    # pl.callbacks.EarlyStopping：这是一个用于提前停止训练的回调函数。如果模型的性能在一段时间内不再提升，则训练会提前停止，防止过拟合或浪费计算资源。
    early_stop = pl.callbacks.EarlyStopping(
        monitor='train_loss',
        min_delta=0.0,
        patience=64,
        verbose=False,
        mode='min'
    )
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
        early_stop,
    ]
    return callbacks


def main_supervised(args):
    # args.batch_size = 128
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    print(dm.batch_size)
    # print(dm.seq_len)
    # print(dm._adj)
    model = get_model(args, dm)
    # print(f"Model: {model}")
    task = get_task(args, model, dm)
    # print(f"Task: {task}")
    # 获取 callbacks 和 checkpoint_callback
    callbacks = get_callbacks(args)
    # print(callbacks)

    # # 清空日志文件夹
    # log_dir = "tb_logs/my_model"
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)  # 删除文件夹及其所有内容
    # # 初始化 TensorBoardLogger
    # logger = TensorBoardLogger("tb_logs", name="my_model")

    # 这段代码是用来创建一个 PyTorch Lightning 的 Trainer 实例的，Trainer 是 PyTorch Lightning 中用于管理模型训练和验证的核心组件。
    # 它封装了训练、验证、测试等流程，简化了深度学习模型的训练。
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,  # 替换 gpus 为 accelerator
        precision=args.precision,
        # accumulate_grad_batches=2,  # 累积2个小批次后再更新梯度
        callbacks=callbacks,  # callbacks 是训练期间触发的回调函数列表。它们允许在特定事件发生时执行一些操作（如保存最佳模型、早停等）。
        logger=False,  # 添加 TensorBoard logger
        log_every_n_steps=10,  # 表示每多少个训练批次记录一次日志。
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,  # 每个 epoch 都进行验证
        enable_checkpointing=True # 启用 checkpoint 保存和恢复模型
    )
    print("Starting training...")
    trainer.fit(task, dm)
    print("Training completed.")
    results = trainer.validate(datamodule=dm, ckpt_path="best")
    print(results)


def main(args):
    # print(vars(args))
    # vars(args) 会将 args 对象转换为字典，将其所有属性和值存储在一个 Python 字典中。这使得你可以更方便地访问所有的命令行参数和值。
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)  # 等效于results = main_supervised(args)
    return results

if __name__ == "__main__":
    # parser = argparse.ArgumentParser() 是 Python 标准库 argparse 模块中的一行代码，用于创建一个命令行参数解析器
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_epochs", type=int, default=2000, help="Number of epochs to train")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPU or TPU) to use")
    parser.add_argument("--precision", type=int, default=16, help="Precision level (16 or 32)")
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="Type of accelerator to use: 'gpu', 'cpu', 'tpu'")
    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("my"), default="my"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("CNNLSTM"),
        default="CNNLSTM",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised"),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the output console log file")
    #  这行代码的作用是解析命令行参数并将其存储在 temp_args 变量中。
    temp_args, _ = parser.parse_known_args()
    # getattr(): 这是一个内置函数，用于获取对象的属性或方法。
    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    #  这一行代码的作用是解析命令行参数并将其存储在 args 变量中。
    args = parser.parse_args()
    # print(args)

    if args.log_path is not None:
        # 这行代码执行了一个日志管理操作，将 PyTorch Lightning 的日志输出到一个指定的文件中。它的作用是将日志记录（通常是训练过程中的信息、错误或警告）重定向到一个指定的文件路径。
        utils.logging.output_logger_to_file(pl._logger, args.log_path)
    try:
        # max_epochs=100, devices=1, precision=32, accelerator='auto', data='my', model_name='GCNTransformer', settings='supervised',
        # log_path='./log', batch_size=64, seq_len=64, pre_len=1, split_ratio=0.8, normalize=True, hidden_dim=64, num_heads=4, num_layers=2,
        # learning_rate=0.001, weight_decay=0.0001, loss='mse'
        results = main(args)
    except:
        traceback.print_exc()

