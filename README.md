## 🚀 项目简介

本项目基于 PyTorch 实现了一套端到端的 CNN+LSTM 模型，用于对无线频谱各信道的功率谱密度（PSD）时序进行预测。通过在时间序列上先用 1D-CNN 提取局部空间特征，再用 LSTM 捕捉长期时序依赖，为动态频谱接入（DSA）系统提供高精度、低延迟的频谱使用趋势预测。

## ✨ 主要功能

- **数据预处理**  
  - 读取 Electrosense 或自定义格式的频谱数据  
  - 滑窗切片、归一化、训练/验证/测试集划分  

- **模型构建**  
  - **1D-CNN 特征提取层**：多层卷积 + 池化，自动学习局部时域特征  
  - **LSTM 时序建模层**：捕捉序列中的长期依赖  
  - **全连接输出层**：支持单步与多步预测  

- **训练与评估**  
  - 可配置超参数（卷积核大小、隐藏单元数、学习率、batch_size 等）  
  - 自动保存最佳模型与断点续训  
  - 计算 RMSE、MAE、R² 等多项误差指标  

- **预测与可视化**  
  - 在线加载模型进行推断  
  - 绘制真实值 vs 预测值对比图  
