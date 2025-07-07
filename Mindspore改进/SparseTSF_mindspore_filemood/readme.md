# SparseTSF_mindspore

## 简介

这个项目是基于mindspore框架实现的SparseTSF模型，用于时间序列预测任务。SparseTSF（Sparse Temporal Feature learning for Time Series Forecasting）是一种针对时间序列数据的高效预测模型，通过稀疏特征学习来提升预测的准确性和效率。

## 安装以及设备要求

### 安装指南

1. 安装Python和相关工具包：确保你的系统中已安装Python（建议版本3.7及以上）。然后，使用pip安装以下必要的Python库：
   - numpy：用于进行科学计算。
   - matplotlib：用于绘制图表和可视化数据。
   - pandas：用于数据处理和分析。
   - scikit-learn：用于机器学习和数据挖掘。
   可以运用命令行进行安装：pip install numpy matplotlib pandas scikit-learn
2. 安装mindspore：根据你的操作系统和硬件环境，从MindSpore的官方网站下载并安装适合的版本。安装完成后，确保环境变量配置正确。

### 设备要求

本项目支持在具有GPU或CPU的设备上运行。确保你的设备满足以下要求：
- CPU：支持AVX指令集的Intel或AMD处理器。
- 操作系统：Windows。

## 使用方法

运行main.py文件，即可开始训练和测试SparseTSF模型。训练过程中，模型会在每个epoch结束时输出训练和验证的损失值。训练完成后，模型会保存到指定的路径。

## 数据集

参数修改：在data_loader.py中修改数据集路径，以及数据集的参数。
具体修改数据：
    - enc_in: 特征数
    - period_len: 周期长度
    - batch_size：批处理大小，用于一次训练所使用的样本数量
    - seq_len: 输入序列长度
    - pred_len: 预测序列长度
    - learning_rate: 学习率
    - epoch: 训练轮数
    - data_path: 数据集路径
    - dataset_name: 数据集名称