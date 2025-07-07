import gc
import os
import sys
import mindspore
import matplotlib
import mindspore.nn as nn
from config import Configs
from SparseTSF import SparseTSF
import data_factory.data_loader as dl
from run_long import SparseTSFModelRun
import data_factory.data_Standardization as ds
from learnrate_adjustment import generate_learning_rate_tensor
from data_factory.data_MSdataset import generateMindsporeDataset
from data_factory.data_transformer import splitData, generateDataInChunks

matplotlib.use('Agg')  # 使用非 GUI 后端
sys.stdout.reconfigure(encoding='utf-8')

def main():
    sys.path.append(os.path.abspath('path/to/data_factory'))

    for i in range(len(dl.pred_len)):
        # 使用内存映射的数据文件
        X_i, Y_i = generateDataInChunks(ds.data, dl.seq_len, dl.pred_len[i], dl.enc_in)  # 生成任务1所用的数据集
        print(f'任务{i + 1}数据集输入数据形状：{X_i.shape}，输出数据形状：{Y_i.shape}')
        
        # 使用生成器分批次生成数据
        train_X_i, train_Y_i, vali_X_i, vali_Y_i, test_X_i, test_Y_i = splitData(X_i, Y_i)
        print(f'任务{i + 1}训练集样本数：{train_X_i.shape[0]}，验证集样本数：{vali_X_i.shape[0]}，测试集样本数：{test_X_i.shape[0]}')  # 输出任务i训练集、验证集和测试集的样本数
        
        print(f'-----------当前预测长度为{dl.pred_len[i]}')

        # 生成对应mindspore数据集
        train_dataset_t = generateMindsporeDataset(train_X_i, train_Y_i, dl.batch_size)
        vali_dataset_t = generateMindsporeDataset(vali_X_i, vali_Y_i, dl.batch_size)
        test_dataset_t = generateMindsporeDataset(test_X_i, test_Y_i, dl.batch_size)

        # 删除不再使用的变量并强制垃圾回收
        del X_i, Y_i, train_X_i, train_Y_i, vali_X_i, vali_Y_i, test_X_i, test_Y_i
        gc.collect()

        configs = Configs(dl.seq_len, dl.pred_len[i], dl.enc_in, dl.period_len, dl.train_epochs, dl.patience, dl.learning_rate, dl.dataset_name, dl.model_type)
        model = SparseTSF(configs)  # 创建模型对象
        loss_fn = nn.MAELoss()  # 定义损失函数
        
        # 使用Adam优化器，并且使用type3学习率调整策略
        steps_per_epoch = train_dataset_t.get_dataset_size()
        learning_rate_tensor = generate_learning_rate_tensor(configs.learning_rate, configs.train_epochs, steps_per_epoch)
        optimizer = nn.Adam(model.trainable_params(), learning_rate_tensor) 
        
        def forward_fn(data, label):  # 定义前向计算的forward_fn函数
            pred = model(data)  # 使用SparseTSF模型进行预测
            loss = loss_fn(pred[:, :, :], label[:, :, :])  # 根据损失函数计算损失值
            return loss, pred  # 返回损失值和预测结果
        
        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)  # 获取用于计算梯度的函数
        model_run = SparseTSFModelRun(model, loss_fn, optimizer, grad_fn)  # 创建MODEL_RUN类对象model_run
        
        # 调用model_run.train方法完成训练
        model_run.train(train_dataset=train_dataset_t, vali_dataset=vali_dataset_t, test_dataset=test_dataset_t, max_epoch_num=configs.train_epochs)

if __name__ == '__main__':
    main()