import numpy as np #导入numpy工具包

data_path = 'dataset/ETTh1.csv' #数据文件路径
dataset_name = 'ETTh1'
enc_in = 7 #特征数
seq_len = 720
pred_len = [96,192,336,720] #预测的时间步数
period_len = 48 # 周期
train_epochs = 30
patience = 5
batch_size = 256
learning_rate = 0.02
model_type = 'linear'
with open(data_path, 'r', encoding='utf-8') as f:
    raw_data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=range(1, enc_in + 1), dtype=np.float32)
print('数据形状：{0}，元素类型：{1}'.format(raw_data.shape, raw_data.dtype))