import mindspore
import numpy as np

def generate_learning_rate_tensor(learning_rate, max_epoch, steps_per_epoch):
    # 使用列表生成学习率序列，每个epoch生成steps_per_epoch个相同的学习率
    lr_list = []
    for epoch in range(1, max_epoch + 1):
        if epoch < 3:
            lr = learning_rate
        else:
            lr = learning_rate * (0.8 ** (epoch - 3))
        # 为当前epoch生成steps_per_epoch个学习率
        lr_list.extend([lr] * steps_per_epoch)
    
    # 转换为 MindSpore Tensor
    lr_tensor = mindspore.Tensor(np.array(lr_list), mindspore.float32)
    return lr_tensor