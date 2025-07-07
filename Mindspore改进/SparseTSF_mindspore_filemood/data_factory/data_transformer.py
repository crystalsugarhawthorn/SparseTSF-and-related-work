import numpy as np

def generateDataInChunks(data, seq_len, pred_len, enc_in, chunk_size=1000):  
    """
    按块生成数据，避免一次性加载过多数据。
    """
    point_num = data.shape[0]
    sample_num = point_num - seq_len - pred_len + 1

    # 创建内存映射文件
    X_path = 'X_data.dat'
    Y_path = 'Y_data.dat'
    X = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(sample_num, seq_len, enc_in))
    Y = np.memmap(Y_path, dtype=np.float32, mode='w+', shape=(sample_num, pred_len, enc_in))

    for start in range(0, sample_num, chunk_size):
        end = min(start + chunk_size, sample_num)
        for i in range(start, end):  
            X[i, :, :] = data[i:i + seq_len, :]
            Y[i, :, :] = data[i + seq_len:i + seq_len + pred_len, :]

    # 写入磁盘
    X.flush()
    Y.flush()
    
    return np.memmap(X_path, dtype=np.float32, mode='r', shape=(sample_num, seq_len, enc_in)), \
           np.memmap(Y_path, dtype=np.float32, mode='r', shape=(sample_num, pred_len, enc_in))

def splitData(X, Y):  # 定义splitData函数
    N = X.shape[0]  # 样本总数
    train_X, train_Y = X[:int(N * 0.6)], Y[:int(N * 0.6)]  # 前60%的数据作为训练集
    vali_X, vali_Y = X[int(N * 0.6):int(N * 0.8)], Y[int(N * 0.6):int(N * 0.8)]  # 中间20%的数据作为验证集
    test_X, test_Y = X[int(N * 0.8):], Y[int(N * 0.8):]  # 最后20%的数据作为测试集
    return train_X, train_Y, vali_X, vali_Y, test_X, test_Y  # 返回划分好的数据集
