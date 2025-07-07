import numpy as np

def generateData(data, seq_len, pred_len, enc_in):#定义generateData函数
    point_num = data.shape[0] #时间点总数
    sample_num = point_num-seq_len-pred_len+1 #生成的总样本数
    X = np.zeros((sample_num, seq_len, enc_in)) #用于保存输入数据
    Y = np.zeros((sample_num, pred_len, enc_in)) #用于保存对应的输出数据
    for i in range(sample_num): #通过遍历逐一生成输入数据和对应的输出数据
        X[i] = data[i:i+seq_len] #前seq_len个时间点数据组成输入数据
        Y[i] = data[i+seq_len:i+seq_len+pred_len]#后pred_len个时间点数据组成输出数据
    return X, Y #返回所生成的模型的输入数据X和输出数据Y


# sub_data = data[:1000,:] # 数据集太大的切割

def splitData(X, Y): #定义splitData函数
    N = X.shape[0] #样本总数
    train_X,train_Y=X[:int(N*0.6)],Y[:int(N*0.6)] #前60%的数据作为训练集
    vali_X,vali_Y=X[int(N*0.6):int(N*0.8)],Y[int(N*0.6):int(N*0.8)] #中间20%的数据作为验证集
    test_X,test_Y=X[int(N*0.8):],Y[int(N*0.8):] #最后20%的数据作为测试集
    return train_X,train_Y, vali_X,vali_Y, test_X,test_Y#返回划分好的数据集




