import numpy as np #导入numpy库
from mindspore.dataset import GeneratorDataset #导入GeneratorDataset类

class MultiTimeSeriesDataset(): #定义MultiTimeSeriesDataset类
    def __init__(self, X, Y): #构造方法
        self.X, self.Y = X, Y #设置输入数据和输出数据
    def __len__(self):
        return len(self.X) #获取数据的长度
    def __getitem__(self, index):
        return self.X[index], self.Y[index] #根据索引值为index的数据

def generateMindsporeDataset(X, Y, batch_size): #定义generateMindsporeDataset函数
    dataset = MultiTimeSeriesDataset(X.astype(np.float32), Y.astype(np.float32)) #根据X和Y创建MultiTimeSeriesDataset类对象
    dataset = GeneratorDataset(dataset, column_names=['data','label']) #创建GeneratorDataset类对象，并指定数据集两列的列名称分别是data和label
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False) #将数据集分成多个批次，以支持批量训练
    return dataset #返回可用于模型训练和测试的数据集