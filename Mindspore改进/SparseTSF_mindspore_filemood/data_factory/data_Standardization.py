from data_factory import data_loader
from sklearn.preprocessing import StandardScaler

# 标准化 data
scaler = StandardScaler()
data = scaler.fit_transform(data_loader.raw_data)  # 标准化后的数据