class Configs:
    def __init__(
        self, 
        seq_len=720, 
        pred_len=96, 
        enc_in=7, 
        period_len=24, 
        train_epochs = 30,
        patience = 5,
        learning_rate = 0.02,
        dataset_name = 'unknown',
        model_type='linear',
        d_model=64
    ):
        """
        初始化配置参数
        :param seq_len: 输入序列长度
        :param pred_len: 预测序列长度
        :param enc_in: 特征维度数
        :param period_len: 周期长度
        :param d_model: MLP的隐藏层维度
        :param model_type: 模型类型（'linear' 或 'mlp'）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len
        self.train_epochs = train_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.d_model = d_model
        self.model_type = model_type
        self.dataset_name = dataset_name
        
    def __str__(self):
        return (
            f"Configs(seq_len={self.seq_len}, pred_len={self.pred_len}, "
            f"enc_in={self.enc_in}, period_len={self.period_len}, "
            f"train_epochs={self.train_epochs}, patience={self.patience}, flearning_rate={self.learning_rate}, "
            f"dataset_name={self.dataset_name},model_type='{self.model_type}')"
        )
    
    def __repr__(self):
        return self.__str__()
