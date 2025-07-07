from mindspore import nn, ops

class SparseTSF(nn.Cell):  # 定义 SparseTSF 类
    def __init__(self, configs):
        super(SparseTSF, self).__init__()
        print(configs)
        # 获取参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.patience = configs.patience
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        self.dataset_name = configs.dataset_name
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 1D卷积层，用于时间序列局部特征提取
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            pad_mode='pad',
            padding=self.period_len // 2,
            has_bias=False
        )

        # 稀疏预测层，支持linear和mlp两种类型
        if self.model_type == 'linear':
            self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.SequentialCell(
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Dense(self.d_model, self.seg_num_y)
            )

    def construct(self, x):
        batch_size = x.shape[0]

        # 归一化并调整维度：b, s, c -> b, c, s
        seq_mean = ops.mean(x, 1, keep_dims=True)
        # seq_std = ops.std(x,1,0,keepdims=True)
        
        # print(x.shape, seq_mean.shape, seq_std.shape)
        #x = ops.permute((x - seq_mean)/seq_std, (0,2,1))
        x = ops.permute((x - seq_mean), (0,2,1))
        # 1D卷积聚合局部特征
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        # 降采样：b, c, s -> bc, n, w -> bc, w, n
        x = ops.permute(x.reshape(-1, self.seg_num_x, self.period_len), (0, 2, 1))
        # 稀疏预测
        if self.model_type == 'linear':
            y = self.linear(x)  # bc, w, m
        elif self.model_type == 'mlp':
            y = self.mlp(x)
        # 上采样：bc, w, m -> bc, m, w -> b, c, s
        y = ops.permute(y,(0, 2, 1)).reshape(batch_size, self.enc_in, self.pred_len)
        
        # 调整维度并还原均值
        # y = ops.permute(y,(0, 2, 1)) * seq_std + seq_mean
        y = ops.permute(y,(0, 2, 1))  + seq_mean
        # print("----mark")
        return y