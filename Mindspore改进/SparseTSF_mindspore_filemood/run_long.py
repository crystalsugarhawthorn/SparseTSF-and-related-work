import os
import time
import mindspore
import numpy as np
from mindspore import Tensor
import matplotlib.pyplot as plt
from EarlyStopping import EarlyStopping

class SparseTSFModelRun:
    def __init__(self, model, loss_fn, optimizer=None, grad_fn=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = grad_fn

    def _train_one_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss

    def _train_one_epoch(self, train_dataset):
        self.model.set_train(True)
        for data, label in train_dataset.create_tuple_iterator():
            self._train_one_step(data, label)

    def evaluate(self, dataset):
        self.model.set_train(False)
        ls_pred, ls_label = [], []
        for data, label in dataset.create_tuple_iterator():
            pred = self.model(data)
            ls_pred += list(pred[:, :, -1].asnumpy())
            ls_label += list(label[:, :, -1].asnumpy())
        
        preds_tensor = Tensor(ls_pred)
        labels_tensor = Tensor(ls_label)
        loss = self.loss_fn(preds_tensor, labels_tensor)
        return loss.asnumpy(), np.array(ls_pred), np.array(ls_label)

    def train(self, train_dataset, vali_dataset, test_dataset, max_epoch_num):
        min_loss = np.finfo(np.float32).max
        steps_per_epoch = train_dataset.get_dataset_size()
        print(f"每个 epoch 的 step 数量: {steps_per_epoch}")
        # 对应文件夹名
        folder = f'{self.model.dataset_name}_{self.model.seq_len}_{self.model.pred_len}_{self.model.period_len}_{self.model.enc_in}_{self.model.model_type}'
        ckpt_path = f'checkpoints/{folder}/checkpoint.ckpt'
        # 如无创建
        os.makedirs(f'checkpoints/{folder}', exist_ok=True)
        # 配置早停
        early_stopping = EarlyStopping(patience=self.model.patience, verbose=True)
        print('>>>>>>>>>开始训练......')

        for epoch in range(1, max_epoch_num + 1):
            start_time = time.time()

            print(f'第{epoch}/{max_epoch_num}轮')
            print(f'当前学习率调整为：{self.optimizer.get_lr()}')
            self._train_one_epoch(train_dataset)

            train_loss, _, _ = self.evaluate(train_dataset)
            val_loss, _, _ = self.evaluate(vali_dataset)
            test_loss, preds, labels = self.evaluate(test_dataset)

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f'训练集损失: {train_loss}, 验证集损失: {val_loss}, 测试集损失: {test_loss}, 用时: {epoch_time:.2f}s')

            if val_loss < min_loss:
                mindspore.save_checkpoint(self.model, ckpt_path)
                min_loss = val_loss

            # 输出模型训练图表
            if epoch % 10 == 1 or epoch == max_epoch_num:
                for data0, label0 in test_dataset.create_tuple_iterator():
                    inputs = data0[:, :, -1].asnumpy()
                    gt = np.concatenate((inputs[0, :], labels[0, :]), axis=0)
                    pd = np.concatenate((inputs[0, :], preds[0, :]), axis=0)
                    break
                self._plot_results(gt, pd, epoch,'train', folder)

            # 判断早停
            early_stopping(val_loss, self.model, ckpt_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('训练完成！')
        self.test(test_dataset, ckpt_path, folder)

    def test(self, test_dataset, ckpt_path, img_folder):
        print('>>>>>>>>>开始测试......')
        mindspore.load_checkpoint(ckpt_path, net=self.model)
        loss, preds, labels = self.evaluate(test_dataset)
        print(f'测试集损失: {loss}')
        for data0, label0 in test_dataset.create_tuple_iterator():
            inputs = data0[:, :, -1].asnumpy()
            gt = np.concatenate((inputs[0, :], labels[0, :]), axis=0)
            pd = np.concatenate((inputs[0, :], preds[0, :]), axis=0)
            break
        self._plot_results(gt, pd, 0,'test', img_folder)

    def _plot_results(self, groundtruth, prediction, identifier, runtype, folder):
        os.makedirs(f"{runtype}_result", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(groundtruth, label='Ground Truth', color='blue', linewidth=2)
        plt.plot(prediction, label='Prediction', color='orange', linewidth=2)
        plt.title(f'Prediction vs Ground Truth - {identifier}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()

        pdf_path = f"{runtype}_result/{folder}/result_{identifier}.pdf"
        # 提取目标文件的目录路径
        folder_path = os.path.dirname(pdf_path)
        # 创建目标文件夹（如果不存在）
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(pdf_path)
        plt.close()
        print(f'结果图已保存到 {pdf_path}')
