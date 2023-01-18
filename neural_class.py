#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/11/11 18:33
# @Author : hlx
# @File : neural_class.py
# @Software: PyCharm

# 神经网络的搭建--分类任务 #
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 激励函数都在这

import os
import pandas as pd
from matplotlib import pyplot as plt
import torch.utils.data as data
from ml.classification.classify_parameter import *

from sklearn.metrics import roc_auc_score


# 读取csv文件
def read_file(filename, path, sep=',', encoding='utf-8'):
    path = os.path.join(path, filename + '.csv')
    return pd.read_csv(path, sep=sep, encoding=encoding)


# 建立神经网络
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_feature)  # 隐藏层线性输出
        self.out = torch.nn.Linear(n_feature, n_output)  # 输出层线性输出
        self.dropout = torch.nn.Dropout(0.3)
        self.hidden2 = torch.nn.Linear(n_feature, n_feature)
        self.w = torch.nn.Parameter(torch.ones(1, n_feature))
        self.BN = torch.nn.BatchNorm1d(n_feature)
        torch.nn.init.normal_(self.w, mean=0.0, std=1.0)
        self.hidden_a = torch.nn.Linear(n_feature, n_feature)

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        # x = torch.nn.functional.normalize(x, 1)
        # pdb.set_trace()
        x = x * (0.5 + F.softmax(self.hidden_a(x)))

        #    x = self.dropout(self.BN(F.tanh(self.hidden(x))))  # 激励函数(隐藏层的线性值)

        x = F.tanh(self.hidden2(x)) + x
        x = self.out(x)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x


def standard_data_by_percentile_w(data, data_t):
    """
    @type 数据归一化处理，方法：使用80分位与20分位数的差对数列进行归一化
    @param data: 输入需要规格化的Dataframe数据
    @return 归一化后的数据
    """
    mean = torch.mean(data, 0).view(1, -1)
    std = torch.std(data, 0).view(1, -1) + 1e-9
    # pdb.set_trace()
    new_data = (data - mean) / std
    new_data_t = (data_t - mean) / std
    return new_data, new_data_t


if __name__ == '__main__':
    # x0，x1是数据，y0,y1是标签
    path = 'D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS\combine\csvfiles'
    test_x_name = 'x_test'
    test_y_name = 'y_test'
    train_x_name = 'x_train'
    train_y_name = 'y_train'
    x_combine_train_name = 'x_combine_train'
    y_combine_train_name = 'y_combine_train'
    test_x = np.array(read_file(filename=test_x_name, path=path))
    test_y = np.array(read_file(filename=test_y_name, path=path))
    train_x = np.array(read_file(filename=train_x_name, path=path))
    train_y = np.array(read_file(filename=train_y_name, path=path))
    x_combine_train = np.array(read_file(filename=x_combine_train_name, path=path))
    y_combine_train = np.array(read_file(filename=y_combine_train_name, path=path))
    net = Net(n_feature=127, n_hidden=256, n_output=2)  # 几个类别就几个 output
    # 训练网络
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()
    # 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
    x = torch.tensor(train_x, dtype=torch.float32)  # FloatTensor = 32-bit floating
    y = torch.tensor(train_y).view(-1)  # LongTensor = 64-bit integer
    # pdb.set_trace()
    x_c = torch.tensor(x_combine_train, dtype=torch.float32)  # FloatTensor = 32-bit floating
    y_c = torch.tensor(y_combine_train).view(-1)  # LongTensor = 64-bit integer
    x_te = torch.tensor(test_x, dtype=torch.float32)  # FloatTensor = 32-bit floating
    y_te = torch.tensor(test_y).view(-1)  # LongTensor = 64-bit integer
    # x, x_te = standard_data_by_percentile_w(x, x_te)
    torch_dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=20000,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=2  # 多少线程来读取数据
    )
    x_c, x_te = standard_data_by_percentile_w(x_c, x_te)
    torch_dataset_C = data.TensorDataset(x_c, y_c)

    loader_c = data.DataLoader(
        dataset=torch_dataset_C,
        batch_size=14000,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=2  # 多少线程来读取数据
    )

    for epoch in range(100):  # 对整套数据训练3次
        for step, (batch_x, batch_y) in enumerate(loader_c):
            net.train()
            out = net(batch_x.detach())  # 喂给 net 训练数据 x, 输出分析值
            # pdb.set_trace()
            out_f = F.softmax(out, 1)

            batch_y = batch_y.detach()
            loss_1 = (- batch_y * torch.log(out_f[torch.arange(batch_x.size(0)), batch_y])).mean()
            loss_2 = (- (1 - batch_y) * torch.log(out[torch.arange(batch_x.size(0)), 1 - batch_y])).mean()
            loss = loss_func(out, batch_y)  # + loss_1 * 0.1   # 计算两者的误差

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 可视化展示
            if step % 10 == 0:
                net.eval()
                out = net(x_te)
                plt.cla()
                prediction = torch.max(F.softmax(out), 1)[1]
                # pdb.set_trace()
                pred_y = prediction.data.numpy().squeeze()
                target_y = y_te.data.numpy()
                accuracy = np.sum(pred_y == target_y) / y_te.size(0)  # 计算准确度
                print(epoch, step, accuracy, loss.item(), loss.item())

                prob_all = []
                lable_all = []
                out_f = F.softmax(out)
                # pdb.set_trace()
                for i in np.arange(x_te.size(0)):
                    # 表示模型的预测输出
                    prob_all.append(out_f.detach()[i, 1].numpy())
                    lable_all.append(y_te[i])
                print("AUC:{:.4f}".format(roc_auc_score(lable_all, prob_all)))
    plt.ioff()  # 停止画图
    plt.show()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    plt.show()
