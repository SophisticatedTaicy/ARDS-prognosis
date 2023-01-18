import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch as torch
import torch.nn as nn
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed
from sklearn.model_selection import train_test_split

# 步数
step_times = 60
# 步长
batch_size = 100
# 迭代次数
epoch = 200
# 特征数
input_size = 95
# 隐藏层
cell_size = 128

if __name__ == '__main__':
    data = pd.read_csv('../ARDS/eicu/result/time-series_60.csv', sep=',', encoding='utf-8')
    x = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])
    # 样本数，特征数，以及步数
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    # 不固定batch_size，预测时可以以1条记录进行分析
    # 将后续的网络层合并组装为一个完整网络
    model = Sequential()
    # units 定义隐藏层神经元个数
    # activation：激活函数使用relu（activation = 'relu'）
    # input_shape：输入维度，首层时，应指定该值（或batch_input_shape，配合使用stateful = True参数)，不限定batch
    model.add(LSTM(units=cell_size, activation='relu', return_sequences=True, input_shape=(step_times, input_size)))
    # Dropout通过使其他隐藏单元存在不可靠性来防止过拟合
    model.add(Dropout(0.2))
    model.add(LSTM(units=cell_size, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=cell_size, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    # Dense全连接层
    model.add(TimeDistributed(Dense(1)))
    # loss 损失函数，optimizer优化器
    model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
    k = x.shape[0] % batch_size
    x_train, x_test, y_train, y_test = (x[k:, :], x[:, k:], y[k:], y[:k])
    model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size)
    model.predict(x_test)
