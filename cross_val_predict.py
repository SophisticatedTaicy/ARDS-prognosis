import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score, RepeatedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml.classification import classify_parameter


# https://blog.csdn.net/weixin_39747087/article/details/110550606
def n_times_k_fold_cross_val(k, n, model, data, label):
    '''
    :param n: 进行交叉检验的次数
    :param k: 进行交叉检验的折数
    :param model: 模型
    :param data: 数据
    :param label: 标签
    :return:
    '''
    scale = MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)
    # 数据归一化
    x_train = scale.fit_transform(x_train)
    x_test = scale.fit_transform(x_test)
    # 训练集上进行模型训练
    model.fit(x_train, y_train)
    kf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=42)
    # n次k折交叉检验预测值
    test_predicted = cross_val_predict(model, x_test, y_test, cv=5)
    # 评分
    test_score = cross_val_score(model, x_test, y_test, cv=kf)
    auc_std = np.std(test_score)
    auc_mean = np.mean(test_score)
    # 测试集上10次五折交叉检验结果
    auc = metrics.roc_auc_score(y_test, test_predicted)
    print('test accuracy : ' + str(auc) + ' auc std : ' + str(auc_std) + ' auc mean : ' + str(auc_mean))


if __name__ == '__main__':
    dataframe = pd.read_csv('ARDS/eicu/result/0801_fill_with_0.csv', sep=',', encoding='utf-8')
    data = np.array(dataframe.iloc[:, 1:-5])
    label = np.array(dataframe.iloc[:, -5])
    label_new = []
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    models = classify_parameter.models
    names = filter.param.names
    for name, model in zip(names, models):
        print('model : ' + str(name))
        n_times_k_fold_cross_val(10, 5, model, data, label_new)
