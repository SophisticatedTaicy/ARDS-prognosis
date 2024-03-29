import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import regression_parameter

if __name__ == '__main__':
    # 打乱数据
    dataframe = shuffle(pd.read_csv('../../ARDS/eicu/result/fill_with_0.csv', sep=',', low_memory=False))
    # 解决RuntimeWarning: invalid value encountered in true_divide
    np.seterr(divide='ignore', invalid='ignore')
    # 取特征数据和标签
    ards = dataframe.iloc[:, 1:-5].values
    label = dataframe.iloc[:, -5].values
    # 多分类转换为二分类
    label_new = []
    for item in label:
        # 长期住院
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    # 划分训练集和测试集，测试集占20%
    x_train, x_test, y_train, y_test = train_test_split(ards, label_new, test_size=0.2, random_state=123)
    # 数据归一化
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)
    # 模型训练
    model = regression_parameter.GBDTreg
    model.fit(x_train, y_train)
    # 模型预测
    # predict二分类预测，最终预测结果为0或者1的离散值
    y_test_predict = model.predict(x_test)
    y_train_predict = model.predict(x_train)
    # predict_proba回归预测，最终预测结果为0到1之间的连续数值
    # print('y_test is : ' + str(y_test) + ' y_test_pred is : ' + str(y_test_predict))
    # 模型真阳率和假阳率计算
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_test_predict)
    mse = mean_squared_error(y_test, y_test_predict)
    # 模型效果展示
    print("The mean squared error (MSE) on test set: {:.4f}" + str(mse))
    # accuracy_score函数
    # 分类正确率分数，函数返回一个分数，这个分数或是正确的比例，或是正确的个数。
    # 函数原型
    # accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    # print('The train accuracy of the GBDT is:', metrics.accuracy_score(y_train, y_train_predict))
    # print('The test accuracy of the GBDT is:', metrics.accuracy_score(y_test, y_test_predict))
    # 计算模型的auc
    roc_auc = metrics.auc(fpr, tpr)
    # 绘制模型ROC曲线
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='test (area = %0.4f)' % (roc_auc))
    # 绘制对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=0.8)
    # 设置横纵坐标
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # 设置横纵坐标名称
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # 设置图标名称
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
