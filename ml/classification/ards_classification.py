import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

import classify_parameter

# 解决RuntimeWarning: invalid value encountered in true_divide
np.seterr(divide='ignore', invalid='ignore')


# 可视化函数
def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['r', 'g', 'b', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)

    for idx, cc in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cc, 0],
                    y=x[y == cc, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cc)


def select_features(data, label):
    model = SelectFromModel(classify_parameter.LR_)
    model.fit(data, label)
    importance = model.estimator_.coef_
    index_0 = np.abs(importance[0]).argsort()
    index_1 = np.abs(importance[1]).argsort()
    index_2 = np.abs(importance[2]).argsort()
    print('index_0 : ' + str(index_0) + ' origin : ' + str(importance[0]))
    print('index_1 : ' + str(index_1) + ' origin : ' + str(importance[1]))
    print('index_2 : ' + str(index_2) + ' origin : ' + str(importance[2]))


if __name__ == '__main__':
    # 打乱数据
    dataframe = pd.read_csv('../ARDS/eicu/result/0801_fill_with_0.csv', sep=',', low_memory=False)
    # 取特征数据和标签
    # 使用所有特征
    ards = np.array(dataframe.iloc[:, 1:-5])
    label = np.array(dataframe.iloc[:, -5])
    # 多分类转换为二分类
    label_new = []
    for item in label:
        # 长期住院
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    scale = StandardScaler()
    # select_features(ards, label)
    # 划分训练集和测试集，测试集占20%
    models = classify_parameter.base_models
    names = classify_parameter.names
    colors = ['r', 'g', 'b', 'gray', 'cyan']
    for name, model in zip(names, models):
        aucs = []
        for i in range(0, 10):
            # 划分数据集
            x_train, x_test, y_train, y_test = train_test_split(ards, label_new, test_size=0.2, random_state=42,
                                                                shuffle=True)
            # 数据归一化
            x_train = scale.fit_transform(x_train)
            x_test = scale.fit_transform(x_test)
            # 模型训练
            model.fit(x_train, y_train)
            # 模型预测
            if name == 'Perceptron':
                test_predict_proba = model._predict_proba_lr(x_test)
                auc = metrics.roc_auc_score(y_test, test_predict_proba[:, 1])
                fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            elif name == 'BayesianRidge' or name == 'LinearRegression':
                test_predict_proba = model.predict(x_test)
                auc = metrics.roc_auc_score(y_test, test_predict_proba)
                fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba, pos_label=1)
            else:
                test_predict_proba = model.predict_proba(x_test)
                auc = metrics.roc_auc_score(y_test, test_predict_proba[:, 1])
                fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            # 计算模型的auc
            roc_auc = metrics.auc(fpr, tpr)

            aucs.append(auc)
        auc_mean = round(np.mean(aucs), 4)
        auc_std_mean = round(np.std(aucs), 3)
        print('model is : ' + str(name) + '  auc mean is : ' + str(auc_mean) + ' auc std is : ' + str(auc_std_mean))
        plt.plot()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # 增加到如下语句后，注意缩进
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.show()
