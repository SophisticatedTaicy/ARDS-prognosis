# 数据不均衡问题处理·
'''
数据处理：少样本数据过采样SMOTE或者多样本数据欠采样
算法处理：代价敏感学习
'''

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import NeighbourhoodCleaningRule as ncr, RandomUnderSampler, NeighbourhoodCleaningRule
import ml.classification.classify_parameter


def random_sample(x, y):
    # 随机过采样
    ros = RandomOverSampler(random_state=42)
    x_oversampled, y_oversampled = ros.fit_resample(x, y)
    return x_oversampled, y_oversampled


def SMOTE_sample(x, y):
    # 过采样的经典算法（解决传统过采样算法引起的过拟合问题）
    smote = SMOTE(random_state=0)
    x_smotesample, y_smotesample = smote.fit_resample(x, y)
    return x_smotesample, y_smotesample


def random_under_sampler(x, y):
    rus = RandomUnderSampler(random_state=42)
    x_rus, y_rus = rus.fit_resample(x, y)
    return x_rus, y_rus


def ADASYN_sample(x, y):
    adasyn = ADASYN(random_state=42)
    x_ada, y_ada = adasyn.fit_resample(x, y)
    return x_ada, y_ada


# 混合采样
def SMOTEENN_sample(x, y):
    return None


def SMOTETomek_sample(x, y):
    return None


# 清洗
def NCR_sample(x, y):
    sampler = NeighbourhoodCleaningRule()
    x_sam, y_sam = sampler.fit_resample(x, y)
    return x_sam, y_sam


if __name__ == '__main__':
    dataframe = pd.read_csv('ARDS/eicu/result/0801_fill_with_0.csv', sep=',')
    data = dataframe.iloc[:, 1:-5]
    label = dataframe.iloc[:, -5]
    # data, label = ADASYN_sample(data, label)
    data, label = NCR_sample(data, label)
    data, label = SMOTE_sample(data, label)
    label_new = []
    for item in label:
        if item == 2:
            label_new.append(1)
        else:
            label_new.append(0)
    x_train, x_test, y_train, y_test = train_test_split(data, label_new, test_size=0.2, shuffle=True)
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)
    model = ml.classification.classify_parameter.XGB
    model.fit(x_train, y_train)
    y_predict = model.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_predict[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r', label='XGBoost Val auc=%.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    plt.legend(loc='lower right', fontsize=7)
    plt.show()
