#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/12/21 19:46
# @Author : hlx
# @File : model_selection.py
# @Software: PyCharm
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import auc, roc_curve
import os

from sklearn.model_selection import train_test_split

from filter.common import read_file, format_label, data_merge, standard_data_by_white
from filter.param import outcome_dict, colors

base_picture_path = './ARDS/combine/pictures/'
base_csv_path = './ARDS/combine/csvfiles'
base_path = os.path.dirname(os.path.abspath(__file__)) + '\ARDS'
from ARDS.data_process.process import Processing

# 在多个机器学习模型上融合训练数据，在统一测试集上查看模型性能
# 使用不同模型测试融合数据性能
def various_model(x_train, x_test, y_train, y_test):
    from ml.classification.classify_parameter import base_models, searchCVnames
    # 将训练集和测试集白化
    x_train, x_test = standard_data_by_white(x_train, x_test)
    for name, model, color in zip(searchCVnames, base_models, colors[:len(base_models)]):
        i = 0
        mean_tpr = []
        mean_fpr = np.linspace(0, 1, 1000)
        tprs = []
        while i < 10:
            model.fit(x_train, y_train)
            if name == 'Perceptron':
                test_predict_proba = model._predict_proba_lr(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            elif name == 'LinearRegression' or name == 'BayesianRidge':
                test_predict_proba = model.predict(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba, pos_label=1)
            else:
                test_predict_proba = model.predict_proba(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            tprs.append(interp(np.linspace(0, 1, 1000), fpr, tpr))
            mean_tpr = np.mean(tprs, axis=0)
            i += 1
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr[:-1:30], mean_tpr[:-1:30], label=r'%s (area=%.3f)' % (name, mean_auc), color=color,
                 marker='o', markersize=2, lw=1.5)
    plt.title(r'%s' % outcome, fontweight='bold', fontsize=15)
    plt.xlabel('False positive rate', fontsize=15, fontweight='bold')
    plt.ylabel('True positive rate', fontsize=15, fontweight='bold')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid()
    plt.gca()
    plt.legend(loc=4, fontsize=7)
    plt.savefig(base_picture_path + 'combine_models_' + str(outcome) + '.png')
    plt.show()


if __name__ == '__main__':
    # 分别对三个数据集划分训练集和测试集
    eicu_path = os.path.join('eicu', 'csvfiles')
    mimic3_path = os.path.join('mimic', 'mimic3', 'csvfiles')
    mimic4_path = os.path.join('mimic', 'mimic4', 'csvfiles')
    base_name = 'new_1228'
    eicu = read_file(path=os.path.join(base_path, eicu_path), filename=base_name)
    mimic3 = read_file(path=os.path.join(base_path, mimic3_path), filename=base_name)
    mimic4 = read_file(path=os.path.join(base_path, mimic4_path), filename=base_name)
    # 找出所有数据集公共列
    common_columns = list(set(eicu.columns).intersection(set(mimic3.columns), set(mimic4.columns)))
    print('common columns : %s, common columns length : %s' % (common_columns, len(common_columns)))
    processer = Processing()
    merge, labels = data_merge(eicu, mimic3, mimic4, common_columns)
    for outcome, label in outcome_dict.items():
        new_labels = format_label(labels, label)
        x_train, x_test, y_train, y_test = train_test_split(merge, new_labels, test_size=0.2, shuffle=True)
        various_model(x_train, x_test, y_train, y_test)
