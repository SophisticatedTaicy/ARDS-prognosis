#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/12/21 19:46
# @Author : hlx
# @File : model_selection.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import interp, mean
from pandas import DataFrame
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import os

from sklearn.model_selection import train_test_split

from filter.common import read_file, format_label, standard_data_by_white, concat_array
from filter.param import outcome_dict, colors
from pylab import mpl

from ml.classification.classify_parameter import model_names_sec, base_models_sec

base_picture_path = './ARDS/combine/pictures/'
base_csv_path = './ARDS/combine/csvfiles'
base_path = os.path.dirname(os.path.abspath(__file__)) + '\ARDS'
from ARDS.data_process.process import Processing


# 在多个机器学习模型上融合训练数据，在统一测试集上查看模型性能
# 使用不同模型测试融合数据性能
# 新增accuracy\precision\recall\f1-score
def various_model(x_train, x_test, y_train, y_test, dataset_name, outcome):
    plt.figure(figsize=(1.92, 2.12))
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    # 将训练集和测试集白化
    x_train, x_test = standard_data_by_white(x_train, x_test)
    english_chinese_dict = {
        'Spontaneous Recovery': '自发恢复',
        'Long Stay': '长期住院',
        'Rapid Death': '快速死亡'
    }
    items = []
    for name, model, color in zip(model_names_sec, base_models_sec, colors[:len(base_models_sec)]):
        i = 0
        mean_tpr = []
        mean_fpr = np.linspace(0, 1, 1000)
        tprs = []
        items = []
        mean_accuracy = []
        mean_f1 = []
        while i < 10:
            model.fit(x_train, y_train)
            if name == 'Perceptron':
                test_predict_proba = model._predict_proba_lr(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            # elif name == 'Linear Regression' or name == 'Bayesian Ridge':
            elif name == 'LinR' or name == 'BR':
                test_predict_proba = model.predict(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba, pos_label=1)
            else:
                test_predict_proba = model.predict_proba(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            tprs.append(interp(np.linspace(0, 1, 1000), fpr, tpr))
            mean_tpr = np.mean(tprs, axis=0)
            i += 1
            if name != 'LinR' and name != 'KNN' and name != 'BR':
                y_predict = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_predict)
                # precision = precision_score(y_test, y_predict)
                # recall = recall_score(y_test, y_predict)
                f1 = f1_score(y_test, y_predict)
            else:
                accuracy = 0
                # precision = 0
                # recall = 0
                f1 = 0
            mean_accuracy.append(accuracy)
            # mean_precision.append(precision)
            # mean_recall.append(recall)
            mean_f1.append(f1)
        mean_auc = np.round(auc(mean_fpr, mean_tpr), 2)
        mean_accuracy = np.round(np.mean(mean_accuracy), 2)
        # mean_precision = np.round(np.mean(mean_precision), 2)
        # mean_recall = np.round(np.mean(mean_recall), 2)
        mean_f1 = np.round(np.mean(mean_f1), 2)
        item = (mean_auc, mean_accuracy, mean_f1)
        print('outcome : %s model : %s evaluation baseline : %s' % (outcome, name, item))
        items.append(item)
        # plt.plot(mean_fpr[:-1:30], mean_tpr[:-1:30], label=r'%s(area=%.2f)' % (name, mean_auc), color=color,
        #          marker='o', markersize=0.6, lw=0.5)
        plt.plot(mean_fpr[:-1:30], mean_tpr[:-1:30], label=r'%s(area=%.2f)' % (name, mean_auc), color=color, lw=0.5)
    plt.xlabel('假阳率', fontsize=9, fontweight='bold')
    plt.ylabel('真阳率', fontsize=9, fontweight='bold')
    plt.yticks(np.arange(0, 1.05, 0.2), fontsize=6, fontproperties='Times New Roman')
    plt.xticks(np.arange(0, 1.05, 0.2), fontsize=6, fontproperties='Times New Roman')
    plt.grid()
    labelss = plt.legend(loc=4, fontsize=5).get_texts()
    [label.set_fontname('Times New Roman') for label in labelss]
    plt.savefig(base_picture_path + '%s_%s.svg' % (outcome, dataset_name), bbox_inches='tight', format='svg')
    DataFrame(items).to_csv('%s_evaluations.csv' % outcome, mode='w', index=False)
    plt.show()


def various_outcome(datas, labels, model, name):
    plt.figure(figsize=(1.92, 2.12), dpi=150)
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    # 将训练集和测试集白化
    anglish_chinese_dict = {
        'Spontaneous Recovery': '自发恢复',
        'Long Stay': '长期住院',
        'Rapid Death': '快速死亡'
    }
    items = []
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']
    for outcome, label, data, color in zip(outcome_dict.keys(), outcome_dict.values(), datas, colors):
        new_labels = format_label(labels, label)
        x_train, x_test, y_train, y_test = train_test_split(np.array(combine_data), np.array(new_labels), test_size=0.2,
                                                            shuffle=True)
        x_train, x_test = standard_data_by_white(x_train, x_test)
        model.fit(x_train, y_train)
        if name == 'Perceptron':
            test_predict_proba = model._predict_proba_lr(x_test)
            fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        # elif name == 'Linear Regression' or name == 'Bayesian Ridge':
        elif name == 'LinR' or name == 'BR':
            test_predict_proba = model.predict(x_test)
            fpr, tpr, threshold = roc_curve(y_test, test_predict_proba, pos_label=1)
        else:
            test_predict_proba = model.predict_proba(x_test)
            fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        mean_auc = auc(fpr, tpr)
        y_predict = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)
        item = (mean_auc, accuracy, precision, recall, f1)
        items.append(item)
        plt.plot(fpr[:-1:30], tpr[:-1:30], label=r'%s(area=%.2f)' % (anglish_chinese_dict[outcome], mean_auc),
                 color=color, marker='o', markersize=0.6, lw=0.5)
    plt.xlabel('假阳率', fontsize=9, fontweight='bold')
    plt.ylabel('真阳率', fontsize=9, fontweight='bold')
    plt.yticks(np.arange(0, 1.05, 0.2), fontsize=7, fontproperties='Times New Roman')
    plt.xticks(np.arange(0, 1.05, 0.2), fontsize=7, fontproperties='Times New Roman')
    plt.grid()
    plt.legend(loc=4, fontsize=6)
    labelss = plt.legend(loc=4, fontsize=7).get_texts()
    print(items)
    [label.set_fontname('Times New Roman') for label in labelss]
    DataFrame(items).to_csv('model_evaluation_score.csv', index=False)
    plt.savefig(base_picture_path + '%s_chinese_no_title.svg' % name, bbox_inches='tight', format='svg')
    plt.show()


if __name__ == '__main__':
    # 分别对三个数据集划分训练集和测试集
    base_path = '.\ARDS'
    total_path = os.path.join('combine', 'csvfiles')
    eicu = read_file(path=os.path.join(base_path, total_path), filename='merge_eicu')
    mimic3 = read_file(path=os.path.join(base_path, total_path), filename='merge_mimic3')
    mimic4 = read_file(path=os.path.join(base_path, total_path), filename='merge_mimic4')
    total = read_file(path=os.path.join(base_path, total_path), filename='merge_data')
    coulmns = list(total.columns)
    coulmns.remove('outcome')
    common_columns = coulmns
    eicu_label = eicu['outcome']
    mimic3_label = mimic3['outcome']
    mimic4_label = mimic4['outcome']
    eicu_data = eicu[common_columns]
    mimic3_data = mimic3[common_columns]
    mimic4_data = mimic4[common_columns]
    combine_data = eicu_data.append(mimic3_data, ignore_index=True).append(mimic4_data)
    combine_label = pd.Series(concat_array([eicu_label, mimic3_label, mimic4_label]))
    processer = Processing()
    datas = [eicu_data, eicu_label, mimic3_data, mimic3_label, mimic4_data, mimic4_label, combine_data, combine_label]
    dataset_names = ['eICU', 'MIMIC III', 'MIMIC IV', 'ARDset']
    # various_outcome(combine_data, combine_label, GBDT_none, 'GBDT')
    for outcome, label, data in zip(outcome_dict.keys(), outcome_dict.values(), datas):
        new_labels = format_label(combine_label, label)
        x_train, x_test, y_train, y_test = train_test_split(np.array(combine_data), np.array(new_labels), test_size=0.2,
                                                            shuffle=True)
        various_model(x_train, x_test, y_train, y_test, 'ARDset', outcome)
    # for i, dataset_name in zip(range(4), dataset_names):
    #     data = datas[2 * i]
    #     labels = datas[2 * i + 1]
    #     new_labels = format_label(labels, label)
    #     x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(new_labels), test_size=0.2,
    #                                                         shuffle=True)
    #     various_model(x_train, x_test, y_train, y_test, dataset_name)
