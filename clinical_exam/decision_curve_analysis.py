#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2023/1/11 13:49
# @Author : hlx
# @File : decision_curve_analysis.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from pandas import DataFrame
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

from dimension_reduction.xgboost_reduction import XGBoost_selective
from filter.common import standard_data_by_white
from filter.param import colors, marks
from ml.classification.classify_parameter import searchCVnames_ab_clinical, \
    base_models_clical, XGBoost_none
from pylab import mpl


# 临床模型的决策曲线分析解读:
# https://www.zxzyl.com/archives/1450/#:~:text=%E8%A7%A3%E8%AF%BB%E5%86%B3%E7%AD%96%E6%9B%B2%E7%BA%BF%E5%88%86%E6%9E%90

def plot_single_model_test_curve(model, data, label, name):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    # model_research = GridSearchCV(model, scoring='roc_auc', n_jobs=10, refit=True, cv=5, verbose=0)
    model.fit(x_train, y_train)
    # if name == 'Perceptron':
    #     test_predict_proba = model._predict_proba_lr(x_train)
    #     fpr, tpr, threshold = metrics.roc_curve(y_train, test_predict_proba[:, 1], pos_label=1)
    # elif name == 'LinearRegression' or name == 'BayesianRidge':
    #     test_predict_proba = model.predict(x_train)
    #     fpr, tpr, threshold = metrics.roc_curve(y_train, test_predict_proba, pos_label=1)
    # else:
    #     test_predict_proba = model.predict_proba(x_train)
    #     fpr, tpr, threshold = metrics.roc_curve(y_train, test_predict_proba[:, 1], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, color='r', label=r'%s train (area=%0.3f)' % (name, auc), lw=1, linestyle='--', markersize=2)
    if name == 'Perceptron':
        test_predict_proba = model._predict_proba_lr(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
    elif name == 'LinearRegression' or name == 'BayesianRidge':
        test_predict_proba = model.predict(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba, pos_label=1)
    else:
        test_predict_proba = model.predict_proba(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color='b', label=r'%s test (area=%0.3f)' % (name, auc), lw=1, linestyle='--', markersize=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
    # plt.title('%s  ROC' % name, fontsize=17)
    plt.legend(loc='lower right', fontsize=7)
    plt.grid()
    plt.savefig('%s_test.png' % name)
    plt.show()


def plot_multiple_model_test_curves(data, label, flag):
    """
    @param data: feature data
    @param label: labels
    @param flag: 0/1 english/chinese
    """
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    for model, name, color, mark in zip(base_models_clical, searchCVnames_ab_clinical, colors, marks):
        x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(label), test_size=0.2,
                                                            shuffle=True, random_state=42)
        x_train, x_test = standard_data_by_white(x_train, x_test)
        model.fit(x_train, y_train)
        if name == 'Perceptron':
            test_predict_proba = model._predict_proba_lr(x_test)
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        elif name == 'LinR' or name == 'BR':
            test_predict_proba = model.predict(x_test)
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba, pos_label=1)
        else:
            test_predict_proba = model.predict_proba(x_test)
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        if name != 'LinR' and name != 'KNN' and name != 'BR':
            y_predict = model.predict(x_test)
            accuracy = np.round(accuracy_score(y_test, y_predict), 3)
            precision = np.round(precision_score(y_test, y_predict), 3)
            recall = np.round(recall_score(y_test, y_predict), 3)
            f1 = np.round(f1_score(y_test, y_predict), 3)
        else:
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
        auc = np.round(metrics.auc(fpr, tpr), 3)
        plt.plot(fpr, tpr, color=color, label=r'%s(area=%.3f)' % (name, auc), lw=1, linestyle='--', marker=mark,
                 markersize=2)
        print('model : %s auc : %s accuracy : %s precision : %s recall : %s f1 : %s' % (
            name, auc, accuracy, precision, recall, f1))
    plt.xticks(np.arange(0, 1.05, 0.2), fontsize=9, fontproperties='Times New Roman')
    plt.yticks(np.arange(0, 1.05, 0.2), fontsize=9, fontproperties='Times New Roman')
    if flag:
        plt.xlabel('1 - 特异性', fontweight='bold', fontsize=12)
        plt.ylabel('敏感度', fontweight='bold', fontsize=12)
    else:
        plt.xlabel('1 - specificity', fontweight='bold', fontsize=12, fontproperties='Times New Roman')
        plt.ylabel('sensitivity', fontweight='bold', fontsize=12, fontproperties='Times New Roman')
    labels = plt.legend(loc='lower right', fontsize=8).get_texts()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.grid()
    if flag:
        plt.savefig('models_test_chi.svg', bbox_inches='tight', format='svg')
    else:
        plt.savefig('models_test.svg', bbox_inches='tight', format='svg')
    plt.show()





# 使用aps数据，查找出各结果的最相关特征
def univariate_analysis(data, label, columns, param):
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(label), test_size=0.2, shuffle=True,
                                                        random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    # 计算各特征优势比的幂
    coefs = np.exp(LogisticRegression().fit(x_train, y_train).coef_)
    # 保存各特征的系数
    coef_lr = pd.DataFrame({'var': columns, 'coef': coefs.flatten()})
    coef_lr.to_csv('analysis.csv', mode='w', encoding='utf-8', index=False, header=['var', 'coef'])
    index_sort = np.abs(coef_lr['coef'] - 1).sort_values().index
    coef_lr = coef_lr.iloc[index_sort, :][-15:]
    vars = np.array(coef_lr['var'])
    coefs = coef_lr['coef']
    # 分别绘制单变量分析结果图
    plot_univariate_analysis(vars, coefs, 'clinical_test')
    # 计算模型在全部数据上的auc
    gclf = RandomizedSearchCV(XGBClassifier(), param, scoring='roc_auc', n_jobs=10, cv=5)
    data_new = DataFrame()
    for column in vars:
        data_new = pd.concat([data_new, data[column]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(np.array(data_new), np.array(label), test_size=0.2,
                                                        shuffle=True, random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    # 输入数据为已归一化后数据 无须归一化
    gclf.fit(x_train, y_train)
    best_estimator = gclf.best_estimator_
    y_pred = best_estimator.predict_proba(x_test)
    print(y_pred)
    fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1], pos_label=1)
    test_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=r'top 15 test (area=%0.3f)' % test_auc, color='b')
    plt.xlabel('False positive rate', fontsize=15)
    plt.ylabel('True positive rate', fontsize=15)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # curve the calibration
    # y_pred = clf.predict_proba(x_test)[:, 1]
    # prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)
    # disp = CalibrationDisplay(prob_true, prob_pred, y_pred)
    # disp.plot()
    plt.legend()
    plt.grid()
    plt.legend(loc=4, fontsize=7)
    plt.savefig('clinical.png')
    plt.show()


def plot_univariate_analysis(vars, coefs, name):
    plt.figure(dpi=500)
    plt.grid(zorder=0)
    for var, coef, i in zip(vars, coefs, range(len(vars))):
        if coef < 1:
            error_kw = {'ecolor': '0.1', 'capsize': 5}
            plt.barh(var, coef, error_kw=error_kw, xerr=(1 - coef) * 0.01, facecolor='w')
            plt.barh(var, left=coef, width=1 - coef, color='green', zorder=5)
            plt.text(coef - 0.4, i, '%.2f' % coef)
        else:
            if int(np.sum(coefs) / coef) == 1:
                coef = int(np.sum(coefs) / coef) + 1
            error_kw = {'ecolor': '0.1', 'capsize': 5}
            plt.barh(var, left=1, width=coef - 1, color='purple', error_kw=error_kw, xerr=(coef - 1) * 0.01,
                     zorder=5)
            plt.text(coef + 0.1, i, '%.2f' % coef)
    pyplot.title(name, fontweight='bold')
    plt.xlim(0.1, 3.5)
    plt.xlabel('Odds ratio', fontweight='bold')
    plt.tight_layout()
    plt.savefig('univariate_anslysis.png')
    plt.show()


def xlsx_to_csv(xlsx_data):
    # 将dataframe中某列的数据类型转换为int
    # origin['术前支架植入'] = origin['术前支架植入'].astype('int')
    data_xls = pd.read_excel(xlsx_data)
    data_xls.to_csv('data_csx.csv', float_format='%.2f', encoding='utf-8', index=False)


if __name__ == '__main__':
    origin = pd.read_csv('data_csx_m.csv')
    columns = origin.columns[1:]
    origin['术前支架植入'] = origin['术前支架植入'].astype('int')
    origin['术前结石CT值'] = origin['术前结石CT值'].astype('float')
    label = np.array(origin.iloc[:, 0])
    data = origin.iloc[:, 1:]
    # plot_multiple_model_test_curves(data, label, 1)
    # plot_single_model_test_curve(XGBoost_none, np.array(data), np.array(label), 'XGBoost')
    # plot_multiple_model_test_curves(data, label)
    XGBoost_selective(data, label, 0)
    # catboost_selective(data, label, columns)
    # plot_decision_curve_analysis_on_test_set(XGBoost_none, data, label, 'XGBoost', 0)
