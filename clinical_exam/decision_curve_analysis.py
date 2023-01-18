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
from scikitplot import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from filter.common import standard_data_by_white
from ml.classification.classify_parameter import GBDT, XGBoost_none


# 临床模型的决策曲线分析解读:
# https://www.zxzyl.com/archives/1450/#:~:text=%E8%A7%A3%E8%AF%BB%E5%86%B3%E7%AD%96%E6%9B%B2%E7%BA%BF%E5%88%86%E6%9E%90

def plot_single_model_test_curve(model, data, label, name):
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
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    plt.title('%s  ROC' % name, fontsize=17)
    plt.legend(loc='lower right', fontsize=7)
    plt.grid()
    plt.savefig('%s_test.png' % name)
    plt.show()


def plot_multiple_model_test_curves(models, data, label, names, colors, marks):
    for model, name, color, mark in zip(models, names, colors, marks):
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True,
                                                            random_state=42)
        x_train, x_test = standard_data_by_white(x_train, x_test)
        model.fit(x_train, y_train)
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
        plt.plot(fpr, tpr, color=color, label=r'%s test (area=%0.3f)' % (name, auc), lw=1, linestyle='--', marker=mark,
                 markersize=2)
        # plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='gray', label='Luck', alpha=0.8) 对角线
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    plt.title('test ROCs', fontsize=17)
    plt.legend(loc='lower right', fontsize=7)
    plt.grid()
    plt.savefig('models_test.png')
    plt.show()


def plot_decision_curve_analysis_on_test_set(model, data, label, name):
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(label), test_size=0.2, shuffle=True,
                                                        random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    # model_research = GridSearchCV(model, scoring='roc_auc', n_jobs=10, refit=True, cv=5, verbose=0)
    model.fit(x_train, y_train)
    if name == 'Perceptron':
        test_predict_proba = model._predict_proba_lr(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        net_benefit_model = calculate_net_benefit_model(threshold, test_predict_proba[:, 1], y_test)
    elif name == 'LinearRegression' or name == 'BayesianRidge':
        test_predict_proba = model.predict(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba, pos_label=1)
        net_benefit_model = calculate_net_benefit_model(threshold, test_predict_proba, y_test)
    else:
        test_predict_proba = model.predict_proba(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        net_benefit_model = calculate_net_benefit_model(threshold, test_predict_proba[:, 1], y_test)
    net_benefit_all = calculate_net_benefit_all(threshold, y_test)
    fig, ax = plt.subplots()
    plot_DCA(ax, threshold, net_benefit_model, net_benefit_all)
    plt.show()


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    # Plot
    ax.plot(thresh_group, net_benefit_model, color='crimson', label='Model')
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

    # Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    return ax


# 使用aps数据，查找出各结果的最相关特征
def univariate_analysis(data, label, columns, model, param, name):
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
    gclf = RandomizedSearchCV(model, param, scoring='roc_auc', n_jobs=10, cv=5)
    # 输入数据为已归一化后数据 无须归一化
    gclf.fit(x_train, y_train)
    best_estimator = gclf.best_estimator_
    y_pred = best_estimator.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(np.array(y_test), y_pred[:, 1], pos_label=1)
    test_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=r'%s (area=%0.3f)' % (name, test_auc), color='r')
    # 选取最重要的特征对应的数据
    data_new = DataFrame()
    for column in vars:
        data_new = pd.concat([data_new, data[column]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(np.array(data_new), np.array(label), test_size=0.2,
                                                        shuffle=True, random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    # 输入数据为已归一化后数据 无须归一化
    best_estimator.fit(x_train, y_train)
    y_pred = best_estimator.predict_proba(x_train)
    fpr, tpr, threshold = roc_curve(y_train, y_pred[:, 1], pos_label=1)
    test_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=r'top 15 train (area=%0.3f)' % test_auc, color='g')
    y_pred = best_estimator.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1], pos_label=1)
    test_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=r'top 15 test (area=%0.3f)' % test_auc, color='b')
    plt.xlabel('False positive rate', fontsize=15)
    plt.ylabel('True positive rate', fontsize=15)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
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


if __name__ == '__main__':
    origin = pd.read_csv('test_1.csv')
    label = origin.iloc[:, 0]
    data = origin.iloc[:, 1:]
    univariate_analysis(data, label, origin.columns[1:], XGBoost_none, {}, 'XGBoost')
    # plot_decision_curve_analysis_on_test_set(XGBoost_none, data, label, 'XGBoost')
