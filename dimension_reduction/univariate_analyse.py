import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot, ticker
from pandas import DataFrame
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import auc, roc_curve

from filter.common import standard_data_by_white
from filter.param import *
from ml.classification.classify_parameter import GBDT_param, GBDT

from neural_class import read_file

base_csv_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'ARDS'))
base_picture_path = os.path.abspath(os.getcwd())
global dataset_names
dataset_names = ['eICU', 'MIMIC III', 'MIMIC IV', 'ARDset']


class univariate_analysis:
    def __init__(self, columns, datatype='eicu'):
        self.picturepath = './univariate_analysis_pictures/'
        self.csvpath = './csvfiles/'
        self.datatype = datatype
        self.columns = columns

    # 单变量分析图形绘制
    def plot_univariate_analysis(self, vars, coefs, datatype, mark, name):
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
        plt.savefig(self.picturepath + str(mark) + '_' + str(datatype) + '_' + str(name) + '.png')
        plt.show()

    # 使用aps数据，查找出各结果的最相关特征
    def univariate_analysis(self, outcome, outcome_label, data, labels, common_columns):
        new_labels = []
        for item in np.array(labels):
            if item == outcome_label:
                new_labels.append(1)
            else:
                new_labels.append(0)
        new_labels = np.array(new_labels)
        x_train, x_test, y_train, y_test = train_test_split(np.array(data), new_labels, test_size=0.2, shuffle=True)

        def compute_auc_changed_by_dimension(data, labels, coefs):
            x_train, x_test, y_train, y_test = train_test_split(np.array(data), labels, test_size=0.2, shuffle=True)
            index_sort = np.abs(coefs['coef'] - 1).sort_values().index
            x_train_ori = DataFrame(x_train)
            x_test_ori = DataFrame(x_test)
            final_aucs = []
            for coef_num in np.arange(len(common_columns), 4, -5):
                coef_lr = coefs.iloc[index_sort, :][-coef_num:]
                vars = np.array(coef_lr['var'])
                new_vars = [item for item in vars]
                new_col = []
                for col_num, name in zip(x_train_ori.columns, self.columns):
                    if name in new_vars:
                        new_col.append(col_num)
                # 分别绘制单变量分析结果图
                # self.plot_univariate_analysis(vars, coefs, self.datatype, 'univariate', outcome)
                # 计算模型在全部数据上的auc
                # gclf = RandomizedSearchCV(XGB, XGB_param, scoring='roc_auc', n_jobs=10, cv=5)
                gclf = RandomizedSearchCV(GBDT, GBDT_param, scoring='roc_auc', n_jobs=10, cv=5)
                gclf = RandomizedSearchCV(GBDT, {}, scoring='roc_auc', n_jobs=10, cv=5)
                # # 输入数据为已归一化后数据 无须归一化
                # y_pred = best_estimator.predict_proba(np.array(x_test))
                # fpr, tpr, threshold = roc_curve(np.array(y_test), y_pred[:, 1], pos_label=1)
                # test_auc = auc(fpr, tpr)
                # print('datatype : %s outcome : %s, all best params : %s, auc : %s' % (
                #     self.datatype, outcome, gclf.best_params_, test_auc))
                # plt.plot(fpr, tpr, label=r'%s (area=%f)' % (self.datatype, test_auc), color='r')
                # # 选取最重要的特征对应的数据
                x_train_new = DataFrame()
                x_test_new = DataFrame()
                for var in new_col:
                    x_train_new = pd.concat([x_train_new, x_train_ori[var]], axis=1)
                    x_test_new = pd.concat([x_test_new, x_test_ori[var]], axis=1)
                x_train_new = np.array(x_train_new)
                x_test_new = np.array(x_test_new)
                x_train_new, x_test_new = standard_data_by_white(x_train_new, x_test_new)
                # clf = RandomizedSearchCV(GBDT, GBDT_param, scoring='roc_auc', n_jobs=10, cv=5)
                # 输入数据为已归一化后数据 无须归一化
                gclf.fit(x_train_new, y_train)
                # best_estimator = gclf.best_estimator_
                y_pred = gclf.predict_proba(x_test_new)
                fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1], pos_label=1)
                test_auc = auc(fpr, tpr)
                final_aucs.append(round(test_auc, 3))

            return final_aucs

        model = LogisticRegression()
        x_train, x_test = standard_data_by_white(x_train, x_test)
        model.fit(np.array(x_train), np.array(y_train))
        # 计算各特征优势比的幂
        coefs = np.exp(model.coef_)
        # 保存各特征的系数
        coef_lr = pd.DataFrame({'var': self.columns, 'coef': coefs.flatten()})
        index_sort = np.abs(coef_lr['coef'] - 1).sort_values().index
        n_vars = coef_lr.iloc[index_sort, :]
        coef_path = self.csvpath + str(self.datatype) + '_' + str(outcome) + '.csv'
        DataFrame(n_vars).to_csv(os.path.join(base_picture_path, coef_path), mode='w', encoding='utf-8', index=False,
                                 header=['var', 'coef'])
        final_te = compute_auc_changed_by_dimension(data, new_labels, coef_lr)
        return final_te


if __name__ == '__main__':
    # 分别对三个数据集划分训练集和测试集
    base_path = 'D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS'
    total_path = os.path.join('combine', 'csvfiles')
    eicu = read_file(path=os.path.join(base_path, total_path), filename='merge_eicu')
    mimic3 = read_file(path=os.path.join(base_path, total_path), filename='merge_mimic3')
    mimic4 = read_file(path=os.path.join(base_path, total_path), filename='merge_mimic4')
    total = read_file(path=os.path.join(base_path, total_path), filename='merge_data')
    coulmns = list(total.columns)
    coulmns.remove('outcome')
    common_columns = coulmns
    dimensions = np.arange(len(common_columns), 4, -5)
    print("common_columns length %s and columns : %s" % (len(common_columns), common_columns))
    eicu_analyser = univariate_analysis(common_columns, datatype='eICU')
    mimic3_analyser = univariate_analysis(common_columns, datatype='MIMIC III')
    mimic4_analyser = univariate_analysis(common_columns, datatype='MIMIC IV')
    total_analyser = univariate_analysis(common_columns, datatype='ARDset')
    colors = ['red', 'orange', 'yellowgreen', 'deepskyblue']
    for outcome, outcome_label in outcome_dict.items():
        eicu_aucs = eicu_analyser.univariate_analysis(outcome, outcome_label, eicu[common_columns], eicu['outcome'],
                                                      common_columns)
        mimic3_aucs = mimic3_analyser.univariate_analysis(outcome, outcome_label, mimic3[common_columns],
                                                          mimic3['outcome'], common_columns)
        mimic4_aucs = mimic4_analyser.univariate_analysis(outcome, outcome_label, mimic4[common_columns],
                                                          mimic4['outcome'], common_columns)
        total_aucs = total_analyser.univariate_analysis(outcome, outcome_label, total[common_columns], total['outcome'],
                                                        common_columns)
        aucs = [eicu_aucs, mimic3_aucs, mimic4_aucs, total_aucs]
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.grid(zorder=0)
        for color, dataset_name, auc in zip(colors, dataset_names, aucs):
            plt.plot(dimensions, auc, 'ro-', color=color, label=dataset_name)
            plt.xlabel('Dimensions', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
            plt.ylabel('AUC', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
            plt.title(outcome, fontweight='bold', fontsize=15, fontproperties='Times New Roman')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        plt.yticks(np.arange(0.6, 1.05, 0.05))
        plt.xticks(np.arange(5, 105, 10))
        plt.grid()
        plt.legend(labels=dataset_names, loc=4)
        plt.savefig('%s.svg' % outcome, format='svg')
        plt.show()
