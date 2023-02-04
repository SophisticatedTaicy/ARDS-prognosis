import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot, ticker
from pandas import DataFrame
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import auc, roc_curve

from ARDS.data_process.process import Processing
from filter.common import standard_data_by_white, data_split_and_combine, judge_label_balance, format_label, \
    concat_array
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

    def select_specific_columns_and_compute_auc(self, x_train, y_train, x_test, y_test, columns):
        x_train_new = np.array(x_train[columns])
        x_test_new = np.array(x_test[columns])
        x_train_new, x_test_new = standard_data_by_white(x_train_new, x_test_new)
        # 输入数据为已归一化后数据 无须归一化
        GBDT.fit(x_train_new, y_train)
        # best_estimator = gclf.best_estimator_
        y_pred = GBDT.predict_proba(x_test_new)
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1], pos_label=1)
        test_auc = auc(fpr, tpr)
        return test_auc

    def univate_test(self, x_train, x_test, y_train, y_test, datas, test_name):
        analyser = univariate_analysis(common_columns)
        # 设置单变量分析的模型
        model = LogisticRegression()
        x_train_new, x_test_new = standard_data_by_white(np.array(x_train), np.array(x_test))
        model.fit(np.array(x_train_new), np.array(y_train))
        # 计算各特征优势比的幂
        coefs = np.exp(model.coef_)
        # 保存各特征的系数
        coef_lr = pd.DataFrame({'var': common_columns, 'coef': coefs.flatten()})
        index_sort = np.abs(coef_lr['coef'] - 1).sort_values().index
        eicu_aucs = []
        combine_aucs = []
        mimic3_aucs = []
        mimic4_aucs = []
        # 计算相应auc，按照5的步长划分特征集个数
        for coef_num in dimensions:
            coefs = coef_lr.iloc[index_sort, :][-coef_num:]
            vars = np.array(coefs['var'])
            eicu_aucs.append(
                analyser.select_specific_columns_and_compute_auc(datas[0], datas[1], x_test, y_test, vars))
            mimic3_aucs.append(
                analyser.select_specific_columns_and_compute_auc(datas[2], datas[3], x_test, y_test, vars))
            mimic4_aucs.append(
                analyser.select_specific_columns_and_compute_auc(datas[4], datas[5], x_test, y_test, vars))
            combine_aucs.append(
                analyser.select_specific_columns_and_compute_auc(datas[6], datas[7], x_test, y_test, vars))
            # print('coef_num : %s vars : %s' % (coef_num, vars))
        aucs = [eicu_aucs, mimic3_aucs, mimic4_aucs, combine_aucs]
        fig, ax = plt.subplots(figsize=(10, 7.5))
        colors = ['red', 'orange', 'yellowgreen', 'deepskyblue']
        for color, dataset_name, auc in zip(colors, dataset_names, aucs):
            plt.plot(dimensions, auc, 'ro-', color=color, label=dataset_name)
            plt.xlabel('Dimensions', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
            plt.ylabel('AUC', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
            plt.title('%s(test = %s)' % (outcome, test_name), fontweight='bold', fontsize=20,
                      fontproperties='Times New Roman')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        plt.yticks(np.arange(0.6, 1.05, 0.05))
        plt.xticks(np.arange(5, 105, 10))
        plt.grid()
        plt.legend(labels=dataset_names, loc=4)
        plt.savefig('%s_test_%s.svg' % (outcome, test_name), format='svg')
        plt.show()


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
    print('common_columns : %s common_columns length : %s' % (common_columns, len(common_columns)))
    processer = Processing()
    analyser = univariate_analysis(common_columns)
    eicu_label = eicu['outcome']
    mimic3_label = mimic3['outcome']
    mimic4_label = mimic4['outcome']
    eicu_data = eicu[common_columns]
    mimic3_data = mimic3[common_columns]
    mimic4_data = mimic4[common_columns]
    while (1):
        eicu_x_train, eicu_x_test, eicu_y_train_ori, eicu_y_test_ori = train_test_split(eicu_data, eicu_label,
                                                                                        test_size=0.2)
        mimic3_x_train, mimic3_x_test, mimic3_y_train_ori, mimic3_y_test_ori = train_test_split(mimic3_data,
                                                                                                mimic3_label,
                                                                                                test_size=0.2)
        mimic4_x_train, mimic4_x_test, mimic4_y_train_ori, mimic4_y_test_ori = train_test_split(mimic4_data,
                                                                                                mimic4_label,
                                                                                                test_size=0.2)
        if judge_label_balance(eicu_y_train_ori, eicu_y_test_ori) and judge_label_balance(mimic3_y_train_ori,
                                                                                          mimic3_y_test_ori) and judge_label_balance(
            mimic4_y_train_ori, mimic4_y_test_ori):
            break
    # 根据不同预后结果划分标签0、1
    dimensions = np.arange(len(common_columns), 4, -5)
    for outcome, label in outcome_dict.items():
        # 标签转换
        eicu_y_train = format_label(eicu_y_train_ori, label)
        eicu_y_test = format_label(eicu_y_test_ori, label)
        mimic3_y_train = format_label(mimic3_y_train_ori, label)
        mimic3_y_test = format_label(mimic3_y_test_ori, label)
        mimic4_y_train = format_label(mimic4_y_train_ori, label)
        mimic4_y_test = format_label(mimic4_y_test_ori, label)
        # 融合数据
        combine_x_train = eicu_x_train.append(mimic3_x_train, ignore_index=True).append(mimic4_x_train)
        combine_y_train = DataFrame(concat_array([eicu_y_train, mimic3_y_train, mimic4_y_train]))
        combine_x_test = eicu_x_test.append(mimic3_x_test, ignore_index=True).append(mimic4_x_test)
        combine_y_test = DataFrame(concat_array([eicu_y_test, mimic3_y_test, mimic4_y_test]))
        # 数据转换为Dataframe
        eicu_x_train = DataFrame(eicu_x_train)
        eicu_x_test = DataFrame(eicu_x_test)
        mimic3_x_train = DataFrame(mimic3_x_train)
        mimic3_x_test = DataFrame(mimic3_x_test)
        mimic4_x_train = DataFrame(mimic4_x_train)
        mimic4_x_test = DataFrame(mimic4_x_test)
        datas = (eicu_x_train, eicu_y_train, mimic3_x_train, mimic3_y_train, mimic4_x_train, mimic4_y_train,
                 combine_x_train, combine_y_train)
        # 设置单变量分析的模型
        analyser.univate_test(eicu_x_train, eicu_x_test, eicu_y_train, eicu_y_test, datas, 'eICU')
        analyser.univate_test(combine_x_train, combine_x_test, combine_y_train, combine_y_test, datas, 'ARDset')
        analyser.univate_test(mimic3_x_train, mimic3_x_test, mimic3_y_train, mimic3_y_test, datas, 'MIMIC III')
        analyser.univate_test(mimic4_x_train, mimic4_x_test, mimic4_y_train, mimic4_y_test, datas, 'MIMIC IV')
