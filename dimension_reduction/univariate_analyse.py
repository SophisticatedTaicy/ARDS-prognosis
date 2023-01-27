import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import auc, roc_curve

from ARDS.data_process.process import Processing
from filter.common import concat_array, data_split_and_combine, judge_label_balance, format_label, \
    standard_data_by_white
from filter.param import *
from ml.classification.classify_parameter import XGB, XGB_param, GBDT_none

from neural_class import read_file

base_csv_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'ARDS'))
base_picture_path = os.path.abspath(os.getcwd())


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
    def univariate_analysis(self, outcome, x_train, x_test, y_train, y_test):
        model = LogisticRegression()
        x_train_new = DataFrame(np.array(x_train))
        x_test_new = DataFrame(np.array(x_test))
        x_train, x_test = standard_data_by_white(np.array(x_train), np.array(x_test))
        model.fit(np.array(x_train), np.array(y_train))
        # 计算各特征优势比的幂
        coefs = np.exp(model.coef_)
        # 保存各特征的系数
        coef_lr = pd.DataFrame({'var': self.columns, 'coef': coefs.flatten()})
        coef_path = self.csvpath + str(self.datatype) + '_' + str(outcome) + '.csv'
        # coef_lr.to_csv(os.path.join(base_picture_path, coef_path), mode='w', encoding='utf-8', index=False,
        #                header=['var', 'coef'])
        index_sort = np.abs(coef_lr['coef'] - 1).sort_values().index
        coef_lr = coef_lr.iloc[index_sort, :][-20:]
        vars = np.array(coef_lr['var'])
        new_vars = [item for item in vars]
        coefs = coef_lr['coef']
        new_col = []
        for col, name in zip(x_train_new.columns, self.columns):
            if name in new_vars:
                new_col.append(col)
        # 分别绘制单变量分析结果图
        self.plot_univariate_analysis(vars, coefs, self.datatype, 'univariate', outcome)
        # 计算模型在全部数据上的auc
        # gclf = RandomizedSearchCV(XGB, XGB_param, scoring='roc_auc', n_jobs=10, cv=5)
        gclf = RandomizedSearchCV(GBDT_none, {}, scoring='roc_auc', n_jobs=10, cv=5)
        # 输入数据为已归一化后数据 无须归一化
        gclf.fit(np.array(x_train), np.array(y_train))
        best_estimator = gclf.best_estimator_
        y_pred = best_estimator.predict_proba(np.array(x_test))
        fpr, tpr, threshold = roc_curve(np.array(y_test), y_pred[:, 1], pos_label=1)
        test_auc = auc(fpr, tpr)
        print('datatype : %s outcome : %s, all best params : %s, auc : %s' % (
            self.datatype, outcome, gclf.best_params_, test_auc))
        plt.plot(fpr, tpr, label=r'%s (area=%f)' % (self.datatype, test_auc), color='r')
        # 选取最重要的特征对应的数据
        x_train, x_test = x_train_new, x_test_new
        x_train_new = DataFrame()
        x_test_new = DataFrame()
        for var in new_col:
            x_train_new = pd.concat([x_train_new, x_train[var]], axis=1)
            x_test_new = pd.concat([x_test_new, x_test[var]], axis=1)
        x_train_new = np.array(x_train_new)
        x_test_new = np.array(x_test_new)
        x_train_new, x_test_new = standard_data_by_white(np.array(x_train_new), np.array(x_test_new))
        # clf = RandomizedSearchCV(GBDT, GBDT_param, scoring='roc_auc', n_jobs=10, cv=5)
        # 输入数据为已归一化后数据 无须归一化
        best_estimator.fit(np.array(x_train_new), y_train)
        # best_estimator = gclf.best_estimator_
        y_pred = best_estimator.predict_proba(np.array(x_test_new))
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1], pos_label=1)
        test_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=r'top 15(area=%f)' % test_auc, color='b')
        plt.title(r'%s' % outcome, fontweight='bold', fontsize=15)
        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate', fontsize=15)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.grid()
        plt.legend(loc=4, fontsize=7)
        plt.savefig('D:/pycharm/ARDS-prognosis-for-eICU-data/univariate.png')
        # plt.savefig(self.picturepath + str(self.datatype) + '_' + str(outcome) + '.png')
        plt.show()


if __name__ == '__main__':
    # 分别对三个数据集划分训练集和测试集
    eicu_path = os.path.join('eicu', 'csvfiles')
    mimic3_path = os.path.join('mimic', 'mimic3', 'csvfiles')
    mimic4_path = os.path.join('mimic', 'mimic4', 'csvfiles')
    base_name = 'valid_1207'
    eicu = read_file(path=os.path.join(base_csv_path, eicu_path), filename=base_name)
    mimic3 = read_file(path=os.path.join(base_csv_path, mimic3_path), filename=base_name)
    mimic4 = read_file(path=os.path.join(base_csv_path, mimic4_path), filename=base_name)
    # 找出所有数据集公共列
    common_columns = list(set(eicu.columns).intersection(set(mimic3.columns), set(mimic4.columns)))
    # 找到单变量分析相关的数据列
    common_columns = [item for item in common_columns if not 'changerate' in item and not 'variance' in item]
    print('common_columns : %s common_columns length : %s' % (common_columns, len(common_columns)))
    processer = Processing()
    eicu_analyser = univariate_analysis(common_columns, datatype='eICU')
    mimic3_analyser = univariate_analysis(common_columns, datatype='MIMIC III')
    mimic4_analyser = univariate_analysis(common_columns, datatype='MIMIC IV')
    eicu_label, mimic3_label, mimic4_label, eicu_data, mimic3_data, mimic4_data = data_split_and_combine(eicu,
                                                                                                         mimic3,
                                                                                                         mimic4,
                                                                                                         common_columns)
    combine_analyser = univariate_analysis(common_columns, datatype='combine')
    while (1):
        eicu_label, mimic3_label, mimic4_label, eicu_data, mimic3_data, mimic4_data = data_split_and_combine(eicu,
                                                                                                             mimic3,
                                                                                                             mimic4,
                                                                                                             common_columns)
        eicu_x_train, eicu_x_test, eicu_y_train_ori, eicu_y_test_ori = train_test_split(eicu_data, eicu_label,
                                                                                        test_size=0.2)
        mimic3_x_train, mimic3_x_test, mimic3_y_train_ori, mimic3_y_test_ori = train_test_split(mimic3_data,
                                                                                                mimic3_label,
                                                                                                test_size=0.2)
        mimic4_x_train, mimic4_x_test, mimic4_y_train_ori, mimic4_y_test_ori = train_test_split(mimic4_data,
                                                                                                mimic4_label,
                                                                                                test_size=0.2)
        common = np.array(common_columns)
        if judge_label_balance(eicu_y_train_ori, eicu_y_test_ori) and judge_label_balance(mimic3_y_train_ori,
                                                                                          mimic3_y_test_ori) and judge_label_balance(
            mimic4_y_train_ori, mimic4_y_test_ori):
            common_columns.remove('outcome')
            break
    # 根据不同预后结果划分标签0、1
    for outcome, label in outcome_dict.items():
        # 标签转换
        temp = np.array(list(mimic3_x_test))
        eicu_y_train = format_label(eicu_y_train_ori, label)
        eicu_y_test = format_label(eicu_y_test_ori, label)
        mimic3_y_train = format_label(mimic3_y_train_ori, label)
        mimic3_y_test = format_label(mimic3_y_test_ori, label)
        mimic4_y_train = format_label(mimic4_y_train_ori, label)
        mimic4_y_test = format_label(mimic4_y_test_ori, label)
        # 融合数据
        combine_x_train = DataFrame(concat_array([eicu_x_train, mimic3_x_train, mimic4_x_train]))
        combine_y_train = DataFrame(concat_array([eicu_y_train, mimic3_y_train, mimic4_y_train]))
        combine_x_test = DataFrame(concat_array([eicu_x_test, mimic3_x_test, mimic4_x_test]))
        combine_y_test = DataFrame(concat_array([eicu_y_test, mimic3_y_test, mimic4_y_test]))
        # 数据转换为Dataframe
        eicu_x_train = DataFrame(eicu_x_train)
        eicu_x_test = DataFrame(eicu_x_test)
        mimic3_x_train = DataFrame(mimic3_x_train)
        mimic3_x_test = DataFrame(mimic3_x_test)
        mimic4_x_train = DataFrame(mimic4_x_train)
        mimic4_x_test = DataFrame(mimic4_x_test)
        # univariate analysis
        eicu_analyser.univariate_analysis(outcome, eicu_x_train, eicu_x_test, eicu_y_train, eicu_y_test)
        mimic3_analyser.univariate_analysis(outcome, mimic3_x_train, mimic3_x_test, mimic3_y_train, mimic3_y_test)
        mimic4_analyser.univariate_analysis(outcome, mimic4_x_train, mimic4_x_test, mimic4_y_train, mimic4_y_test)
        combine_analyser.univariate_analysis(outcome, combine_x_train, combine_x_test, combine_y_train, combine_y_test)
