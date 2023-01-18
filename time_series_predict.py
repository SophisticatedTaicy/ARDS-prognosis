#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/12/28 20:10
# @Author : hlx
# @File : time_series_predict.py
# @Software: PyCharm
import os

import numpy as np
from pandas import DataFrame

from ARDS.data_process.process import Processing
from dimension_reduction.univariate_analyse import univariate_analysis
from filter.common import read_file, split_train_test, data_split_and_combine, judge_label_balance, format_label, \
    concat_array
from filter.param import outcome_dict

base_csv_path = os.path.abspath(os.path.join(os.getcwd(), 'ARDS'))
base_picture_path = os.path.abspath(os.getcwd())


def time_series_predict(self):
    """
    1.merge three datasets into a merged data
    2.use logistic regression to carry on univariate feature analysis and select the top 10 features
    3.use GBDT to predict patient 24 hours prognosis with the top 10 features data
    4.cricle step 1 to 3 and respectively predict patient 48 hours and 72 hours
    5.plot the picture for patient 24 hours, 48 hours, and 72 hours of the same prognosis in one picture
    6.plot all prognosis time series ROCs.
    """




if __name__ == '__main__':
    # 分别对三个数据集划分训练集和测试集
    eicu_path = os.path.join('eicu', 'csvfiles')
    mimic3_path = os.path.join('mimic', 'mimic3', 'csvfiles')
    mimic4_path = os.path.join('mimic', 'mimic4', 'csvfiles')
    base_name = 'valid_1207'
    base_48 = 'valid_1207_48'
    base_72 = 'valid_1207_72'
    eicu = read_file(path=os.path.join(base_csv_path, eicu_path), filename=base_name)
    mimic3 = read_file(path=os.path.join(base_csv_path, mimic3_path), filename=base_name)
    mimic4 = read_file(path=os.path.join(base_csv_path, mimic4_path), filename=base_name)
    # 找出所有数据集公共列
    common_columns = list(set(eicu.columns).intersection(set(mimic3.columns), set(mimic4.columns)))
    # 找到单变量分析相关的数据列
    common_columns = [item for item in common_columns if not 'changerate' in item and not 'variance' in item]
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
        eicu_x_train, eicu_x_test, eicu_y_train_ori, eicu_y_test_ori = split_train_test(eicu_data, eicu_label,
                                                                                        test_size=0.2)
        mimic3_x_train, mimic3_x_test, mimic3_y_train_ori, mimic3_y_test_ori = split_train_test(mimic3_data,
                                                                                                mimic3_label,
                                                                                                test_size=0.2)
        mimic4_x_train, mimic4_x_test, mimic4_y_train_ori, mimic4_y_test_ori = split_train_test(mimic4_data,
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
