#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/11/7 10:55
# @Author : hlx
# @File : combine.py
# @Software: PyCharm
import pandas as pd
from pandas import DataFrame

from filter.common import save_file, read_file


def combine_datas():
    """
    @param eicu: eicu数据集
    @param mimic3: mimic3数据集
    @param mimic4: mimic4数据集
    @return 三个数据集合并结果
    """
    # 将三个数据集中且非空的列找出
    # 将当前列数据合并为一列，使用80-20分位数的差进行合并
    # 去除各自数据和合并最终所有数据
    eicu = read_file(path='D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS\eicu\csvfiles', filename='valid_1108_72')
    mimic3 = read_file(path='D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS\mimic\mimic3\csvfiles', filename='valid_1108_72')
    mimic4 = read_file(path='D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS\mimic\mimic4\csvfiles', filename='valid_1108_72')
    base_path = 'D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS\combine\csvfiles'
    header = list(set(eicu.columns) & set(mimic3.columns) & set(mimic4.columns))
    combine_columns = []
    eicu_median_new = DataFrame()
    mimic3_median_new = DataFrame()
    mimic4_median_new = DataFrame()
    combine_median_new = DataFrame()
    combine_median_columns = []
    for column in header:
        # if current column is non column for any database,then discard
        mimic3_column = list(mimic3[column])
        eicu_column = list(eicu[column])
        mimic4_column = list(mimic4[column])
        combine_columns.append(column)
        column_data = mimic3_column + eicu_column + mimic4_column
        # 单变量分析中动态数据归一化结果汇总
        if 'changerate' in column or 'variances' in column:
            continue
        eicu_median_new = pd.concat([eicu_median_new, DataFrame(eicu_column)], axis=1)
        mimic3_median_new = pd.concat([mimic3_median_new, DataFrame(mimic3_column)], axis=1)
        mimic4_median_new = pd.concat([mimic4_median_new, DataFrame(mimic4_column)], axis=1)
        combine_median_new = pd.concat([combine_median_new, DataFrame(column_data)], axis=1)
        combine_median_columns.append(column)
    save_file(file=eicu_median_new, path=base_path, filename='combine_median_eicu_1108_72', header=combine_median_columns)
    save_file(file=mimic3_median_new, path=base_path, filename='combine_median_mimic3_1108_72',
              header=combine_median_columns)
    save_file(file=mimic4_median_new, path=base_path, filename='combine_median_mimic4_1108_72',
              header=combine_median_columns)
    save_file(file=combine_median_new, path=base_path, filename='combine_median_1108_72', header=combine_median_columns)


if __name__ == '__main__':
    combine_datas()
