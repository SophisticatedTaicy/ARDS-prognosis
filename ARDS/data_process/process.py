#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/11/7 10:48
# @Author : hlx
# @File : process.py
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from filter.common import save_file, read_file
from filter.param import *

base_path = os.path.dirname(os.path.abspath(__file__)) + '\ARDS'


class Processing:
    def compute_blank_rate(self, data):
        '''
        data:dataframe格式数据 每一维为一个特征
        return： 每一维数据的缺失率，即数据为0 或-的数量占总体数据的比率
        '''
        blank_rate = {}
        columns = data.columns
        for column in columns:
            current_data = np.array(data[column])
            blank_data_sum = len([item for item in current_data if item == '-' or float(item) <= 0])
            blank_rate[column] = round(blank_data_sum / len(current_data), 4)
        return blank_rate

    def extra_not_blank_data(self, data):
        """
        @param data: 提取的ARDS数据
        @return: 不全为空值或异常值的列数据的组合以及相应列名
        """
        header = drug_list + diagnosis_abbrevation_list + person_info_list + dynamic_detail_list
        final_columns = []
        new_data = DataFrame()
        # 挑选出不为空的所有数据列的数据,使用0填充空值
        for column in header:
            # 对当前列中大于1的数据除以100?
            current = data[column]
            current_data = [0 if np.isnan(item) or item < 0 else item for item in current]
            blank_data_sum = len([item for item in current_data if float(item) == 0])
            if blank_data_sum / len(current_data) < 1:
                new_data = pd.concat([new_data, DataFrame(current_data)], axis=1)
                final_columns.append(column)
        return new_data, final_columns

    def fill_invalid_data_with_0(self, data, columns):
        '''
        data:去空列后的数据
        result:元数据中存储有预后结果的数据项
        return:在空值处补充0，并使用80分位数-20分位数的方式归一化数据
        '''
        new_data = DataFrame(data.iloc[:, :36])
        median_data = DataFrame(data.iloc[:, :36])
        median_columns = columns[:36]
        # 计算动态指标中位数标准化后的数值
        for column in columns[36:]:
            current_data = data[column]
            if not self.judge_missingness_indicator(current_data):
                # 对非0、1数列进行归一化
                non_zero_data = [item for item in current_data if item > 0]
                per_2 = np.percentile(non_zero_data, 20)
                per_8 = np.percentile(non_zero_data, 80)
                dif = per_8 - per_2
                if dif:
                    current_data = np.round(current_data / dif, 4)
            new_data = pd.concat([new_data, DataFrame(current_data)], axis=1)
            if 'changerate' in column or 'variances' in column:
                continue
            else:
                median_data = pd.concat([median_data, DataFrame(current_data)], axis=1)
                median_columns.append(column)
        return new_data, median_data, median_columns

    def extract_data_from_eICU(self, data, columns):
        '''
        data：eICU
        columns:需要抽取的列数据
        return：eICU数据中抽取到的数据
        '''
        columns = list(columns)
        new_data = DataFrame()
        # 取出对应的所有列数据
        for column in columns:
            new_data = pd.concat([new_data, data[column]], axis=1)
        # 提取静态态指标数据列
        data = new_data.iloc[:, :40]
        # 动态指标规格化
        for i in range(40, len(columns) - 1):
            current_data = new_data[columns[i]]
            per_2 = np.percentile(current_data, 20)
            per_8 = np.percentile(current_data, 80)
            dif = per_8 - per_2
            if dif:
                current_data = np.round(current_data / dif, 4)
            data = pd.concat([data, DataFrame(current_data)], axis=1)
        data = pd.concat([data, new_data['outcome']], axis=1)
        return data

    def format_ards_data(self, type):
        """
        @param datas:查找的所有ARDS患者数据
        """
        if type == '3' or type == '4':
            path = 'mimic\mimic' + str(type) + '\csvfiles'
        else:
            path = 'eicu\csvfiles'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
        datas = read_file(filename='result_1228', path=path)
        # 删除重复数据
        result = datas.drop_duplicates()
        header = result_header + ['outcome_48', 'outcome_72']
        # # 保存取出重复项后的数据并添加表名
        save_file(file=result, path=path, filename='result_1228', header=header)
        # 需要读取数据及表名
        result = read_file(filename='result_1228', path=path)
        # 将非全部异常列数据计算出来并保存
        valid_data, columns = self.extra_not_blank_data(result)
        save_file(pd.concat([valid_data, result['outcome']], axis=1), path=path, filename='new_1228',
                  header=columns + ['outcome'])
        save_file(pd.concat([valid_data, result['outcome_48']], axis=1), path=path, filename='new_1228_48',
                  header=columns + ['outcome_48'])
        save_file(pd.concat([valid_data, result['outcome_72']], axis=1), path=path, filename='new_1228_72',
                  header=columns + ['outcome_72'])

    # 判断数列是否只包含0，1
    def judge_missingness_indicator(self, data):
        for item in data:
            if item == 1 or item == 0:
                continue
            else:
                return False
        return True
