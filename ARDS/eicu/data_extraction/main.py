# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import threading

import numpy as np
from pandas import DataFrame

import query_sql
from ARDS.data_process.process import Processing
from ARDS.eicu.data_extraction.query_sql import Query
from filter.common import save_file, init_dict, log_print, read_file
from filter.param import result_header

path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'csvfiles')


class MyThread(threading.Thread):
    def __init__(self, data, save=False):
        super(MyThread, self).__init__()
        self.data = data
        self.is_first = True
        self.save = save

    def run(self):
        print('%s区间数据在运行中-----------' % str(self.name))
        for stay_id in self.data:
            eICU_Acquisition(self.save).get_ards_data_by_stay_id(stay_id)


class eICU_Acquisition:
    def __init__(self, issave=False, mutilprocess=False, process=0):
        self.issave = issave
        # 是否多线程运行
        self.mutilprocess = mutilprocess
        self.process = process

    # 查找满足呼吸衰竭和呼吸机支持的患者病案号和ICU病案号
    def get_ARDS_datas(self):
        """
        @param query: eICU数据查询
        @return:ARDS患者相关指标数据
        """
        query = Query()
        # 根据患者年龄、呼吸机支持、呼吸衰竭筛选患者住院记录
        age_ids = query.filter_with_age()
        vent_ids = query.filter_with_vent_support()
        respiratory_ids = query.filter_with_respiratory_failure_without_congestive_heart_failure()
        # 取三者交集，并去除重复值
        ids = list(set(age_ids).intersection(vent_ids, respiratory_ids))
        print('patient require age, ven, respiratory is :%s' % len(ids))
        if self.issave:
            save_file(DataFrame(ids), path=path, filename='stayids_1207', header=['patientunitstays'])
        split_List = []
        # 取出数据重复项
        id_len = len(ids)
        if self.mutilprocess:
            # 分线程运行
            for i in range(self.process):
                split_List.append(
                    ids[i * int(id_len / self.process): i * int(id_len / self.process) + int(id_len / self.process)])
            split_List.append(ids[self.process * int(id_len / self.process):])
            for data in split_List:
                MyThread(data=data, save=self.issave).start()
        else:
            p_f = 0
            is_peep = 0
            invalid = 0
            result = read_file(filename='result_1228', path=path).iloc[:, 0]
            for stay_id in ids:
                if stay_id in result:
                    continue
                else:
                    pf, peep, valid = self.get_ards_data_by_stay_id(stay_id)
                    p_f += pf
                    is_peep += peep
                    invalid += valid
            print('p_f : %s peep : %s invalid : %s ' % (p_f, is_peep, invalid))

    # 检验住院记录是否满足ARDS定义,若满足,则提取其完整信息
    def get_ards_data_by_stay_id(self, stay_id):
        query = Query()
        is_ards, identification, severity = query.filter_with_p_f(stay_id)
        # 数据满足氧合指数和呼吸末正压条件
        pf = 0
        peep = 0
        valid = 0
        if is_ards:
            pf = 1
            isrequired_peep = query.filter_with_peep(stay_id, identification)
            if isrequired_peep:
                peep = 1
                enrollment = identification + 1440
                isvalid = query.isvalid_data(stay_id, identification)
                # 获取该住院记录的ARDS信息
                if isvalid:
                    valid = 1
                    header = init_dict(result_header)
                    dynamic = query.filter_dynamic(stay_id, identification, enrollment)
                    query.filter_static(stay_id, identification, enrollment, header)
                    outcome, detail, unitstay, hospitalstay, status_28, outcome_48, outcome_72 = query.access_outcome(
                        stay_id, enrollment)
                    header['outcome'] = outcome
                    header['detail'] = detail
                    header['unit'] = unitstay
                    header['hospital'] = hospitalstay
                    header['status_28'] = status_28
                    header['severity'] = severity
                    header['identification'] = identification
                    header['enrollment'] = enrollment
                    header['outcome_48'] = outcome_48
                    header['outcome_72'] = outcome_72
                    # 计算动态数据的中位数、方差以及变化率
                    result_dict, dynamic_dict = query_sql.compute_dynamic(stay_id, dynamic, header)
                    if self.issave:
                        save_file(DataFrame([result_dict]), filename='result_1207', path=path, mode='a')
                        save_file(DataFrame([dynamic_dict]), filename='dynamic_1207', path=path, mode='a')
        return pf, peep, valid

    def fill_invalid_data_with_average(self, data):
        data = np.array(data)
        log_print(data)
        sum = 0
        non_zero_sum = 0
        # BMI以及apache非法值使用均值填充
        for j in range(39, 42):
            average = 0
            # BMI小于10为异常值
            for i in range(len(data)):
                # 计算当前列非零数值的均值
                item = float(data[i][j])
                # 计算非0和非-项的总和与数量
                if item > 10:
                    sum += item
                    non_zero_sum += 1
            if non_zero_sum > 0:
                average = round(sum / non_zero_sum, 3)
                # 使用均值填充不存在的数值项
                for i in range(len(data)):
                    item = float(data[i][j])
                    if np.isnan(item) or item <= 10:
                        data[i][j] = average
        for j in range(42, 207):
            sum = 0
            non_zero_sum = 0
            for i in range(len(data)):
                # 计算当前列非零数值的均值
                item = data[i][j]
                # 计算非0和非-项的总和与数量
                if item != '-' and float(item) > 0:
                    sum += float(item)
                    non_zero_sum += 1
            if non_zero_sum > 0:
                average = round(sum / non_zero_sum, 3)
                # 使用均值填充不存在的数值项
                for i in range(len(data)):
                    item = data[i][j]
                    if item == '-' or float(item) <= 0:
                        data[i][j] = average
            else:
                for i in range(len(data)):
                    item = data[i][j]
                    if item == '-' or float(item) <= 0:
                        data[i][j] = 0
        data = DataFrame(data)
        return data

    def fill__with_0(self, data):
        data = np.array(data)
        # BMI以及apache分数异常值填充
        for j in range(38, 42):
            # BMI小于10为异常值
            non_zero_sum = 0
            sum = 0
            for i in range(0, len(data)):
                # 计算当前列非零数值的均值
                item = float(data[i][j])
                # 计算非0和非-项的总和与数量
                if item > 10:
                    sum += item
                    non_zero_sum += 1
            if non_zero_sum > 0:
                # print('index %d average %f' % (j, sum / non_zero_sum))
                average = round(sum / non_zero_sum, 3)
                # 使用均值填充不存在的数值项
                for i in range(0, len(data)):
                    item = data[i][j]
                    if np.isnan(item):
                        data[i][j] = average
                    else:
                        if item <= 10:
                            data[i][j] = average
        for j in range(42, 207):
            for i in range(len(data)):
                # 计算当前列非零数值的均值
                item = data[i][j]
                # 计算非0和非-项的总和与数量
                if item == '-' or float(item) < 0:
                    data[i][j] = 0
        data = DataFrame(data)
        return data


if __name__ == '__main__':
    # eicu = eICU_Acquisition(mutilprocess=False, process=20, issave=False)
    # eicu.get_ARDS_datas()
    # processer = Processing()
    # processer.format_ards_data('4')
    query_sql.population_mean_value(0.5)
