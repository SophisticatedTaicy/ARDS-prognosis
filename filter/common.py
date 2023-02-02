import os
import time

__all__ = ['log_time']

import numpy as np
import pandas as pd
from pandas import DataFrame

merge_data_path = os.path.join(os.getcwd(), 'ARDS/combine/csvfileS')


def log_time(func):
    def wrapper(*args, **kw):
        import time
        start_time = time.time()
        print("\n[%30s] startTime: %s" % (
            (' ' + func.__name__).rjust(30, '>'), time_format(start_time),))
        try:
            return func(*args, **kw)
        finally:
            end_time = time.time()
            print("\n[%30s] endTime: %s,  cost: %s\n\n" % (
                (' ' + func.__name__).rjust(30, '<'), time_format(end_time), time_format(end_time - start_time)))

    def time_format(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        h = h % 24
        return '%02d:%02d:%02d' % (h, m, s)

    return wrapper


def standard_data_by_percentile(data):
    """
    @type 数据归一化处理，方法：使用80分位与20分位数的差对数列进行归一化
    @param data: 输入需要规格化的Dataframe数据
    @return 归一化后的数据
    """
    per_2 = np.percentile(data, 20)
    per_8 = np.percentile(data, 80)
    dif = per_8 - per_2
    if dif:
        new_data = np.round(DataFrame(data) / dif, 4)
    else:
        new_data = data
    return new_data


def standard_data_by_white(data, data_t):
    """
    @type 数据归一化处理，方法：使用80分位与20分位数的差对数列进行归一化
    @param data: 输入需要规格化的Dataframe数据
    @return 归一化后的数据
    """
    mean = np.reshape(np.mean(data, axis=0), (1, -1))
    std = np.reshape(np.std(data, axis=0) + 1e-9, (1, -1))
    new_data = (data - mean) / std
    new_data_t = (data_t - mean) / std
    return new_data, new_data_t


def standard_data_by_percentile_t(data, data_te):
    """
    @type 数据归一化处理，方法：使用80分位与20分位数的差对数列进行归一化
    @param data: 输入需要规格化的Dataframe数据
    @return 归一化后的数据
    """
    per_2 = np.percentile(data, 20)
    per_8 = np.percentile(data, 80)
    dif = per_8 - per_2
    if dif:
        new_data = np.round(DataFrame(data) / dif, 4)
        new_data_t = np.round(DataFrame(data_te) / dif, 4)
    else:
        new_data = data
        new_data_t = data_te
    return new_data, new_data_t


# 保存csv文件
def save_file(file, filename, path, mode='w', header=None, index=False):
    """
    @param file: 需要保存的dataframe文件
    @param filename: 需要保存的文件名
    @param mode: 保存文件的模式，a追加，w写入
    @param data_type: 数据源:EICU/MIMIC3/MIMIC4
    @param header: 是否添加表头
    @param index: 是否添加默认序号
    """
    path = os.path.join(path, filename + '.csv')
    file.to_csv(path, mode=mode, header=header, index=index)


# 读取csv文件
def read_file(filename, path, sep=',', encoding='utf-8'):
    path = os.path.join(path, filename + '.csv')
    return pd.read_csv(path, sep=sep, encoding=encoding)


# 初始化字典数据
def init_dict(header_list):
    result_dict = {}
    for item in header_list:
        result_dict[item] = 0
    return result_dict


# 打印过程数据
def log_print(data):
    print('item : %s len is : %s ' % (data, len(data)))


def special_list(list, types):
    """
    @param list: 原始列表
    @param type: 需要自动生成的列表的别名,changerate,variance,median,mean,std
    """
    new_list = []
    for type in types:
        for item in list:
            new_list.append(item + '_' + type)
    return new_list




def concat_array(list):
    new = []
    for sublist in list:
        new.extend(sublist)
    return new


def init_component(insert_num):
    mean_tpr = []
    mean_fpr = np.linspace(0, 1, insert_num)
    tprs = []
    t_mean_tpr = []
    t_mean_fpr = np.linspace(0, 1, insert_num)
    t_tprs = []
    return mean_tpr, mean_fpr, tprs, t_mean_tpr, t_mean_fpr, t_tprs


def data_split_and_combine(eicu, mimic3, mimic4, columns):
    eicu_new = DataFrame()
    mimic3_new = DataFrame()
    mimic4_new = DataFrame()
    for column in columns:
        eicu_new = pd.concat([eicu_new, eicu[column]], axis=1)
        mimic3_new = pd.concat([mimic3_new, mimic3[column]], axis=1)
        mimic4_new = pd.concat([mimic4_new, mimic4[column]], axis=1)
    eicu_label = np.array(eicu_new['outcome'])
    mimic3_label = np.array(mimic3_new['outcome'])
    mimic4_label = np.array(mimic4_new['outcome'])
    del eicu_new['outcome']
    del mimic3_new['outcome']
    del mimic4_new['outcome']
    eicu_data = np.array(eicu_new)
    mimic3_data = np.array(mimic3_new)
    mimic4_data = np.array(mimic4_new)
    return eicu_label, mimic3_label, mimic4_label, eicu_data, mimic3_data, mimic4_data


def data_merge(eicu, mimic3, mimic4, columns):
    eicu_new = DataFrame()
    mimic3_new = DataFrame()
    mimic4_new = DataFrame()
    # get valid column data
    for column in columns:
        eicu_new = pd.concat([eicu_new, eicu[column]], axis=1)
        mimic3_new = pd.concat([mimic3_new, mimic3[column]], axis=1)
        mimic4_new = pd.concat([mimic4_new, mimic4[column]], axis=1)
    save_file(eicu_new, 'merge_eicu', merge_data_path, header=columns)
    save_file(mimic3_new, 'merge_mimic3', merge_data_path, header=columns)
    save_file(mimic4_new, 'merge_mimic4', merge_data_path, header=columns)
    merge = pd.concat([eicu_new, mimic3_new])
    merge = pd.concat([merge, mimic4_new])
    save_file(merge, 'merge_data', merge_data_path, header=columns)
    labels = merge['outcome']
    del merge['outcome']
    merge = np.array(merge)
    labels = np.array(labels)
    return merge, labels


# 判断划分数据时，是否包含三种标签 ，若是则需要重新划分
def judge_label_balance(label_tr, label_te):
    if len(set(label_tr)) == 3 and len(set(label_te)) == 3:
        return True
    else:
        return False


def format_label(data, label):
    new_label = []
    for item in data:
        if item == label:
            new_label.append(1)
        else:
            new_label.append(0)
    new_label = np.array(new_label)
    return new_label


def filter_invalid_dynamic_items(data):
    dynamic_names = ['albumin', 'ALT', 'AST', 'bands', 'Base Excess', 'basos', 'bicarbonate', 'bilirubin', 'BUN',
                     'calcium', 'CO2', 'creatinine', 'eos', 'FIO2', 'glucose', 'Hemoglobin', 'INR', 'ionized calcium',
                     'lactate', 'magnesium', 'paCO2', 'paO2', 'P/F ratio', 'PEEP', 'pH', 'platelets', 'potassium',
                     'PTT', 'PIP', 'sodium', 'Temperature', 'WBC', 'Mean Airway Pressure', 'Plateau Pressure', 'SaO2',
                     'SpO2', 'TV', 'CVP', 'ETCO2', 'diastolic_PAP', 'mean_PAP', 'systolic_PAP', 'Eyes', 'GCS', 'Motor',
                     'Verbal', 'Heart Rate', 'I_BP_diastolic', 'I_BP_mean', 'I_BP_systolic', 'NI_BP_diastolic',
                     'NI_BP_mean', 'NI_BP_systolic', 'Respiratory Rate', 'hematocrit']
    dynamic_min = [0, 7, 10, 0, 0, 0, 5, 3.4, 0, 0, 10, 0, 0, 20, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 6.5, 0, 0, 20, -1000,
                   95, 30, 0, 5, 4, 80, 80, -1000, -5, 3.4, 1, 1, 1, 0, 0, 0, 0, 30, 30, 30, 30, 30, 30, 30, 1, 14]
    dynamic_max = [7, 1000, 1000, 70, 1, 1, 40, 20, 115, 50, 45, 8, 8, 100, 500, 16, 15, 10, 15, 10, 100, 650, 650,
                   25, 7.7, 1000, 12, 160, 1000, 215, 45, 60, 30, 50, 100, 100, 1000, 100, 20, 40, 60, 80, 4, 15, 6, 5,
                   175, 150, 150, 250, 150, 150, 250, 40, 55]
    new_data = []
    for name, max, min in zip(dynamic_names, dynamic_max, dynamic_min):
        for item in data:
            if item[0] == name:
                if item[1] > max or item[1] < min:
                    continue
                else:
                    new_data.append(item)
                if name == 'P/F ratio':
                    print(item)
                    # print(new_data)
    # print(new_data)
    return new_data


if __name__ == '__main__':
    @log_time
    def start():
        for i in range(0, 3):
            print(i, flush=True)
            time.sleep(0.3)


    start()
