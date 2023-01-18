import threading

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from filter.param import *

dynamic_dataframe = np.array(pd.read_csv('../result/0802_origin.csv', sep=',', encoding='utf-8', low_memory=False))
static_dataframe = np.array(pd.read_csv('../result/0801_feature_data.csv', sep=',', encoding='utf-8', low_memory=False))
iden_enroll_infos = np.array(pd.read_csv('../result/peep_filter.csv', sep=',', encoding='utf-8', low_memory=False))


class MyThread(threading.Thread):
    def __init__(self, name, data, frequency):
        super(MyThread, self).__init__(name=name)
        self.name = name
        self.data = data
        self.frequency = frequency

    def run(self):
        print('%s区间数据正在运行中。。。。。' % str(self.name))
        for id in self.data:
            extract_time_series_data_by_id(id, self.frequency, dynamic_dataframe, static_dataframe, iden_enroll_infos)


def init_data():
    header = {}
    dynamic = filter.param.time_series_header
    for item in dynamic:
        header[item] = -1
    return header


def extract_by_frequency(identification, frequency, data, result):
    '''
    根据指定的频率抽取患者确诊到入科时间内的时序数据
    :param identification: 患者ARDS确诊时间
    :param data: 患者确诊到入科时间内的所有指标的所有测量数据
    :param frequency: 提取数据的频率，如每30分钟采集一次或者15分钟采集一次，若采集的当前时刻没有数据，则以前面数据为准
    :param result: 当前患者采样数据
    :return: 当前患者按照指定频率采集的时序数据,时序数据为id,time,static,dynamic,status
    '''
    times = int(1440 / frequency)
    # 按照数组第三列，第一列，第二列的优先顺序排序
    arrSortedIndex = np.lexsort((data[:, 1], data[:, 0], data[:, 2]))
    data = data[arrSortedIndex, :]
    new_result = []
    for i in range(0, times):
        # 当前采样数据初始化为-1
        # 当前采样时间确定
        subsample_time = identification + i * frequency
        # 查到二维数组中，第三列数据小于subsample——time的下标
        max_index_list = np.argwhere(data[:, 2] <= subsample_time)[:, 0]
        # print('subsample_time : ' + str(subsample_time) + ' max_index_list : ' + str(
        #     max_index_list) + ' max index item : ' + str(data[max_index_list[-1]]))
        j = len(max_index_list) - 1
        # 找到当前采样时间对应的数据
        header = {}
        for i in range(0, len(result)):
            if result[i]['time'] == subsample_time:
                header = result[i]
                break
        if header == {}:
            print('没有当前采样时间的数据！')
            break
        # 遍历所有下标确定此次采样的数据
        while j >= 0:
            index = max_index_list[j]
            name = data[index][0]
            value = data[index][1]
            # 如果当前指标没有采样过,则采样
            if header[name] == -1:
                header[name] = float(value)
            j -= 1
        new_result.append(header)
    return new_result


def extract_time_series_data_by_id(id, frequency, dynamic_dataframe, static_dataframe, iden_enroll_infos):
    # 找到id在原数据的行数
    static_index = np.argwhere(static_dataframe == id)[0][0]
    # 找到id在确诊时间表的行数
    iden_index = np.argwhere(iden_enroll_infos == id)[0][0]
    identification = iden_enroll_infos[iden_index][1]
    # 找到患者住院期间所有测量指标数据下标
    dynamic_indexs = np.argwhere(dynamic_dataframe == id)[:, 0]
    # 找到指定患者的所有测量指标
    start = dynamic_indexs[0]
    end = dynamic_indexs[-1] + 1
    datas = dynamic_dataframe[start:end, 1:]
    result = []
    times = int(1440 / frequency)
    static_list = drug_list + diagnosis_abbrevation_list + person_info_list
    for i in range(0, times):
        subsample_time = identification + i * frequency
        header = init_data()
        header['id'] = id
        # 采样时间
        header['time'] = subsample_time
        # 静态数据采样
        for j in range(0, 41):
            header[static_list[j]] = static_dataframe[static_index][j + 1]
        # 患者最终状态采样
        header['status'] = static_dataframe[static_index][-5]
        result.append(header)
    id_results = extract_by_frequency(identification, frequency, datas, result)
    dataframe = DataFrame(id_results)
    dataframe.to_csv('result/time-series_60.csv', mode='a', encoding='utf-8', header=False, index=False)
    print('id : ' + str(id) + ' data has been writen done!')


if __name__ == '__main__':
    # 存储所有住院记录时序数据
    ids = static_dataframe[:, 0]
    # 设置五分钟去一次记录,总计抽取
    frequency = 60
    static_list = drug_list + diagnosis_abbrevation_list + person_info_list
    # 将数组切分多线程执行
    data_list = []
    # for id in ids:
    #     print(str(id))
    #     extract_time_series_data_by_id(id, frequency, dynamic_dataframe, static_dataframe, iden_enroll_infos)
    for i in range(0, 10):
        data_list.append(ids[i * 800:i * 800 + 800])
    data_list.append([ids[8000:-1]])
    # frequency:60--->25分钟
    for data in tqdm(data_list):
        # print(str(data))
        name = str(data[0]) + '-' + str(data[-1])
        MyThread(name=name, data=data, frequency=frequency).start()
