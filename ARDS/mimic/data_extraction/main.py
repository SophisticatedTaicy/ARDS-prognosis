import os
import threading

from pandas import DataFrame

from ARDS.eicu.data_extraction import query_sql
from ARDS.mimic.data_extraction.mimiciv_ards_query import Query
from filter.common import save_file, init_dict, filter_invalid_dynamic_items
from filter.param import result_header

base_path = os.path.abspath(os.path.dirname(os.getcwd()))


class MyThread(threading.Thread):
    def __init__(self, name, data, database_type, save=False):
        """
        @param name: 该部分数据的最先和最后一个记录id
        @param data: 该部分数据
        @param database_type:数据库类型
        @param save: 是否保存
        """
        super(MyThread, self).__init__(name=name)
        self.name = name
        self.data = data
        self.type = database_type
        self.save = save
        self.is_first = True

    def run(self):
        print('%s区间数据在运行中-----------' % str(self.name))
        for stay_id in self.data:
            query = Query(self.type)
            mimic_a = MIMIC_Acquisition(issave=self.save, type=self.type)
            mimic_a.get_ards_data_by_stay_id(stay_id, query)


class MIMIC_Acquisition:
    def __init__(self, type='3', issave=False, mutilprocess=False, path=base_path):
        self.type = type
        self.issave = issave
        # 是否多线程运行
        self.mutilprocess = mutilprocess
        self.path = os.path.join(path, 'mimic' + str(type), 'csvfiles')

    # 查找满足呼吸衰竭和呼吸机支持的患者病案号和ICU病案号
    def get_ARDS_datas(self):
        """
        @param query: mimic数据查询
        1.有呼吸衰竭，但非心血管衰竭
        2.年龄18及以上
        3.呼吸机支持
        4.pao2/fio2<=300 持续八小时
        5.peep>=5
        @return:满足呼吸衰竭和呼吸机支持的病案号和ICU病案号
        """
        query = Query(self.type)
        stay_ids = query.extract_meet_respiratory_failure_and_ventilator_support_condition_stay_ids()
        stay_ids = [item[0] for item in stay_ids]
        if self.issave:
            save_file(DataFrame(stay_ids), path=self.path, filename='stayids_1207', header=['stay_id'])
        split_List = []
        # 取出数据重复项
        id_len = len(stay_ids)
        if self.mutilprocess:
            # 分线程运行
            for i in range(10):
                split_List.append(stay_ids[i * int(id_len / 10): i * int(id_len / 10) + int(id_len / 10)])
            split_List.append(stay_ids[10 * int(id_len / 10):])
            for data in split_List:
                name = str(data[0]) + str('--') + str(data[-1])
                MyThread(name, data=data, database_type=self.type, save=self.issave).start()
        else:
            for stay_id in stay_ids:
                query = Query(type=self.type)
                self.get_ards_data_by_stay_id(stay_id, query)

    def get_ards_data_by_stay_id(self, stay_id, query=Query()):
        is_ards, identification, severity = query.filter_stay_id_by_p_f_ratio_and_peep(stay_id)
        # 数据满足氧合指数和呼吸末正压条件
        if is_ards:
            isvalid = query.invalid_data_filter(stay_id, identification)
            if isvalid:
                enrollment = identification + 1440
                header = init_dict(result_header)
                # 获取当前患者的静态数据信息，如用药信息、入院诊断以及年龄等
                query.get_static_data_by_stay_id(stay_id, header)
                # 获取当前住院期间的所有动态指标数据
                dynamic = query.get_dynamic_data_by_hadm_id_and_stay_id(stay_id, identification, enrollment)
                # 将数据转换为标准指标数据
                dynamic_format = query.standardize_dynamic_data(dynamic)
                print('patientunitstayid : %s  before dynamic item len : %s' % (stay_id, len(dynamic_format)))
                result_list = filter_invalid_dynamic_items(dynamic_format)
                print('after dynamic item len : %s' % len(result_list))
                dynamic_format = filter_invalid_dynamic_items(dynamic_format)
                result_dict, dynamic_dict = query_sql.compute_dynamic(stay_id, dynamic_format, header)
                # 获取当前患者的额外数据信息，如28天预后，24小时后预后以及住院天数等信息
                # outcome, status_28, detail, unitstay, hospitalstay, outcome_48, outcome_72 = query.access_ards_severity_after_enrollment(
                #     stay_id, enrollment)
                # result_dict['outcome'] = outcome
                # result_dict['detail'] = detail
                # result_dict['unit'] = unitstay
                # result_dict['hospital'] = hospitalstay
                # result_dict['status_28'] = status_28
                # result_dict['severity'] = severity
                # result_dict['identification'] = identification
                # result_dict['enrollment'] = enrollment
                # result_dict['outcome_48'] = outcome_48
                # result_dict['outcome_72'] = outcome_72
                # result_dict = DataFrame([result_dict])
                if self.issave:
                    # save_file(file=result_dict, path=self.path, filename='result_1207', mode='a')
                    print(dynamic_dict)
                    save_file(file=DataFrame([dynamic_dict]), path=self.path, filename='dynamic_0203', mode='a')


if __name__ == '__main__':
    mimic_a = MIMIC_Acquisition(type='4', mutilprocess=True, issave=True)
    mimic_a.get_ARDS_datas()

    # datas = read_file(filename='result', path=mimic_a.path)
    # processer = Processing()
    # processer.format_ards_data('3')
