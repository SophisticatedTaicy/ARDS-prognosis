import json

import numpy as np
import pandas as pd
import psycopg2
from pandas import DataFrame
from ARDS.data_solution.param import eicu_path, nursec_key_names, lab_key_names, respi_key_names, \
    vitalaperiodic_names, mimic3_path, mimic4_path, physical_key_names, vitalperiodic_key_names, mimic_dynamic_list
from filter.param import *


class solution:
    def __init__(self, type='EICU'):
        self.type = type
        self.rows = 0
        self.result = {}
        self.save = False

    # 按照数据表格指定列对数据分布进行统计
    def analysis_data_by_dimension(self, path, column, show_items=None, save=False, key_names=None, type=None):
        '''
        :param data: 数据表
        :param column: 需要统计的数据列名
        :param show_items: 统计的内容项数
        :param save: 是否保存
        :return:
        '''
        self.result = {}
        self.save = save
        if type:
            self.type = type
        global data
        if self.type == 'EICU':
            data = pd.read_csv(eicu_path + path)
        elif self.type == 'MIMIC3':
            data = pd.read_csv(mimic3_path + path)
        elif self.type == 'MIMIC4':
            data = pd.read_csv(mimic4_path + path)
        self.rows = data.shape[0]
        if len(column) == 1:
            if column == ['DOB']:
                new_data = data['DOB']
                data = pd.Series([item[:4] for item in new_data])
            else:
                data = data[column[0]]
            if key_names:
                # data = data.value_counts().to_dict()
                data = data.value_counts().to_dict()
                result = self.compute_num_info_for_single_column(data, key_names=key_names)
            else:
                result = {}
                non_num = len([item for item in data if pd.isnull(item)])
                non_rate = format(non_num / self.rows, '.2%')
                if len(data.value_counts()) < show_items:
                    column_detail = data.value_counts()
                else:
                    column_detail = data.value_counts()[:show_items]
                result[column[0]] = column[0], column_detail.to_json(force_ascii=False), non_num, non_rate
        else:
            result = self.summary_data_by_column_data(data, columns=column, show_num=show_items)
        if self.save:
            self.save_file(data=result)
            print('path : %s data saved!' % path)

    def summary_data_by_column_data(self, data, columns, show_num=False):
        '''
        :param data: 多个列所有数据信息 dataframe
        :param show_num: 需要展示的子项个数
        :return:
        '''
        result = {}
        for column in columns:
            non_num = len([item for item in data[column] if pd.isnull(item)])
            non_rate = format(non_num / self.rows, '.2%')
            column_detail = data[column].value_counts()[:show_num]
            result[column] = column, column_detail.to_json(force_ascii=False), non_num, non_rate
        return result

    # 计算具体项的数量信息
    def compute_num_info_for_single_column(self, data, key_names):
        '''
        data:当前列的数据信息,不同项以及对应数量
        key_names：需要细分的数据项
        '''
        result = {}
        for key_name in key_names:
            key_dict = {}
            count = 0
            for key, value in data.items():
                if key_name in key.lower():
                    if len(key_dict) < 5:
                        key_dict[key] = value
                    count += value
            # 存储每种疾病大类名称，出现详细项数以及对应数量以及当前疾病的总数量
            result[key_name] = key_name, key_dict, count
        return result

        # 计算列表中不同项，以及相对应的数量

    def count_item_in_list(self, item_list):
        set_list = set(item_list[:0])
        result = {}
        for item in set_list:
            item_count = sum([it[1] for it in item_list if item == it[0]])
            result[item] = item_count
        return result

    def save_file(self, data=None):
        if type(data) == np.ndarray or isinstance(data, list):
            DataFrame(data).to_csv('value_info_1.csv', mode='a', header=False, index=False)
        elif type(data) == np.typeDict:
            DataFrame(list(data.items())).T.to_csv('value_info_1.csv', mode='a', header=False, index=False)
        else:
            DataFrame(data).T.to_csv('value_info_1.csv', mode='a', header=False, index=False)

    def compute_dynamic_feature_datas(self, data, labnamelist, mark, save=False):
        '''
        data:动态数据详细：数据内容，数量
        labnamelist：所有动态指标名称
        mark：实验检查表
        返回：mark_feature,{data_item,count},nan,nan_rate
        '''
        lab_feature_infos = []
        for feature in labnamelist:
            # 将当前特征值按照频次,特征项，数值排序
            current_features = [item for item in data if feature.lower() in item[0].lower()]
            nan = sum([item[1] for item in current_features if item[0] is None])
            count = sum([item[1] for item in current_features])
            nan_rate = 0
            if count:
                nan_rate = format(nan / count, '.2f')
            # 按照频率排序
            current_features = sorted(current_features, key=lambda x: x[1], reverse=True)
            item_dict = {}
            # 选最高频的前五项
            for i in range(5):
                if i < len(current_features):
                    item_dict[str(current_features[i][0])] = current_features[i][1]
                else:
                    break
            if item_dict != {}:
                current_feature_Info = mark + feature, json.dumps(item_dict), count, nan, nan_rate
                lab_feature_infos.append(current_feature_Info)
                print(lab_feature_infos)
        if save and lab_feature_infos != []:
            self.save_file(lab_feature_infos)


class Postgres_server:
    def __init__(self, type='EICU'):
        self.database = EICU_DATABASE
        self.user = USER
        self.password = PASSWORD
        self.host = HOST
        self.port = PORT
        self.search_path = EICU_SEARCH_PATH
        # 更改数据库连接配置
        if type:
            if type == 'MIMIC3':
                self.database = MIMIC3_DATABASE
                self.search_path = MIMIC3_SEARCH_PATH
            elif type == 'MIMIC4':
                self.database = MIMIC4_DATABASE
                self.search_path = MIMIC4_SEARCH_PATH
        self.connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port)

    def server_execute_and_close(self, sql):
        cursor = self.connection.cursor()
        sql_default = "set search_path to " + self.search_path + ";"
        cursor.execute(sql_default)
        cursor.execute(sql)
        list = cursor.fetchall()
        cursor.close()
        return list

    def eICU_start(self):
        self.extract_lab_features_frequency()
        self.extract_nursecharting_features_frequency()
        self.extract_respiratoryCharting_features_frequency()
        self.extract_physicalexam_features_frequency()
        self.extract_vitalaperiodic_frequency()
        self.extract_apacheapsvar_frequency()
        self.extract_vitalperiod_frequency()
        self.connection.close()

    # 计算EICU数据库中动态数据分布
    def extract_lab_features_frequency(self):
        sql = " select distinct labresultoffset, labname, cast(labresult as float), labmeasurenamesystem as unit, count(*) " \
              " from lab " \
              " where labname in ( " \
              "      'bedside glucose', " \
              "      'potassium', " \
              "      'sodium', " \
              "      'glucose', " \
              "      'creatinine', " \
              "      'BUN', " \
              "      'calcium', " \
              "      'bicarbonate', " \
              "      'platelets x 1000', " \
              "      'WBC x 1000', " \
              "      'magnesium', " \
              "      '-eos', " \
              "      '-basos', " \
              "      'albumin', " \
              "      'AST (SGOT)', " \
              "      'ALT (SGPT)', " \
              "      'total bilirubin', " \
              "      'paO2', " \
              "      'paCO2', " \
              "      'pH', " \
              "      'PT - INR', " \
              "      'HCO3', " \
              "      'FiO2', " \
              "      'Base Excess', " \
              "      'PTT', " \
              "      'lactate', " \
              "      'Total CO2', " \
              "      'ionized calcium', " \
              "      'Temperature', " \
              "      'PEEP', " \
              "      'Methemoglobin', " \
              "      'Carboxyhemoglobin', " \
              "      'Oxyhemoglobin', " \
              "      'TV', " \
              "      'direct bilirubin', " \
              "      '-bands', " \
              "      'Respiratory Rate') " \
              "  group by  labresultoffset,labname, labresult,labmeasurenamesystem" \
              "  order by count(*)"
        lablist = self.server_execute_and_close(sql)
        result = []
        self.format_column_data(lablist, result)
        solution.compute_dynamic_feature_datas(result, lab_key_names, 'lab_', save=True)

    def extract_nursecharting_features_frequency(self):
        sql = " select distinct nursingchartoffset, nursingchartcelltypevalname, nursingchartvalue, nursingchartcelltypevallabel as label, count(*) " \
              " from nursecharting " \
              " where nursingchartcelltypevalname in ( " \
              "     'Heart Rate', " \
              "     'Respiratory Rate', " \
              "     'Non-Invasive BP Diastolic', " \
              "     'Non-Invasive BP Systolic', " \
              "     'Non-Invasive BP Mean', " \
              "     'Temperature (C)', " \
              "     'Temperature (F)', " \
              "     'Invasive BP Diastolic', " \
              "     'Invasive BP Systolic', " \
              "     'GCS Total', " \
              "     'Invasive BP Mean', " \
              "     'Eyes', " \
              "     'Motor', " \
              "     'Verbal', " \
              "     'Bedside Glucose', " \
              "     'CVP', " \
              "     'PA Diastolic', " \
              "     'PA Systolic', " \
              "     'PA Mean', " \
              "     'End Tidal CO2') " \
              " group by nursingchartoffset, nursingchartcelltypevalname, nursingchartvalue, nursingchartcelltypevallabel " \
              " order by count(*); "
        nurseclist = self.server_execute_and_close(sql)
        result = []
        self.format_column_data(nurseclist, result)
        solution.compute_dynamic_feature_datas(result, nursec_key_names, 'nurse_', save=True)

    def extract_respiratoryCharting_features_frequency(self):
        sql = " select distinct respchartentryoffset,respchartvaluelabel,respchartvalue,respcharttypecat, count(*) " \
              " from respiratorycharting " \
              " where respchartvaluelabel in (" \
              " 'FiO2'," \
              " 'PEEP'," \
              " 'Tidal Volume(set)', " \
              " 'TV / kg IBW', " \
              " 'Mean Airway Pressure', " \
              " 'Peak Insp. Pressure', " \
              " 'SaO2', " \
              " 'Plateau Pressure', " \
              " 'FIO2( %)', " \
              " 'PEEP/CPAP', " \
              " 'Tidal Volume Observed(VT)'," \
              " 'ETCO2'," \
              " 'Adult Con Pt/Vent SpO2'," \
              " 'NIV Pt/Vent SpO2_5'," \
              " 'Tidal Volume, Delivered')" \
              " group by respchartentryoffset,respchartvaluelabel,respchartvalue,respcharttypecat" \
              " order by count(*);"
        resclist = self.server_execute_and_close(sql)
        result = []
        self.format_column_data(resclist, result)
        solution.compute_dynamic_feature_datas(result, respi_key_names, 'respiratory_', save=True)

    # 将数据格式化为项内容：数量
    def format_column_data(self, data, result, label=None):
        '''
        data:list,内容包含测量项的时间、数值、单位、名称以及数量等详细信息
        result:格式化存储的数据
        label:列名称
        '''
        for item in data:
            value = item[-1]
            key = item[:-1]
            if label:
                key = key + tuple([label])
            result.append((str(key), value))

    def extract_physicalexam_features_frequency(self):
        sql = " select distinct physicalexamoffset, physicalexamvalue, physicalexamtext, '', count(*) " \
              " from physicalexam " \
              " where physicalexamvalue like 'CVP' " \
              " group by physicalexamoffset, physicalexamvalue, physicalexamtext " \
              " order by count(*); "
        phylist = self.server_execute_and_close(sql)
        result = []
        self.format_column_data(phylist, result)
        solution.compute_dynamic_feature_datas(result, physical_key_names, 'physical_', save=True)

    def extract_vitalaperiodic_frequency(self):
        vitalsperiod_list = []
        sql = " select  observationoffset, noninvasivediastolic, count(*)" \
              " from vitalaperiodic" \
              " group by observationoffset, noninvasivediastolic" \
              " order by count(*) desc;"
        NI_BPD = self.server_execute_and_close(sql)
        self.format_column_data(NI_BPD, vitalsperiod_list, label='noninvasivediastolic', )
        sql = " select  observationoffset,noninvasivesystolic, count(*)" \
              " from vitalaperiodic" \
              " group by observationoffset,noninvasivesystolic" \
              " order by count(*) desc;"
        NI_BPS = self.server_execute_and_close(sql)
        self.format_column_data(NI_BPS, vitalsperiod_list, label='noninvasivesystolic', )
        sql = " select observationoffset, noninvasivemean, count(*)" \
              " from vitalaperiodic" \
              " group by observationoffset,noninvasivemean" \
              " order by count(*) desc;"
        NI_BPM = self.server_execute_and_close(sql)
        self.format_column_data(NI_BPM, vitalsperiod_list, label='noninvasivemean', )
        solution.compute_dynamic_feature_datas(vitalsperiod_list, vitalaperiodic_names, 'vitalaperio_', save=True)

    def extract_apacheapsvar_frequency(self):
        sql = "select hematocrit, count(*) " \
              "from apacheapsvar " \
              "group by hematocrit " \
              "order by count(*) desc; "
        apa_list = self.server_execute_and_close(sql)
        result = []
        self.format_column_data(apa_list, result, label='hematocrit', )
        solution.compute_dynamic_feature_datas(result, ['hematocrit'], 'apacheapsvar_')

    def extract_vitalperiod_frequency(self):
        result = []
        sql = " select observationoffset,cvp, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset,cvp " \
              " order by count(*); "
        cvp_list = self.server_execute_and_close(sql)
        self.format_column_data(cvp_list, result, label='cvp', )
        sql = " select observationoffset,etco2, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,etco2 " \
              " order by count(*); "
        etco2_list = self.server_execute_and_close(sql)
        self.format_column_data(etco2_list, result, label='etco2', )
        sql = " select observationoffset ,heartrate, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,heartrate " \
              " order by count(*); "
        heartrate_list = self.server_execute_and_close(sql)
        self.format_column_data(heartrate_list, result, label='heartrate', )
        sql = " select observationoffset ,padiastolic, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,padiastolic " \
              " order by count(*); "
        padiastolic_list = self.server_execute_and_close(sql)
        self.format_column_data(padiastolic_list, result, label='padiastolic', )
        sql = " select observationoffset ,pamean, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,pamean " \
              " order by count(*); "
        pamean_list = self.server_execute_and_close(sql)
        self.format_column_data(pamean_list, result, label='pamean', )
        sql = " select observationoffset ,pasystolic, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,pasystolic " \
              " order by count(*); "
        pasystolic_list = self.server_execute_and_close(sql)
        self.format_column_data(pasystolic_list, result, label='pasystolic', )
        sql = " select observationoffset ,respiration, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,respiration " \
              " order by count(*); "
        respiration_list = self.server_execute_and_close(sql)
        self.format_column_data(respiration_list, result, label='respiration', )
        sql = " select observationoffset ,sao2, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,sao2 " \
              " order by count(*); "
        sao2_list = self.server_execute_and_close(sql)
        self.format_column_data(sao2_list, result, label='sao2', )
        sql = " select observationoffset ,temperature, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,temperature " \
              " order by count(*); "
        temperature_list = self.server_execute_and_close(sql)
        self.format_column_data(temperature_list, result, label='temperature', )
        sql = " select observationoffset ,systemicdiastolic, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,systemicdiastolic " \
              " order by count(*); "
        systemicdiastolic_list = self.server_execute_and_close(sql)
        self.format_column_data(systemicdiastolic_list, result, label='systemicdiastolic', )
        sql = " select observationoffset ,systemicmean, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,systemicmean " \
              " order by count(*); "
        systemicmean_list = self.server_execute_and_close(sql)
        self.format_column_data(systemicmean_list, result, label='systemicmean', )
        sql = " select observationoffset ,systemicsystolic, count(*) " \
              " from vitalperiodic " \
              " group by observationoffset ,systemicsystolic " \
              " order by count(*); "
        systemicsystolic_list = self.server_execute_and_close(sql)
        self.format_column_data(systemicsystolic_list, result, label='systemicsystolic', )
        solution.compute_dynamic_feature_datas(result, mark='vitalperiod_', labnamelist=vitalperiodic_key_names,
                                               save=True)

    def compute_MIMIC_features_frequency(self, type):
        self.__init__(type)
        sql = " select * " \
              " from dynamic_features_view; "
        mimic_list = self.server_execute_and_close(sql)
        result = []
        self.format_column_data(mimic_list, result)
        solution.compute_dynamic_feature_datas(result, mark=type + '_', labnamelist=mimic_dynamic_list, save=True)

    def extract_drug_use(self, save=False):
        sql = " select infusionoffset, drugname, drugrate, drugamount, volumeoffluid, count(*) " \
              " from infusiondrug " \
              " where lower(drugname) like any (array['%dobutamine%', '%dopamine%', '%epinephrine%', '%heparin%', '%milrinone%'," \
              "                                       '%norepinephrine%', '%phenylephrine%', '%vasopressin%']) " \
              " group by infusionoffset, drugname, drugrate, drugamount, volumeoffluid " \
              " order by count(*); "
        result = []
        infu_drug = self.server_execute_and_close(sql)
        self.format_column_data(infu_drug, result)
        sql = " select drugenteredoffset, drugname, drugdosage, drugunit, drugadmitfrequency, count(*)" \
              "  from admissiondrug " \
              "  where lower(drugname) like '%warfarin%' " \
              "  group by drugenteredoffset, drugname, drugdosage, drugunit, drugadmitfrequency " \
              "  order by count(*); "
        admi_drug = self.server_execute_and_close(sql)
        self.format_column_data(admi_drug, result)
        solution.compute_dynamic_feature_datas(result, drug_list, mark='eICU_', save=save)
        result = []
        self.__init__(type='MIMIC3')
        sql = " select startdate, drug_type, drug, prod_strength, dose_unit_rx, dose_val_rx,route, count(*)" \
              " from prescriptions" \
              " where lower(drug) like any (array ['%warfarin%', '%dobutamine%', '%dopamine%', '%epinephrine%', '%heparin%'," \
              " '%milrinone%', '%norepinephrine%', '%phenylephrine%', '%vasopressin%', '%vasopressor%'])" \
              " group by startdate, drug_type, drug, prod_strength, dose_unit_rx, dose_val_rx, route;"
        mimic3_drug = self.server_execute_and_close(sql)
        self.format_column_data(mimic3_drug, result)
        solution.compute_dynamic_feature_datas(result, drug_list, mark='MIMIC3_', save=save)
        self.__init__(type='MIMIC4')
        sql = " select starttime, drug_type, drug, prod_strength, dose_unit_rx, dose_val_rx, doses_per_24_hrs, route, count(*)" \
              " from prescriptions" \
              " where lower(drug) like any (array ['%warfarin%', '%dobutamine%', '%dopamine%', '%epinephrine%', '%heparin%'," \
              " '%milrinone%', '%norepinephrine%', '%phenylephrine%', '%vasopressin%', '%vasopressor%'])" \
              " group by starttime, drug_type, drug, prod_strength, dose_unit_rx, dose_val_rx, doses_per_24_hrs, route;"
        mimic4_drug = self.server_execute_and_close(sql)
        self.format_column_data(mimic4_drug, result)
        solution.compute_dynamic_feature_datas(result, drug_list, mark='MIMIC4_', save=save)


if __name__ == '__main__':
    solution = solution()
    # 患者基本信息
    solution.analysis_data_by_dimension('patient.csv', column=['gender', 'age', 'hospitaladmitsource'], show_items=4,
                                        save=True)
    solution.analysis_data_by_dimension(path='PATIENTS.csv', type='MIMIC3', column=['GENDER'], save=True)
    solution.analysis_data_by_dimension(path='PATIENTS.csv', type='MIMIC3', column=['DOB'], save=True, show_items=5)
    solution.analysis_data_by_dimension(path='ADMISSIONS.csv', type='MIMIC3', column=['ADMISSION_LOCATION'], save=True,
                                        show_items=5)
    solution.analysis_data_by_dimension(path='admissions.csv', type='MIMIC4', column=['admission_location'], save=True,
                                        show_items=5)
    solution.analysis_data_by_dimension(path='patients.csv', type='MIMIC4', column=['gender', 'anchor_age'], save=True,
                                        show_items=5)
    # 入院诊断
    solution.analysis_data_by_dimension('diagnosis.csv', column=['diagnosisstring'], key_names=diagnosis_key_names,
                                        save=True)
    solution.analysis_data_by_dimension(path='ADMISSIONS.csv', type='MIMIC3', column=['DIAGNOSIS'], save=True,
                                        key_names=diagnosis_key_names)
    solution.analysis_data_by_dimension(path='d_icd_diagnoses.csv', column=['long_title'], save=True, type='MIMIC4',
                                        key_names=diagnosis_key_names)
    # 入院用药
    solution.analysis_data_by_dimension('infusiondrug.csv', column=['drugname'], key_names=drug_list, save=False)
    solution.analysis_data_by_dimension('admissiondrug.csv', column=['drugname'], key_names=['warfarin'], save=True)
    solution.analysis_data_by_dimension(path='PRESCRIPTIONS.csv', type='MIMIC3', column=['DRUG'],
                                        key_names=drug_list + ['warfarin'], save=True)
    solution.analysis_data_by_dimension(path='PRESCRIPTIONS.csv', type='MIMIC4', column=['drug'],
                                        key_names=drug_list + ['warfarin'], save=True)
    server = Postgres_server()
    server.extract_drug_use(save=True)
    lab_list = server.extract_lab_features_frequency()
    dynamic = Postgres_server('MIMIC3')
    server.compute_MIMIC_features_frequency('MIMIC3')
    server.compute_MIMIC_features_frequency('MIMIC4')
    # 计算每项指标数据的最高频次数据，异常值个数，异常值占比
    solution.analysis_data_by_dimension(path='admissions.csv', column=['diagnosis'], key_names=diagnosis_key_names)
    solution.analysis_data_by_dimension(path='prescriptions.csv', column=['drug'], key_names=drug_list + 'warfarin')
    solution.analysis_data_by_dimension(path='icustays.csv', column=[''])
