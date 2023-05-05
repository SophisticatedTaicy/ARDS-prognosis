import os
from math import sqrt

import pandas as pd
import numpy as np
import psycopg2
from pandas import DataFrame

import filter.param
from filter.common import filter_invalid_dynamic_items, filter_invalid_items_by_name
from filter.param import EICU_DATABASE, USER, PASSWORD, HOST, PORT, EICU_SEARCH_PATH, dynamic_item_list


class Query:
    # 数据库配置初始化
    def __init__(self):
        self.database = EICU_DATABASE
        self.user = USER
        self.password = PASSWORD
        self.host = HOST
        self.port = PORT
        self.search_path = EICU_SEARCH_PATH
        self.connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port)

    # @log_time
    def cursor_connection_and_close(self, sql):
        cursor = self.connection.cursor()
        default_sql = "set search_path to " + self.search_path + ";"
        cursor.execute(default_sql)
        cursor.execute(sql)
        list = cursor.fetchall()
        cursor.close()
        return list

    # filter data with respiratory failure
    def filter_with_respiratory_failure_without_congestive_heart_failure(self):
        sql = " select di.patientunitstayid from diagnosis as di" \
              " where di.icd9code like '%518.81%' " \
              " or di.icd9code like '%518.82%'" \
              " or di.icd9code like '%518.85%';"
        list = self.cursor_connection_and_close(sql)
        new_list = [item[0] for item in list]
        print('patient diagnosis respiratory and missing congestive heart failure : ' + str(len(list)))
        return new_list

    # 筛选入院是满足柏林定义或者入院后满足八小时氧合指数满足柏林定义的患者住院记录
    # @log_time
    def filter_with_p_f(self, unitid):
        """
        @param unitid: 患者住院记录id
        @return:
        is_ards：是否满足ARDS
        identification：确诊时间
        severity：严重性等级
        """
        # 先判断患者入院时，是否满足柏林定义
        sql = " select pao2, fio2" \
              " from apacheapsvar" \
              " where fio2 > 0" \
              "   and pao2 > 0" \
              "   and patientunitstayid = " + str(unitid) + \
              "   limit 1; "
        pa_fi = self.cursor_connection_and_close(sql)
        current_severity = 4
        if pa_fi:
            pa_fi = pa_fi[0]
            pao2 = pa_fi[0]
            fio2 = pa_fi[1]
            if fio2 > 1:
                fio2 /= 100
            p_f = pao2 / fio2
            # 患者入院时满足ARDS，严重级别需要按照入院前八小时内最差氧合指数确定
            if p_f <= 300:
                current_severity = 3
                if p_f <= 100:
                    current_severity = 1
                elif p_f <= 200:
                    current_severity = 2
        # 入院后8小时是否满足柏林定义
        sql = " select labresultoffset as time, labname as name, labresult as value " \
              " from lab " \
              " where labname in ('FiO2', 'paO2') " \
              "   and patientunitstayid = " + str(unitid)
        pa_fi = self.cursor_connection_and_close(sql)
        if pa_fi:
            pa_fi = np.array(pa_fi)
            flag, identification, severity = access_ards(pa_fi)
        else:
            flag, identification, severity = False, -1, -1
        if current_severity < 4:
            flag = True
            identification = 0
            if severity > 0:
                severity = min(current_severity, severity)
            else:
                severity = current_severity
        return flag, identification, severity

    # filter data with peep 判断在指定确诊时间内peep是否都>=5，
    # @log_time
    def filter_with_peep(self, id, identification):
        '''
        :param id: 满足呼吸衰竭且心血管衰竭，年龄不小于18，需要呼吸机支持的患者住院记录id
        :param identification: 患者满足连续八小时
        :return:True/False 当前患者的peep是否满足柏林定义
        '''
        # 只有peep,fio2和pao2数值在多张表中查找,在此处记录peep,fio2,pao2和p/f的动态数值
        # 确定确诊八小时内患者peep是否满足条件
        sql = " select labresultoffset as time, labresult as value " \
              " from lab " \
              " where labname = 'PEEP' " \
              "     and patientunitstayid =" + str(id)
        peep_list = self.cursor_connection_and_close(sql)
        peep_list = [item for item in peep_list if
                     item[1] and item[0] > identification - 480 and item[0] <= identification and item[1] < 5]
        if len(peep_list):
            return False
        sql = "   select respchartentryoffset as time, cast(respchartvalue as numeric) as value " \
              "   from respiratorycharting " \
              "   where respchartvaluelabel in ('PEEP', 'PEEP/CPAP') " \
              "   and patientunitstayid = " + str(id)
        peep_list = self.cursor_connection_and_close(sql)
        peep_list = [item for item in peep_list if
                     item[0] and item[0] > identification - 480 and item[0] <= identification and item[1] < 5]
        if len(peep_list):
            return False
        sql = "   select physicalexamoffset as time, cast(physicalexamtext as numeric) as value " \
              "   from physicalexam " \
              "   where physicalexamvalue = 'PEEP' " \
              "   and patientunitstayid = " + str(id)
        peep_list = self.cursor_connection_and_close(sql)
        peep_list = [item for item in peep_list if
                     item[0] and item[0] > identification - 480 and item[0] <= identification and item[1] < 5]
        # 判断在确诊时间前八小时，peep是否满足柏林定义
        # 将列表按照时间排序
        if len(peep_list) > 0:
            return False
        return True

    # filter data with age
    def filter_with_age(self):
        # 当使用age>='18'时年龄为2，。。，9或者None时会被误选中,需要二次筛选
        sql = " select patientunitstayid, " \
              " case when age like '> 89' then 90 when age='' then 0 else cast(age as Numeric) end as age " \
              " from patient; "
        list = self.cursor_connection_and_close(sql)
        new_list = [item[0] for item in list if item[1] >= 18]
        print('patient age not less than 18 : ' + str(len(new_list)))
        return new_list

    def filter_with_aps(self, id):
        sql = " select meds,dialysis,eyes,motor,verbal,urine,wbc,temperature,respiratoryrate,sodium,heartrate,meanbp,ph," \
              " hematocrit,creatinine,albumin,pao2,pco2,bun,glucose,bilirubin,fio2" \
              " from apacheapsvar where patientunitstayid = " + str(id)
        list = self.cursor_connection_and_close(sql)
        return list

    def filter_with_vent_support(self):
        # 住院记录中有呼吸机开始或者持续，患者吸入氧气浓度>21(正常人正常吸入氧气浓度为21，大于21，说明是使用氧气面罩或者呼吸机，当呼吸流速不大于3时为氧气面罩吸氧)
        sql = " select distinct patientunitstayid" \
              " from respiratorycharting" \
              " where respchartvaluelabel like 'FiO2'"
        list = self.cursor_connection_and_close(sql)
        new_list = [item[0] for item in list]
        print('need ventilator support patient unit stay : ' + str(len(list)))
        return new_list

    # todo::find out details for ALP,GCS(intub),GCS(unable)
    # find all dynamic data for each unit stay record
    def filter_dynamic(self, id, identification, enrollment):
        '''
        :param id: 患者住院记录
        :param identification: 患者ARDS确诊时间
        :param enrollment: 患者入科时间
        :return: 患者ARDS确诊时间到入科时间内的动态特征数据
        '''
        # find dynamic item in lab table
        sql = " select labname as name,  " \
              " labresult as result," \
              " labresultoffset as time " \
              " from lab as lb" \
              " where lb.patientunitstayid=" + str(id) + \
              " and  lb.labname in (" \
              " 'bedside glucose'," \
              " 'potassium'," \
              " 'sodium'," \
              " 'glucose'," \
              " 'creatinine'," \
              " 'BUN'," \
              " 'calcium'," \
              " 'bicarbonate'," \
              " 'platelets x 1000'," \
              " 'WBC x 1000'," \
              " 'magnesium'," \
              " '-eos'," \
              " '-basos'," \
              " 'albumin'," \
              " 'AST (SGOT)'," \
              " 'ALT (SGPT)'," \
              " 'total bilirubin'," \
              " 'paO2'," \
              " 'paCO2'," \
              " 'pH'," \
              " 'PT - INR'," \
              " 'HCO3'," \
              " 'FiO2'," \
              " 'Base Excess'," \
              " 'PTT'," \
              " 'lactate'," \
              " 'Total CO2'," \
              " 'ionized calcium'," \
              " 'Temperature'," \
              " 'PEEP'," \
              " 'Methemoglobin'," \
              " 'Carboxyhemoglobin'," \
              " 'Oxyhemoglobin'," \
              " 'TV'," \
              " '-bands'," \
              " 'Hct'," \
              " 'Respiratory Rate');"
        lab_list = self.cursor_connection_and_close(sql)
        # 选择满足确诊时间和登记时间内的患者数据
        lab_list = [item for item in lab_list if
                    item[1] and item[2] >= identification and item[2] <= enrollment and item[1] > 1e-6]
        lab_newlist = []
        if lab_list:
            for item in lab_list:
                name = item[0]
                result = float(item[1])
                time = item[2]
                # print('lab name %s result %s time %s ' % (name, result, time))
                if name == 'bedside glucose':
                    name = 'glucose'
                elif name == 'platelets x 1000':
                    name = 'platelets'
                elif name == 'WBC x 1000':
                    name = 'WBC'
                elif name == 'AST (SGOT)':
                    name = 'AST'
                elif name == 'ALT (SGPT)':
                    name = 'ALT'
                elif name == 'total bilirubin':
                    name = 'bilirubin'
                elif name == 'PT - INR':
                    name = 'INR'
                elif name == 'Total CO2':
                    name = 'CO2'
                elif name == 'Methemoglobin':
                    if result > 1:
                        result = result / 100
                    name = 'Hemoglobin'
                elif name == 'Carboxyhemoglobin':
                    if result > 1:
                        result = result / 100
                    name = 'Hemoglobin'
                elif name == 'Oxyhemoglobin':
                    if result > 1:
                        result = result / 100
                    name = 'Hemoglobin'
                elif name == 'direct bilirubin':
                    name = 'bilirubin'
                elif name == 'Hct':
                    if result > 1:
                        result = result / 100
                    name = 'hematocrit'
                elif name == 'FiO2':
                    if result > 1:
                        result = result / 100
                    name = 'FIO2'
                elif name == 'HCO3':
                    name = 'bicarbonate'
                elif name == '-bands' or name == '-eos' or name == '-basos':
                    name = name[1:]
                    if result > 1:
                        result = result / 100
                lab_newlist.append((name, result, time))
        # find dynamic item in nursecharting table
        sql = " select nursingchartcelltypevalname as name," \
              "        cast(nc.nursingchartvalue as numeric)  as result," \
              "        nc.nursingchartoffset as time " \
              " from nursecharting as nc" \
              " where nc.patientunitstayid =" + str(id) + \
              " and nursingchartcelltypevalname in (" \
              " 'Heart Rate'," \
              " 'Respiratory Rate'," \
              " 'Non-Invasive BP Diastolic'," \
              " 'Non-Invasive BP Systolic'," \
              " 'Non-Invasive BP Mean'," \
              " 'Temperature (C)'," \
              " 'Temperature (F)'," \
              " 'Invasive BP Diastolic'," \
              " 'Invasive BP Systolic'," \
              " 'GCS Total'," \
              " 'Invasive BP Mean'," \
              " 'Eyes'," \
              " 'Motor'," \
              " 'Verbal'," \
              " 'Bedside Glucose'," \
              " 'CVP'," \
              " 'PA Diastolic'," \
              " 'PA Systolic'," \
              " 'PA Mean'," \
              " 'End Tidal CO2')" \
              "  and nc.nursingchartvalue ~* '[0-9]';"
        nurse_list = self.cursor_connection_and_close(sql)
        nurse_list = [item for item in nurse_list if item[1] and
                      item[2] >= identification and item[2] < enrollment and item[1] > 1e-6]
        nurse_newlist = []
        if nurse_list:
            for item in nurse_list:
                name = item[0]
                result = float(item[1])
                time = item[2]
                # print('nurse name %s result %s time %s ' % (name, result, time))
                if name == 'Non-Invasive BP Diastolic':
                    name = 'NI_BP_diastolic'
                elif name == 'Non-Invasive BP Systolic':
                    name = 'NI_BP_systolic'
                elif name == 'Non-Invasive BP Mean':
                    name = 'NI_BP_mean'
                # 华氏度转换为摄氏度
                elif name == 'Temperature (F)':
                    result = (result - 32) / 1.8
                    name = 'Temperature'
                elif name == 'Temperature (C)':
                    name = 'Temperature'
                elif name == 'GCS Total':
                    name = 'GCS'
                elif name == 'Invasive BP Diastolic':
                    name = 'I_BP_diastolic'
                elif name == 'Invasive BP Systolic':
                    name = 'I_BP_systolic'
                elif name == 'Invasive BP Mean':
                    name = 'I_BP_mean'
                elif name == 'PA Diastolic':
                    name = 'diastolic_PAP'
                elif name == 'PA Systolic':
                    name = 'systolic_PAP'
                elif name == 'PA Mean':
                    name = 'mean_PAP'
                elif name == 'End Tidal CO2':
                    name = 'CO2'
                elif name == 'Bedside Glucose':
                    name = 'glucose'
                nurse_newlist.append((name, result, time))
        # print('nurse new list : ' + str(nurse_newlist))

        # find dynamic item in respiratorycharting table
        sql = " select rc.respchartvaluelabel as name, cast(rc.respchartvalue as numeric) as result,rc.respchartoffset as time " \
              " from respiratorycharting as rc" \
              " where rc.patientunitstayid = " + str(id) + \
              "  and rc.respchartvalue ~* '[0-9]'" \
              "  and rc.respchartvaluelabel in ( " \
              " 'FiO2'," \
              " 'PEEP'," \
              " 'Mean Airway Pressure', " \
              " 'Peak Insp. Pressure', " \
              " 'SaO2', " \
              " 'Plateau Pressure', " \
              " 'PEEP/CPAP', " \
              " 'ETCO2'," \
              " 'Adult Con Pt/Vent SpO2'," \
              " 'NIV Pt/Vent SpO2_5'," \
              " 'Tidal Volume, Delivered')" \
              " and not respchartvalue like '%\%%' "
        respchart_list = self.cursor_connection_and_close(sql)
        respchart_list = [item for item in respchart_list if
                          item[1] and item[2] >= identification and item[2] < enrollment and item[1] > 1e-6]
        respchart_newlist = []
        if respchart_list:
            for item in respchart_list:
                name = item[0]
                result = float(item[1])
                time = item[2]
                # print('respchart name %s result %s time %s ' % (name, result, time))
                if name == 'Peak Insp. Pressure':
                    name = 'PIP'
                elif name == 'Tidal Volume, Delivered':
                    name = 'TV'
                elif name == 'FiO2':
                    if result > 1:
                        result = result / 100
                    name = 'FIO2'
                elif name == 'NIV Pt/Vent SpO2_5':
                    name = 'SpO2'
                elif name == 'Adult Con Pt/Vent SpO2':
                    name = 'SpO2'
                elif name == 'PEEP/CPAP':
                    name = 'PEEP'
                respchart_newlist.append((name, result, time))

        sql = " select physicalexamvalue as name, cast(physicalexamtext as NUMERIC) as result, physicalexamoffset as time" \
              " from physicalexam" \
              " where physicalexamvalue in (" \
              "     'BP (systolic) Current'," \
              "     'BP (diastolic) Current'," \
              "     'FiO2%'," \
              "     'PEEP'," \
              "     'CVP')" \
              " and not physicalexamtext like '%\%%'"
        physicalexam_list = self.cursor_connection_and_close(sql)
        physicalexam_list = [item for item in physicalexam_list if item[1] and
                             item[2] >= identification and item[2] < enrollment and item[1] > 1e-6]
        physical_newlist = []
        if physicalexam_list:
            for item in physicalexam_list:
                # print('item : ' + str(item))
                name = item[0]
                result = float(item[1])
                time = item[2]
                if name == 'BP (systolic) Current':
                    name = 'I_BP_systolic'
                elif name == 'BP (diastolic) Current':
                    name = 'I_BP_diastolic'
                elif name == 'FiO2%':
                    if result > 1:
                        result = result / 100
                    name = 'FIO2'
                physical_newlist.append((name, result, time))
        result_list = lab_newlist + nurse_newlist + respchart_newlist + physical_newlist
        result_list = list(set(result_list))
        result_list = filter_invalid_dynamic_items(result_list)
        return result_list

    # 获取患者入院诊断信息
    def check_admission_diagnosis_by_id(self, id):
        '''
        :param id:
        :return:
        '''
        sql = " select " \
              " case when apacheadmissiondx like '%Angina, unstable%' then 1 else 0 end                                            as ACSD," \
              " case when apacheadmissiondx like '%MI%' then 1 else 0 end                                                          as AMI," \
              " case when apacheadmissiondx like '%Renal %'  then 1 else 0 end                                                     as ARF," \
              " case when apacheadmissiondx like '%Rhythm%' then 1 else 0 end                                                      as Arrhythmia," \
              " case when apacheadmissiondx like '%Asthma%' or apacheadmissiondx like '%Emphysema%' then 1 else 0 end              as Asthma_Emphysema," \
              " case when apacheadmissiondx like '%Cancer%' or apacheadmissiondx like '%Leukemia,%' then 1 else 0 end              as Cancer," \
              " case when apacheadmissiondx like '%Cardiac arrest%' then 1 else 0 end                                              as Cardiac_Arrest," \
              " case when apacheadmissiondx like '%Shock, cardiogenic%' then 1 else 0 end                                          as Cardiogenic_Shock," \
              " case when apacheadmissiondx like '%Cardiovascular medical%'  then 1 else 0 end                                     as CardM," \
              " case when apacheadmissiondx like '%Angina, stable%' or apacheadmissiondx like '%Pericardi%'  then 1 else 0 end     as CardO," \
              " case when apacheadmissiondx like '%cerebrovascular%' or apacheadmissiondx like '%Hemorrhage/%' then 1 else 0 end   as CAS," \
              " case when apacheadmissiondx like '%Chest pain,%'   then 1 else 0 end                                               as CPUO," \
              " case when apacheadmissiondx like '%Coma/change%' or apacheadmissiondx like '%Nontraumatic coma%' then 1 else 0 end as Coma," \
              " case when apacheadmissiondx like 'CABG%' then 1 else 0 end                                                         as CABG," \
              " case when apacheadmissiondx like '%Diabetic%' then 1 else 0 end                                                    as Diabetic_Ketoacidosis," \
              " case when apacheadmissiondx like '%Bleeding%' or apacheadmissiondx like '%GI perforation/%' then 1 else 0 end      as GI_Bleed," \
              " case when apacheadmissiondx like '%GI obstruction%' then 1 else 0 end                                              as GI_Obstruction," \
              " case when apacheadmissiondx like '%,neurologic%' or apacheadmissiondx like '%Neoplasm%' or apacheadmissiondx like '%Seizures%' or apacheadmissiondx like 'Neuro%' then 1 else 0 end as Neurologic," \
              " case when apacheadmissiondx like '%Overdose,%' or apacheadmissiondx like '%Toxicity, drug%' then 1 else 0 end      as Overdose," \
              " case when apacheadmissiondx like '%Pneumonia,%' then 1 else 0 end                                                  as Pneumonia," \
              " case when apacheadmissiondx like any (array['Apnea%','%respiratory (without%', 'Atelectasis', '%open lung', '%,pleural', 'Embolus%','%syndrome%', '%hemoptysis%', '%mothorax%','%edema%','Respiratory - medical%','%fibrosis%','Tracheostomy','%mechanical ventilation%'])  then 1 else 0 end as RespiMO," \
              " case when apacheadmissiondx like '%Sepsis%' then 1 else 0 end                                                      as Sepsis," \
              " case when apacheadmissiondx like '%Thoracotomy%' then 1 else 0 end                                                 as Thoracotomy," \
              " case when apacheadmissiondx like '% trauma%' or apacheadmissiondx like  'Trauma%' then 1 else 0 end                as Trauma," \
              " case when apacheadmissiondx like '%valve%'  and apacheadmissiondx not like '%CABG%' then 1 else 0 end              as Valve_Disease," \
              " case when apacheadmissiondx like '%, other' then 1 else 0 end                                                      as others_disease   " \
              " from patient" \
              " where patientunitstayid = " + str(id) + \
              " limit 1;"
        diagnosis_list = self.cursor_connection_and_close(sql)
        if diagnosis_list:
            diagnosis_list = diagnosis_list[0]
        return diagnosis_list

    # 找到住院记录对应的静态数据项， 药物使用（10）+疾病诊断（25）+入院来源+年龄+性别+最终结果（存活，死亡）+BMI+入院评分=41个静态特征
    def filter_static(self, unitid, identification, enrollment, header):
        '''
        :param item: 包含患者住院记录id，患者ARDS确诊时间以及患者入科时间分别为id,identification,enrollment
        :return:
        '''
        header['id'] = unitid
        # 华法林药物使用
        sql = " select max(case when drugname like 'WARFARIN%' then 1 else 0 end) as warfarin" \
              " from admissiondrug" \
              " where patientunitstayid = " + str(unitid) + \
              " limit 1 ;"
        warfarin = self.cursor_connection_and_close(sql)
        if warfarin:
            warfarin = warfarin[0]
            if warfarin[0] and warfarin[0] > 0:
                header['warfarin'] = warfarin[0]
        # 其他八种药物使用
        sql = " select " \
              " max(case when lower(drugname) like '%dobutamine%' then 1 else 0 end)     as dobutamine," \
              " max(case when lower(drugname) like '%dopamine%' then 1 else 0 end)       as dopamine," \
              " max(case when lower(drugname) like '%epinephrine%' then 1 else 0 end)    as epinephrine," \
              " max(case when lower(drugname) like '%heparin%' then 1 else 0 end)        as heparin," \
              " max(case when lower(drugname) like '%milrinone%' then 1 else 0 end)      as milrinone," \
              " max(case when lower(drugname) like '%norepinephrine%' then 1 else 0 end) as norepinephrine," \
              " max(case when lower(drugname) like '%phenylephrine%' then 1 else 0 end)  as phenylephrine," \
              " max(case when lower(drugname) like '%vasopressin%' then 1 else 0 end)    as vasopressin" \
              " from infusiondrug " \
              " where patientunitstayid = " + str(unitid) + \
              " and infusionoffset >= " + str(identification) + \
              " and infusionoffset < " + str(enrollment) + \
              " limit 1 ;"
        infusiondrug_list = self.cursor_connection_and_close(sql)
        drug_list = ['dobutamine', 'dopamine', 'epinephrine', 'heparin', 'milrinone', 'norepinephrine', 'phenylephrine',
                     'vasopressin']
        if infusiondrug_list:
            infusiondrug_list = infusiondrug_list[0]
            for i, item in enumerate(infusiondrug_list):
                if item and item > 0:
                    header[drug_list[i]] = item
        # 这几类药物属于vasopressor类
        if header['norepinephrine'] or header['phenylephrine'] or header['vasopressin'] or header['dopamine'] or header[
            'epinephrine']:
            header['vasopressor'] = 1

        # 入院诊断
        sql = " select " \
              " case when apachedxgroup = 'ACS' then 1 else 0 end                 as ACSD," \
              " case when apachedxgroup = 'AMI' then 1 else 0 end                 as AMI," \
              " case when apachedxgroup = 'ARF'  then 1 else 0 end                as ARF," \
              " case when apachedxgroup = 'Arrhythmia' then 1 else 0 end          as Arrhythmia," \
              " case when apachedxgroup = 'Asthma-Emphys' then 1 else 0 end       as Asthma_Emphysema," \
              " case when apachedxgroup = 'Cancer' then 1 else 0 end              as Cancer," \
              " case when apachedxgroup = 'CardiacArrest' then 1 else 0 end       as Cardiac_Arrest," \
              " case when apachedxgroup = 'CHF' then 1 else 0 end                 as Cardiogenic_Shock," \
              " case when apachedxgroup = 'CVMedical'  then 1 else 0 end          as CardM," \
              " case when apachedxgroup = 'CVOther'  then 1 else 0 end            as CardO," \
              " case when apachedxgroup = 'CVA' then 1 else 0 end                 as CAS," \
              " case when apachedxgroup = 'ChestPainUnknown'   then 1 else 0 end  as CPUO," \
              " case when apachedxgroup = 'Coma' then 1 else 0 end                as Coma," \
              " case when apachedxgroup = 'CABG' then 1 else 0 end                as CABG," \
              " case when apachedxgroup = 'DKA' then 1 else 0 end                 as Diabetic_Ketoacidosis," \
              " case when apachedxgroup = 'GIBleed' then 1 else 0 end             as GI_Bleed," \
              " case when apachedxgroup = 'GIObstruction' then 1 else 0 end       as GI_Obstruction," \
              " case when apachedxgroup = 'Neuro' then 1 else 0 end               as Neurologic," \
              " case when apachedxgroup = 'Overdose' then 1 else 0 end            as Overdose," \
              " case when apachedxgroup = 'PNA' then 1 else 0 end                 as Pneumonia," \
              " case when apachedxgroup = 'RespMedOther'  then 1 else 0 end       as RespiMO," \
              " case when apachedxgroup = 'Sepsis' then 1 else 0 end              as Sepsis," \
              " case when apachedxgroup = 'Thoracotomy' then 1 else 0 end         as Thoracotomy," \
              " case when apachedxgroup = 'Trauma' then 1 else 0 end              as Trauma," \
              " case when apachedxgroup = 'ValveDz' then 1 else 0 end             as Valve_Disease" \
              " from apache_groups" \
              " where patientunitstayid = " + str(unitid) + \
              " limit 1 ;"
        diagnosis_list = self.cursor_connection_and_close(sql)
        if diagnosis_list:
            diagnosis_list = diagnosis_list[0]
            list = ['ACSD', 'AMI', 'ARF', 'Arrhythmia', 'Asthma_Emphysema', 'Cancer', 'Cardiac_Arrest',
                    'Cardiogenic_Shock', 'CardM', 'CardO', 'CAS', 'CPUO', 'Coma', 'CABG', 'Diabetic_Ketoacidosis',
                    'GI_Bleed', 'GI_Obstruction', 'Neurologic', 'Overdose', 'Pneumonia', 'RespiMO', 'Sepsis',
                    'Thoracotomy', 'Trauma', 'Valve_Disease']
            for i in range(len(diagnosis_list)):
                if diagnosis_list[i]:
                    header[list[i]] = diagnosis_list[i]
            if np.sum(diagnosis_list) < 1:
                header['other_disease'] = 1
        else:
            header['other_disease'] = 1

        # 统计学信息
        sql = " select case " \
              " when hospitaladmitsource like 'Emergency Department' then 1 " \
              " when hospitaladmitsource in ('Operating Room', 'Recovery Room', 'PACU') then 2 " \
              " when hospitaladmitsource in " \
              "      ('Direct Admit', 'Acute Care/Floor', 'Step-Down Unit (SDU)', 'Other ICU', 'Chest Pain Center'," \
              "       'ICU to SDU', 'ICU') then 3 " \
              " when hospitaladmitsource = 'Floor' then 4 " \
              " else 5 end  as admitsource," \
              " case when gender like 'Female' then 1 when  gender like 'Male' then 0  else -1 end as gender," \
              " case when age like '> 89' then 90 else cast(age as numeric) end as age," \
              " admissionweight as weight," \
              " admissionheight as height " \
              " from patient " \
              " where patientunitstayid = " + str(unitid) + \
              " limit 1 ;"
        source = self.cursor_connection_and_close(sql)
        if source:
            source = source[0]
            header['admitsource'] = source[0]
            header['gender'] = source[1]
            header['age'] = float(source[2])
            weight = source[3]
            height = source[4]
            if weight and height:
                # 身高转换
                if height > 1 and height < 2.5:
                    header['BMI'] = float(round(weight / pow(height, 2), 2))
                if height >= 100:
                    header['BMI'] = float(round(weight / pow(height / 100, 2), 2))
                else:
                    header['BMI'] = 0
        sql = " select case when apachescore>0 then apachescore else 0 end as admission_score " \
              " from apachepatientresult as aps " \
              " where patientunitstayid = " + str(unitid) + \
              " limit 1 ;"
        admission_score = self.cursor_connection_and_close(sql)
        if admission_score:
            admission_score = admission_score[0]
            if admission_score[0]:
                header['admission_score'] = admission_score[0]

    # 评估患者的住院结果
    def access_outcome(self, id, enrollment):
        '''
        :param id: 患者住院记录id
        :param enrollment: 患者入科时间
        :return: 患者预后结果评估，快速恢复0，长期住院1，快速死亡3
        '''
        sql = " select labresultoffset as time,labname as name,labresult as value" \
              " from lab" \
              " where patientunitstayid = " + str(id) + \
              " and labname in( 'paO2', 'FiO2')" \
              " and labresultoffset>=" + str(enrollment + 1440) + \
              " order by time asc; "
        pa_fi = self.cursor_connection_and_close(sql)
        # 是否为ARDS患者、确诊时间、患者严重程度评级
        flag, identification, severity = access_ards(pa_fi)
        sql = " select" \
              " unitdischargestatus ," \
              " hospitaldischargestatus," \
              " unitdischargeoffset/1440 as unitstay," \
              " (hospitaldischargeoffset - hospitaladmitoffset)/1440 as hospitalstay," \
              " unitdischargelocation," \
              " hospitaldischargelocation," \
              " cast(unitdischargeoffset as numeric), " \
              " cast(hospitaldischargeoffset as numeric), " \
              " cast(hospitaladmitoffset as numeric) " \
              " from patient" \
              " where patientunitstayid = " + str(id)
        patient = self.cursor_connection_and_close(sql)
        unitdischargestatus = patient[0][0]
        hospitaldischargestatus = patient[0][1]
        unitstay = patient[0][2]
        hospitalstay = patient[0][3]
        unitdischargelocation = patient[0][4]
        hospitaldischargelocation = patient[0][5]
        unitdischargeoffset = int(patient[0][6])
        hospitaldischargeoffset = int(patient[0][7])
        hospitaladmitoffset = int(patient[0][8])
        # 先判断患者快速恢复，长期住院，快速死亡和其他状态
        # 快速死亡
        # 判断28天内转归情况
        if hospitalstay <= 28 and hospitaldischargestatus == 'Expired':
            status_28 = 1
        elif unitdischargestatus == 'Expired' and (unitdischargeoffset - hospitaladmitoffset) / 1440 <= 28:
            status_28 = 1
        else:
            status_28 = 0
        # 判断患者注册后24小时内转归情况
        if unitdischargestatus == 'Expired' and unitdischargeoffset <= enrollment + 1440:
            outcome = 2
        # 快速死亡
        elif hospitaldischargestatus == 'Expired' and hospitaldischargeoffset <= enrollment + 1440:
            outcome = 2
        # 快速恢复
        elif flag == False:
            if unitdischargestatus != 'Expired' and unitdischargeoffset <= enrollment + 1440:
                outcome = 0
            elif hospitaldischargestatus != 'Expired' and hospitaldischargeoffset <= enrollment + 1440:
                outcome = 0
            else:
                outcome = 1
        else:
            outcome = 1
        # 判断患者注册后48小时内转归情况
        if unitdischargestatus == 'Expired' and unitdischargeoffset <= enrollment + 2880:
            outcome_48 = 2
        # 快速死亡
        elif hospitaldischargestatus == 'Expired' and hospitaldischargeoffset <= enrollment + 2880:
            outcome_48 = 2
        # 快速恢复
        elif flag == False:
            if unitdischargestatus != 'Expired' and unitdischargeoffset <= enrollment + 2880:
                outcome_48 = 0
            elif hospitaldischargestatus != 'Expired' and hospitaldischargeoffset <= enrollment + 2880:
                outcome_48 = 0
            else:
                outcome_48 = 1
        else:
            outcome_48 = 1
        # 判断患者注册后48小时内转归情况
        if unitdischargestatus == 'Expired' and unitdischargeoffset <= enrollment + 4320:
            outcome_72 = 2
        # 快速死亡
        elif hospitaldischargestatus == 'Expired' and hospitaldischargeoffset <= enrollment + 4320:
            outcome_72 = 2
        # 快速恢复
        elif flag == False:
            if unitdischargestatus != 'Expired' and unitdischargeoffset <= enrollment + 4320:
                outcome_72 = 0
            elif hospitaldischargestatus != 'Expired' and hospitaldischargeoffset <= enrollment + 4320:
                outcome_72 = 0
            else:
                outcome_72 = 1
        else:
            outcome_72 = 1
        # 再判断患者详细状态
        detail_outcome = ''
        # 存活
        detail = ''
        if unitdischargestatus == 'Expired':
            detail_outcome += 'Expired in '
            if hospitaldischargelocation != 'Home':
                detail_outcome += 'ICU and stay'
            else:
                detail_outcome += 'Home and stay'
            if unitstay < 28:
                detail_outcome += ' less than 28 days'
            else:
                detail_outcome += ' not less than 28 days'
        elif hospitaldischargestatus == 'Expired':
            detail_outcome += 'Expired in '
            if unitdischargelocation != 'Home':
                detail_outcome += 'Hospital and stay'
            else:
                detail_outcome += 'Home and stay'
            if hospitalstay < 28:
                detail_outcome += ' less than 28 days'
            else:
                detail_outcome += ' not less than 28 days'
        if 'Expired' in detail_outcome:
            if 'ICU' in detail_outcome:
                # ICU死亡
                detail += '1'
            else:
                detail += '0'
            if 'less than 28 days' in detail_outcome and 'not' not in detail_outcome:
                # 28天内死亡
                detail += '2'
            else:
                detail += '0'
            if 'Home' not in detail_outcome:
                # 医院死亡
                detail += '4'
            else:
                detail += '0'
        else:
            # 其他死亡
            detail += '3'
        return outcome, detail, unitstay, hospitalstay, status_28, outcome_48, outcome_72

    # 删除异常患者住院记录
    # @log_time
    def isvalid_data(self, id, identification):
        # 1 缺乏死亡时间--> 无
        # 没有入住icu --> 科室住院类型不为ICU
        # 死亡后患者确诊  确诊时间>死亡时间
        # 同一患者在同一次住院期间有多个住院记录
        # icu住院时间比医院住院时间长  -->医院离开时间小于ICU离开时间
        # icu住院时间为0分钟  -->科室离开时间为0
        sql = " select unitdischargeoffset as unitstay,hospitaldischargeoffset-hospitaladmitoffset as hospitalstay,hospitaldischargeoffset,unitdischargestatus,hospitaldischargestatus " \
              " from patient " \
              " where patientunitstayid = " + str(id)
        patientinfo = self.cursor_connection_and_close(sql)
        unitstay = patientinfo[0][0]
        hospitalstay = patientinfo[0][1]
        hospitaldischargeoffset = patientinfo[0][2]
        unitdischargestatus = patientinfo[0][3]
        hospitaldischargestatus = patientinfo[0][4]
        # 患者科室住院类型不是ICU,患者科室离开时间晚于医院离开时间，患者ICU住院时间为0
        if unitstay > hospitalstay:
            print('患者科室离开时间晚于医院离开时间')
            return False
        if unitstay == 0:
            print('患者ICU住院时间为0')
            return False
        if unitdischargestatus == 'Expired':
            if unitstay < identification:
                print('患者科室中死亡，且死亡时间早于确诊时间')
                return False
        if hospitaldischargestatus == 'Expired':
            if hospitaldischargeoffset < identification:
                print('患者死于医院，且死亡时间早于确诊时间')
                return False
        sql = " select uniquepid, hospitaldischargeyear, hospitalid, patienthealthsystemstayid, hospitaladmittime24, " \
              " hospitaldischargetime24, hospitaladmitoffset, patientunitstayid" \
              " from patient " \
              " where uniquepid =(select uniquepid from patient where patientunitstayid = " + str(id) + ");"
        patientinfo = self.cursor_connection_and_close(sql)
        valid_patients = []
        # 筛选出所有合法的住院记录id
        for i in range(len(patientinfo) - 1):
            j = i + 1
            min_index = i
            while j < len(patientinfo):
                if isCommon(patientinfo[j], patientinfo[min_index]):
                    if patientinfo[j][-2] < patientinfo[min_index][-2]:
                        min_index = j
                j += 1
            if not patientinfo[min_index][-1] in valid_patients:
                valid_patients.append(patientinfo[min_index][-1])
        if len(patientinfo) > 1 and not id in valid_patients:
            print('住院记录为同一次住院的后续住院')
            return False
        else:
            return True

    # 抽取ApacheIV变量数量
    def extra_apacheIV_data_by_id(self, id, header):
        sql = " select gender, teach" \
              "type, admitsource, age, admitdiagnosis, meds, thrombolytics, diedinhospital, aids, " \
              " hepaticfailure, lymphoma, metastaticcancer, leukemia, immunosuppression, cirrhosis, electivesurgery, " \
              " activetx, readmit, ima, midur, ventday1, oobintubday1, oobventday1, diabetes, dischargelocation, " \
              " visitnumber, amilocation, sicuday, saps3day1, saps3today, saps3yesterday, verbal, motor, eyes, pao2, " \
              " fio2, creatinine, day1meds, day1verbal, day1motor, day1eyes, day1pao2, day1fio2" \
              " from apachepredvar" \
              " where patientunitstayid= " + str(id)
        apache = self.cursor_connection_and_close(sql)
        header_list = filter.param.apache_header[1:]
        if apache:
            for i in range(len(header_list)):
                header[header_list[i]] = apache[i]
            return True
        return False


class sofa_computer:
    def __init__(self):
        self.database = EICU_DATABASE
        self.user = USER
        self.password = PASSWORD
        self.host = HOST
        self.port = PORT
        self.search_path = EICU_SEARCH_PATH
        self.connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port)


def access_ards(pa_fi):
    '''
    :param pa_fi: 输入患者指定时间内的pao2和fio2数值列表，time:检查时间,name：检查项名称，value：检查项数值
    :return: 是否为ARDS患者、确诊时间、患者严重程度评级
    '''
    # 将数据按照时间进行排序
    pa_fi = sorted(pa_fi, key=lambda x: x[0])
    # 若采集项为空或者数据采集时间小于八小时, 则不计算p / f
    pa_mean = [item[2] for item in pa_fi if item[1] == 'paO2' and item[2]]
    fi_mean = [item[2] for item in pa_fi if item[1] != 'paO2' and item[2]]
    if not len(pa_mean) * len(fi_mean):
        return False, -1, -1
    # 若只有一种采集项,则不计算p/f
    pa_mean = np.mean(pa_mean)
    fi_mean = np.mean(fi_mean)
    # pao2和fio2数值列表规格化
    temp = []
    for i, item in enumerate(pa_fi):
        # time,name,value
        # 1.使用对应均值填充为None
        if item[2] is None:
            if item[1] == 'paO2':
                temp.append((item[0], item[1], pa_mean))
            else:
                temp.append((item[0], item[1], fi_mean))
        else:
            temp.append(item)
    # 2.若当前时间点的值不存在,需要找到对应的前面一个时间点的数据来填充当前值
    pa_fi = temp
    i = 0
    while i < len(pa_fi) - 1:
        # time,name,value
        if pa_fi[i][0] == pa_fi[i + 1][0] and pa_fi[i][1] != pa_fi[i + 1][1]:
            i += 1
        else:
            j = i - 1
            while j >= 0:
                # 当前检查项与指定检查项不同时
                if pa_fi[j][1] != pa_fi[i][1]:
                    # 两个检查项时间不一样说明需要填充前面的数值
                    extra_item = list(pa_fi[j])
                    extra_item[0] = pa_fi[i][0]
                    extra_item = tuple(extra_item)
                    pa_fi.insert(i + 1, extra_item)
                    i += 1
                    break
                else:
                    j -= 1
        i += 1
    i = 0
    while i < len(pa_fi) - 1:
        if pa_fi[i][0] != pa_fi[i + 1][0]:
            del pa_fi[i]
        else:
            break
    # 4.计算患者identification时间以及pao2, fio2, p / f的中位数, 方差以及变化率
    p_f = []
    # 分别计算统计pao2,fio2,p/f
    i = 0
    min_pf = 301
    while i < len(pa_fi) - 1:
        if pa_fi[i][1] == 'paO2':
            # 吸入氧气浓度小于1
            fio2 = pa_fi[i + 1][2]
            pao2 = pa_fi[i][2]
        else:
            fio2 = pa_fi[i][2]
            pao2 = pa_fi[i + 1][2]
        if fio2 > 1:
            fio2 = fio2 / 100
        if fio2 < 1e-6:
            if fi_mean > 1:
                fi_mean = fi_mean / 100
            fio2 = fi_mean
        current_p_f = pao2 / fio2
        if current_p_f < min_pf:
            min_pf = current_p_f
        p_f.append((pa_fi[i][0], current_p_f))
        i += 2
    i = 0
    p_f_identification = -1
    if len(p_f) == 1 and p_f[0][1] <= 300:
        p_f_identification = p_f[0][0]
    else:
        while i < len(p_f):
            # 当前时间点的p/f满足柏林定义
            if p_f[i][1] <= 300:
                start = p_f[i][0]
                j = i + 1
                while j < len(p_f):
                    end = p_f[j][0]
                    # 下一个时间点满足柏林定义
                    if p_f[j][1] <= 300:
                        # 当前时间间隔小于八小时
                        if j + 1 < len(p_f) and end - start < 480:
                            j += 1
                        else:
                            p_f_identification = end
                            break
                    else:
                        break
                if p_f_identification > 0:
                    break
                else:
                    i = j
            else:
                i += 1
    if p_f_identification != -1:
        flag = True
    else:
        flag = False
    if min_pf <= 100:
        severity = 1
    elif min_pf <= 200:
        severity = 2
    elif min_pf <= 300:
        severity = 3
    else:
        severity = 4
    return flag, p_f_identification, severity


def searchFile(dictionary, fileType):
    fileList = []
    for root, dirs, files in os.walk(dictionary):
        for file in files:
            if os.path.splitext(file)[1] == fileType:
                print(file)
                file = os.path.join(root, file)
                columns = pd.read_csv(file, low_memory=False, error_bad_lines=False)
                for col in columns:
                    if col not in fileList:
                        fileList.append(col)
    return fileList


# 计算每次住院记录中的p/f的中位数，方差以及变化率
def compute_pf(data):
    # time name value
    data.sort()
    pao2 = [float(item[2]) for item in data if item[1] == 'paO2' and item[2] > 1e-6]
    fio2 = [float(item[2]) for item in data if item[1] == 'FIO2' and item[2] > 1e-6]
    # 只有pao2或者只有fio2时，三项数值置0
    if len(pao2) < 1 or len(fio2) < 1:
        return None
    # 计算均值
    pao2_mean = np.mean(np.array(pao2))
    fio2_mean = np.mean(np.array(fio2))
    # 用均值填充缺省值
    for i in range(len(data)):
        if data[i][1] == 'paO2' and data[i][2] < 1e-6:
            data[i] = list(data[i])
            data[i][2] = pao2_mean
            data[i] = tuple(data[i])
        if data[i][1] == 'FIO2' and data[i][2] < 1e-6:
            data[i] = list(data[i])
            data[i][2] = fio2_mean
            data[i] = tuple(data[i])
    i = 0
    # time name value
    while i < len(data) - 1:
        # 当前项与下一项是同一时刻的pao2和fio2
        if data[i][0] == data[i + 1][0] and data[i][1] != data[i + 1][1]:
            i += 1
        else:
            item_name = data[i][1]
            # 当前时刻pao2或fio2不存在时，使用对应[前面项]填充
            j = i - 1
            while j >= 0:
                if data[j][1] != item_name:
                    if data[i][0] != data[j][0]:
                        extra_item = list(data[j])
                        extra_item[0] = data[i][0]
                        extra_item = tuple(extra_item)
                        data.insert(i + 1, extra_item)
                    i = i + 1
                    break
                else:
                    j -= 1
        i += 1
    i = 0
    # 计算所有p/f值
    p_f = []
    while i < len(data) - 1:
        item_name = data[i][1]
        item_time = data[i][0]
        next_name = data[i + 1][1]
        next_time = data[i + 1][0]
        if item_name != next_name and item_time == next_time:
            if item_name == 'paO2':
                pao2 = data[i][2]
                fio2 = data[i + 1][2]
                if fio2 > 1:
                    fio2 /= 100
                elif fio2 < 1e-6:
                    if fio2_mean > 1:
                        fio2_mean /= 100
                    fio2 = fio2_mean
                p_f.append(('P/F ratio', round(pao2 / fio2, 4), next_time))
            else:
                fio2 = data[i][2]
                if fio2 > 1:
                    fio2 /= 100
                elif fio2 < 1e-6:
                    if fio2_mean > 1:
                        fio2_mean /= 100
                    fio2 = fio2_mean
                pao2 = data[i + 1][2]
                p_f.append(('P/F ratio', round(pao2 / fio2, 4), next_time))
            i += 2
        else:
            i += 1
    # 将数值排序
    p_f.sort()
    return p_f


def compute_dynamic(id, data, header):
    '''
    :param data: 一个元素为元组的数组，元组内部分别为检查项名称name，检查项值value以及检查项时间time
    :param header: 为最终表格表头
    :return:最终动态特征的中位数、方差以及变化率
    '''
    # 对于多名称特征数据进行更名
    new_data = [(item[0], float(round(item[1], 4)), item[2]) for item in data]
    pao2_fio2 = [(item[2], item[0], float(item[1])) for item in data if item[0] == 'paO2' or item[0] == 'FIO2']
    # 计算p/f的值
    p_f = compute_pf(pao2_fio2)
    if p_f:
        p_f = [item for item in p_f if item[0] == 'P/F ratio' and item[1] and item[1] > 0 and item[1] < 650]
    if p_f:
        new_data += p_f
    new_data.sort()
    filter_invalid_dynamic_items(new_data)
    # 动态数据中位数，方差，变化率的计算
    # deal with dynamic data :
    # 1.id:X,drug:[~],dignosis:[~],personsonlist:[~],itemexample:{min, max, mean, std, perce_25, perce_50, perce_75, median, variance,change_rate, [dicts]}
    dynamic_dict = {}
    for i, (key, value) in enumerate(header.items()):
        if i < 42:
            dynamic_dict[key] = value
    for name in dynamic_item_list:
        # store each item infos in the format of time:value
        temp = [float(item[1]) for item in new_data if item[0] == name]
        dynamic_temp = {item[2]: item[1] for item in new_data if item[0] == name}
        # 将所有百分比数值转化为小数
        temp = [round(item / 100, 3) if name in filter.param.ratio_list and item > 1 else item for item in temp]
        min = 0
        max = 0
        mean = 0
        std = 0
        per_25 = 0
        per_50 = 0
        per_75 = 0
        median = 0
        variance = 0
        change_rate = 0
        if len(temp) < 1:
            header[name + '_median'] = -1
            header[name + '_variances'] = -1
            header[name + '_changerate'] = -1
        else:
            temp.sort()
            temp = np.array(temp)
            size = len(temp)
            median = 0
            variance = 0
            change_rate = 0
            min = np.min(temp)
            max = np.max(temp)
            mean = np.round(np.mean(temp), 3)
            std = np.round(np.std(temp), 3)
            per_25 = np.round(np.percentile(temp, 25))
            per_50 = np.round(np.percentile(temp, 50))
            per_75 = np.round(np.percentile(temp, 75))
            if size == 1:
                median = temp[0]
            if size >= 2:
                if size % 2 == 0:
                    median = (temp[int(size / 2) - 1] + temp[int(size / 2)]) / 2
                else:
                    median = temp[int(size / 2)]
                variance = np.round(np.var(temp), 3)
                if min < 0.0001:
                    change_rate = 0
                else:
                    change_rate = round((max - min) / min, 3)
            header[name + '_median'] = round(median, 2)
            header[name + '_variances'] = round(variance, 2)
            header[name + '_changerate'] = round(change_rate, 2)
        dynamic_dict[name] = (
            min, max, mean, std, per_25, per_50, per_75, median, variance, change_rate, dynamic_temp)
    dynamic_dict['outcome'] = header['outcome']
    return header, dynamic_dict


def isCommon(t1, t2):
    for i in range(6):
        if t1[i] != t2[i]:
            return False
    return True
