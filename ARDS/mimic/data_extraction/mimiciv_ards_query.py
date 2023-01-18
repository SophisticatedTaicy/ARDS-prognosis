import time

import numpy as np
import pandas as pd
from pandas import DataFrame

from ARDS.eicu.data_extraction.query_sql import access_ards
from filter.common import init_dict
from filter.param import *
import psycopg2


class Query:
    # Postgres数据库配置初始化
    def __init__(self, type='EICU'):
        if type == '3':
            self.database = MIMIC3_DATABASE
            self.search_path = MIMIC3_SEARCH_PATH
        elif type == '4':
            self.database = MIMIC4_DATABASE
            self.search_path = MIMIC4_SEARCH_PATH
        else:
            self.database = EICU_DATABASE
            self.search_path = EICU_SEARCH_PATH
        self.user = USER
        self.password = PASSWORD
        self.host = HOST
        self.port = PORT
        # 设置eicu数据搜索路径
        self.connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port)

    # Mysql数据库配置初始化
    # def __init__(self):
    #     self.database = mysql_mimic_database_3
    #     self.user = mysql_user
    #     self.password = mysql_password
    #     self.host = mysql_HOST
    #     self.port = mysql_POST
    #     # 设置MIMIC数据搜索路径
    #     self.connection = pymysql.connect(
    #         database=self.database,
    #         user=self.user,
    #         password=self.password,
    #         host=self.host,
    #         port=self.port)

    def cursor_connection_and_close(self, sql):
        cursor = self.connection.cursor()
        default_sql = "set search_path to " + self.search_path + ";"
        cursor.execute(default_sql)
        cursor.execute(sql)
        list = cursor.fetchall()
        cursor.close()
        return list

    def mimic_cursor_connection_and_close(self, sql):
        cursor = self.connection.cursor()
        cursor.execute(sql)
        list = cursor.fetchall()
        cursor.close()
        self.connection.close()
        return list

    # 查找满足呼吸衰竭、呼吸机支持且住院时间不小于一天、住院时间不小于icu住院时间的患者病案号和icu病案号
    def extract_meet_respiratory_failure_and_ventilator_support_condition_stay_ids(self):
        sql = " select r.stay_id" \
              " from respiratory_ventilator r join patient_new p on r.stay_id=p.stay_id;"
        stay_ids = self.cursor_connection_and_close(sql)
        return stay_ids

    # 提取患者住院期间的apacheIV评分数据
    def extract_patient_apacheIV_data(self, hadm_id):
        sql = " select charttime, label, value " \
              " from chartevents,d_items " \
              " where hadm_id = " + str(hadm_id) + \
              " and d_items.itemid = chartevents.itemid " \
              " and d_items.label like '%ApacheIV%'; "
        return self.cursor_connection_and_close(sql)

    def filter_stay_id_by_p_f_ratio_and_peep(self, stay_id):
        # 采集患者所有peep\fio2\pao2信息-->PEEP/FIO2/paO2
        # start = time.time()
        sql = " select timeoffset,case when label LIKE '%PEEP%' then 'PEEP' else 'paO2' end as label,value" \
              " from labevents_new " \
              " where (label LIKE '%PEEP%' or label like '%pO2%') and stay_id=" + str(stay_id) + \
              " union " \
              " select timeoffset,case when label LIKE 'PEEP set%' then 'PEEP' when label LIKE 'PO2 (Mixed Venous)%' then 'paO2' else 'FIO2' end as label,value " \
              " from chartevents_new " \
              " where label like any (array ['PEEP set%','PO2 (Mixed Venous)%','FiO2%','Inspired O2 Fraction%']) and stay_id=" + str(
            stay_id)
        feature_datas = self.cursor_connection_and_close(sql)
        feature_datas = list(set(feature_datas))
        pao2 = len([item for item in feature_datas if 'paO2' in item[1]])
        fio2 = len([item for item in feature_datas if 'FIO2' in item[1]])
        if pao2 and fio2:
            pa_fi = [item for item in feature_datas if item[1] != 'PEEP']
            flag, identification, severity = access_ards(pa_fi)
            # 患者在入院24小时内确诊
            if flag:
                peep = [item[2] for item in feature_datas if
                        item[1] == 'PEEP' and item[0] >= identification and item[0] <= identification + 480]
                peep_less_than_5 = len([item for item in peep if item < 5 and item > 0])
                if peep_less_than_5 > 0:
                    return False, None, None
                else:
                    return True, identification, severity
            else:
                return False, None, None
        else:
            return False, None, None

    def invalid_data_filter(self, stay_id, identification):
        sql = "  select hospital_stay_days,icu_stay_days,deathtimeoffset from patient_new " \
              "  where stay_id = " + str(stay_id)
        has_icu_stay = self.cursor_connection_and_close(sql)
        # 患者有icu住院记录，排出研究队列
        if has_icu_stay:
            hospital_info = self.cursor_connection_and_close(sql)
            # print('stay id : %s 患者有效性查询耗时 ： %s' % (stay_id, end - start))
            if hospital_info:
                icu_stay_days = hospital_info[0][1]
                deathtimeoffset = hospital_info[0][2]
                # 在确诊前死亡
                if (deathtimeoffset > 0 and deathtimeoffset <= identification) or icu_stay_days == 0:
                    return False
                else:
                    return True
            else:
                return False

    # 评估患者的最后预后结果
    def access_ards_severity_after_enrollment(self, stay_id, enrollment):
        sql = "  select timeoffset, 'paO2' as label ,value" \
              "  from labevents_new " \
              "  where label like 'pO2%' " \
              "  and stay_id= " + str(stay_id) + \
              "  and timeoffset>=" + str(enrollment + 1440) + \
              "  union " + \
              "  select timeoffset,case when label='PO2 (Mixed Venous)' then 'paO2' else 'FIO2' end as label,value" \
              "  from chartevents_new " \
              "  where label like any (array ['PO2 (Mixed Venous)','FiO2%','Inspired O2 Fraction']) " \
              "  and stay_id= " + str(stay_id) + \
              "  and timeoffset>= " + str(enrollment + 1440) + ";"
        enrollment_ards_severity = self.cursor_connection_and_close(sql)
        if enrollment_ards_severity:
            flag, identification, severity = access_ards(enrollment_ards_severity)
        else:
            flag = False
            identification = -1
            severity = -1
        sql = " select icuoutstatus, hopitaloutstatus, icu_stay_days, hospital_stay_days,icudischarge_location," \
              " hospitaldischarge_location,icudischargeoffset,hospitaldischargeoffset,hospitaladmitoffset " \
              " from patient_new" \
              " where stay_id = " + str(stay_id)
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
        return outcome, status_28, detail, unitstay, hospitalstay, outcome_48, outcome_72

    # 获取患者药物使用、入院诊断
    def get_static_data_by_stay_id(self, stay_id, header):
        static_header = drug_list + diagnosis_abbrevation_list + person_info_list
        header['id'] = stay_id
        # 药物使用
        sql = " select " \
              " max(case when lower(drug) like '%warfarin%' then 1 else 0 end) as warfarin," \
              " max(case when lower(drug) like '%dobutamine%' then 1 else 0 end) as dobutamine," \
              " max(case when lower(drug) like '%dopamine%' then 1 else 0 end) as dopamine," \
              " max(case when lower(drug) like '%epinephrine%' then 1 else 0 end) as epinephrine," \
              " max(case when lower(drug) like '%heparin%' then 1 else 0 end) as heparin," \
              " max(case when lower(drug) like '%milrinone%' then 1 else 0 end) as milrinone," \
              " max(case when lower(drug) like '%norepinephrine%' then 1 else 0 end) as norepinephrine," \
              " max(case when lower(drug) like '%phenylephrine%' then 1 else 0 end) as phenylephrine," \
              " max(case when lower(drug) like '%vasopressin%' then 1 else 0 end) as vasopressin," \
              " max(case when lower(drug) like '%vasopressor%'  then 1 else  0 end) as vasopressor" \
              " from drugusage " \
              " where stay_id=" + str(stay_id)
        drug_usage = self.cursor_connection_and_close(sql)
        if drug_usage and drug_usage[0]:
            for i in range(10):
                header[drug_list[i]] = drug_usage[0][i]
        if header['norepinephrine'] or header['phenylephrine'] or header['vasopressin'] or header['dopamine'] or header[
            'epinephrine']:
            header['vasopressor'] = 1
        sql = " select max(case when chinese_title like '%急性冠状动脉%' then 1 else 0 end)      as ACSD," \
              " max(case when chinese_title like '%急性心肌梗塞%' then 1 else 0 end)      as AMI," \
              " max(case when chinese_title like '%急性肾衰竭%' then 1 else 0 end)      as ARF," \
              " max(case when chinese_title like '%心律不整%' then 1 else 0 end)      as Arrhythmia," \
              " max(case when chinese_title like '%哮喘%' or chinese_title like '%肺气肿%' then 1 else 0 end)      as Asthma_Emphysema," \
              " max(case when chinese_title like '%癌症%' then 1 else 0 end)      as Cancer," \
              " max(case when chinese_title like '%心脏骤停%' then 1 else 0 end)      as Cardiac_Arrest," \
              " max(case when chinese_title like '%心因性休克%' or chinese_title like '%术后休克，心源性%'  then 1 else 0 end)      as Cardiogenic_Shock," \
              " max(case when chinese_title like '%心血管病%' or chinese_title like '%心血管梅毒%' then 1 else 0 end)      as CardM," \
              " max(case when chinese_title like '%其他心血管%' then 1 else 0 end)      as CardO," \
              " max(case when chinese_title like '%中风%' then 1 else 0 end)      as CAS," \
              " max(case when chinese_title like '%胸痛%' then 1 else 0 end)      as CPUO," \
              " max(case when chinese_title like '%昏迷%' then 1 else 0 end)      as Coma," \
              " max(case when chinese_title like '%心脏搭桥%' then 1 else 0 end)      as CABG," \
              " max(case when chinese_title like '%酮症酸中毒%' then 1 else 0 end)      as Diabetic_Ketoacidosis," \
              " max(case when chinese_title like '%胃肠道出血%' then 1 else 0 end)      as GI_Bleed," \
              " max(case when chinese_title like '%肠或腹膜粘连伴梗阻（术后）（感染后）%' then 1 else 0 end) as GI_Obstruction," \
              " max(case when chinese_title like '%神经%' then 1 else 0 end)    as Neurologic," \
              " max(case when chinese_title like '%药物中毒%' then 1 else 0 end)  as Overdose," \
              " max(case when chinese_title like '%肺炎%' then 1 else 0 end)    as Pneumonia," \
              " max(case when chinese_title like '%呼吸系统%' then 1 else 0 end)  as RespiMO," \
              " max(case when chinese_title like '%脓毒症%' then 1 else 0 end)   as Sepsis," \
              " max(case when chinese_title like '%开胸术%' then 1 else 0 end)      as Thoracotom," \
              " max(case when chinese_title like '%创伤%' then 1 else 0 end)    as Trauma," \
              " max(case when chinese_title like '%心脏瓣膜%' then 1 else 0 end) as Valve_Disease" \
              " from diagnosis" \
              " where stay_id = " + str(stay_id)
        diagnosis_usage = self.cursor_connection_and_close(sql)
        diagnosis_list = static_header[10:35]
        if diagnosis_usage and diagnosis_usage[0]:
            for i in range(25):
                header[diagnosis_list[i]] = diagnosis_usage[0][i]
        else:
            header['other_disease'] = 1

        sql = " select admitsource, gender, anchor_age" \
              " from patient_new" \
              " where stay_id = " + str(stay_id)
        physio = self.cursor_connection_and_close(sql)
        if physio:
            header['admitsource'] = physio[0][0]
            header['gender'] = physio[0][1]
            header['age'] = physio[0][2]
        sql = " select label, value" \
              " from patient_BMI" \
              " where stay_id =" + str(stay_id)
        bmi = self.cursor_connection_and_close(sql)

        if bmi:
            w_list = [item[1] for item in bmi if item[0] == 'Weight']
            h_list = [item[1] for item in bmi if item[0] == 'Height']
            weight = 0
            height = 0
            if w_list:
                weight = max(w_list)
            if h_list:
                height = max(h_list)
            if height > 0 and weight > 0:
                header['BMI'] = round(weight / pow(height, 2), 2)

    # 获取患者动态数据信息
    def get_dynamic_data_by_hadm_id_and_stay_id(self, stay_id, identification, enrollment):
        sql = " select timeoffset, label, case when valueuom = '%' and value > 1 then value / 100 else value end as value" \
              " from chartevents_new" \
              " where stay_id =  " + str(stay_id) + \
              " and timeoffset > " + str(identification) + \
              " and timeoffset < " + str(enrollment) + \
              " union " \
              " select timeoffset, label, case when valueuom = '%' and value > 1 then value / 100 else value end as value" \
              " from labevents_new" \
              " where stay_id =  " + str(stay_id) + \
              " and timeoffset > " + str(identification) + \
              " and timeoffset < " + str(enrollment)
        dynamic_list = self.cursor_connection_and_close(sql)
        return dynamic_list

    def standardize_dynamic_data(self, datas):
        items = [item[1].lower() for item in datas if not item[2] is None]
        values = [float(item[2]) for item in datas if not item[2] is None]
        # 对应的动态指标以及其在数据集中出现的名称形式
        final_datas = []
        for key, value in mimic_dynamic_dict.items():
            if key == 'bicarbonate':
                index = [i for i, item in enumerate(items) if 'hco3' in item or 'bicarbonate' in item]
            elif key == 'FIO2':
                index = [i for i, item in enumerate(items) if 'fio2' in item or 'inspired o2 fraction' in item]
            else:
                index = [i for i, item in enumerate(items) if value in item.lower()]
            if len(index) > 0:
                for i in index:
                    final_datas.append((key, values[i], 0))
        final_datas.sort()
        return final_datas
