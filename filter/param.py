from filter.common import special_list

USER = "postgres"
PASSWORD = "123456"
HOST = "172.16.60.173"
PORT = "3307"
EICU_DATABASE = "eicu"
EICU_SEARCH_PATH = 'eicu_crd'
MIMIC4_DATABASE = 'MIMIC'
MIMIC3_DATABASE = 'mimic_iii'
MIMIC4_SEARCH_PATH = 'public'
MIMIC3_SEARCH_PATH = 'iii'

mysql_user = 'root'
mysql_password = 'MIMIC'
# mysql_HOST = '127.0.0.1'
mysql_HOST = '172.168.1.183'
mysql_POST = 3306
mysql_mimic_database_3 = 'mimic_3'
mysql_mimic_database_4 = 'mimic'
drug_detail_list = ['warfarin\nuse', 'dobutamine\nuse', 'dopamine\nuse', 'epinephrine\nuse', 'heparin\nuse',
                    'milrinone\nuse', 'norepinephrine\nuse', 'phenylephrine\nuse', 'vasopressin\nuse',
                    'vasopressor\nuse']
drug_list = ['warfarin', 'dobutamine', 'dopamine', 'epinephrine', 'heparin', 'milrinone', 'norepinephrine',
             'phenylephrine', 'vasopressin', 'vasopressor']
diagnosis_abbrevation_list = ['ACSD', 'AMI', 'ARF', 'Arrhythmia', 'Asthma_Emphysema', 'Cancer', 'Cardiac_Arrest',
                              'Cardiogenic_Shock', 'CardM', 'CardO', 'CAS', 'CPUO', 'Coma', 'CABG',
                              'Diabetic_Ketoacidosis', 'GI_Bleed', 'GI_Obstruction', 'Neurologic', 'Overdose',
                              'Pneumonia', 'RespiMO', 'Sepsis', 'Thoracotomy', 'Trauma', 'Valve_Disease',
                              'other_disease']
diagnosis_chinese_list = ['急性冠状动脉', '急性心肌梗塞', '急性肾衰竭', '心律不整', '哮喘', '癌症', '心脏骤停', '心因性休克',
                          '心血管病', '其他心血管', '中风', '胸痛', '昏迷', '心脏搭桥', '酮症酸中毒', '胃肠道出血',
                          '肠或腹膜粘连伴梗阻', '神经', '药物中毒', '肺炎', '呼吸系统', '脓毒症', '开胸术', '创伤', '心脏瓣膜',
                          '其他疾病']
diagnosis_list = ['Acute Coronary\nSyndrome', 'Acute Myocardial\nInfarction', 'Acute Renal\nFailure', 'Arrhythmia',
                  'Asthma or\nEmphysema', 'Cancer', 'Cardiac Arrest', 'Cardiogenic Shock', 'Cardiovascular\n(Medical)',
                  'Cardiovascular\n(Other)', 'Cerebrovascular\nAccident/Stroke', 'Chest Pain\nUnknown Origin', 'Coma',
                  'Coronary Artery\nBypass Graft', 'Diabetic\nKetoacidosis', 'Gastrointestinal\nBleed',
                  'Gastrointestinal\nObstruction', 'Neurologic', 'Overdose', 'Pneumonia',
                  'Respiratory\n(Medical/Other)', 'Sepsis', 'Thoracotomy', 'Trauma', 'Valve Disease', 'others diease']
diagnosis_key_names = ['cancer', 'arrhythmia', 'cardiac arrest', 'cardiogenic shock', 'asthma',
                       'cardiovascular medical',
                       'acute myocardial infarction', 'acute coronary syndrome', 'acute renal failure', 'pneumonia',
                       'coma', 'stroke', 'neurologic', 'overdose', 'cabg', 'gi bleed', 'gi obstruction', 'chest pain',
                       'diabetic', 'cerebrovascular', 'hemorrhage', 'trauma', 'sepsis', 'thoracotomy',
                       'valve replacement', 'respiratory', 'ards', 'cardiovascular']
person_info_list = ['admitsource', 'gender', 'age', 'BMI', 'admission_score']

dynamic_item_list = ['albumin', 'ALT', 'AST', 'bands', 'Base Excess', 'basos', 'bicarbonate', 'bilirubin', 'BUN',
                     'calcium', 'CO2', 'creatinine', 'eos', 'FIO2', 'glucose', 'Hemoglobin', 'INR', 'ionized calcium',
                     'lactate', 'magnesium', 'paCO2', 'paO2', 'P/F ratio', 'PEEP', 'pH', 'platelets', 'potassium',
                     'PTT', 'PIP', 'sodium', 'Temperature', 'WBC', 'Mean Airway Pressure', 'Plateau Pressure', 'SaO2',
                     'SpO2', 'TV', 'CVP', 'ETCO2', 'diastolic_PAP', 'mean_PAP', 'systolic_PAP', 'Eyes', 'GCS', 'Motor',
                     'Verbal', 'Heart Rate', 'I_BP_diastolic', 'I_BP_mean', 'I_BP_systolic', 'NI_BP_diastolic',
                     'NI_BP_mean', 'NI_BP_systolic', 'Respiratory Rate', 'hematocrit']
dynamic_median_list = special_list(dynamic_item_list, ['median'])
dynamic_detail_list = special_list(dynamic_item_list, ['median', 'variances', 'changerate'])
mimic_dynamic_dict = {'albumin': 'albumin', 'ALT': 'alt', 'AST': 'ast', 'bands': 'bands',
                      'Base Excess': 'base excess', 'basos': 'basos', 'bicarbonate': 'bicarbonate',
                      'bilirubin': 'bilirubin', 'BUN': 'bun', 'calcium': 'calcium', 'CO2': 'co2',
                      'creatinine': 'creatinine', 'eos': 'eos', 'FIO2': 'fio2',
                      'glucose': 'glucose', 'Hemoglobin': 'hemoglobin', 'INR': 'inr',
                      'ionized calcium': 'ionized calcium', 'lactate': 'lactate', 'magnesium': 'magnesium',
                      'paCO2': 'paco2', 'paO2': 'po2', 'P/F ratio': 'p/f', 'PEEP': 'peep', 'pH': 'ph',
                      'platelets': 'platelets', 'potassium': 'potassium', 'ptt': 'ptt', 'PIP': 'pip',
                      'sodium': 'sodium', 'Temperature': 'temperature', 'WBC': 'wbc',
                      'Mean Airway Pressure': 'mean airway pressure', 'Plateau Pressure': 'Plateau pressure',
                      'SaO2': 'sao2', 'SpO2': 'spo2', 'TV': 'tidal volume', 'CVP': 'cvp', 'ETCO2': 'etco2',
                      'diastolic_PAP': 'pap d', 'mean_PAP': 'pap m', 'systolic_PAP': 'pap s', 'Eyes': 'eye',
                      'GCS': 'gcs', 'Motor': 'motor', 'Verbal': 'verbal', 'Heart Rate': 'hr',
                      'I_BP_diastolic': 'arterial blood pressure d', 'I_BP_mean': 'arterial blood pressure m',
                      'I_BP_systolic': 'arterial blood pressure s', 'NI_BP_diastolic': 'NI_BP_diastolic',
                      'NI_BP_mean': 'NI_BP_mean',
                      'NI_BP_systolic': 'NI_BP_systolic', 'Respiratory Rate': 'rr', 'hematocrit': 'hematocrit'}

extra_list = ['outcome', 'detail', 'unit', 'hospital', 'status_28', 'severity', 'identification', 'enrollment']
aps_header = ['id'] + drug_list + diagnosis_abbrevation_list + person_info_list + ['dialysis', 'meds', 'Eyes', 'motor',
                                                                                   'verbal', 'urine', 'wbc',
                                                                                   'Temperature',
                                                                                   'respiratoryrate', 'sodium',
                                                                                   'heartrate', 'meanbp', 'ph',
                                                                                   'hematocrit', 'creatinine',
                                                                                   'albumin', 'pao2',
                                                                                   'pco2', 'bun', 'glucose',
                                                                                   'bilirubin', 'FIO2'] + extra_list
aps_part_header = drug_detail_list + diagnosis_list + person_info_list + ['dialysis', 'meds', 'Eyes', 'motor', 'verbal',
                                                                          'urine', 'wbc', 'Temperature',
                                                                          'respiratoryrate', 'sodium', 'heartrate',
                                                                          'meanbp', 'ph', 'hematocrit', 'creatinine',
                                                                          'albumin', 'pao2', 'pco2', 'bun', 'glucose',
                                                                          'bilirubin', 'FIO2']
ratio_list = ['Hemoglobin', 'hematocrit', 'FIO2', 'bands', 'eos', 'basos']

# spo2,peak Insp Pressure
paper_recovery = ['Eyes_median', 'P/F ratio_median', 'Mean Airway Pressure_median', 'vasopressin', 'FIO2_median',
                  'mean_PAP_median', 'GCS_median', 'CPUO', 'Plateau Pressure_median', 'diastolic_PAP_median',
                  'Overdose', 'norepinephrine', 'Verbal_median', 'Neurologic']
# peak Insp Pressure
paper_long_stay = ['Eyes_median', 'P/F ratio_median', 'mean_PAP_median', 'Mean Airway Pressure_median', 'FIO2_median',
                   'vasopressin', 'diastolic_PAP_median', 'CPUO', 'Overdose', 'Plateau Pressure_median',
                   'norepinephrine', 'GCS_median', 'Verbal_median', 'milrinone', 'Neurologic']
paper_death = ['Eyes_median', 'CO2_median', 'Base Excess_median', 'bicarbonate_median', 'admission_score', 'GCS_median',
               'pH_median', 'vasopressin', 'I_BP_systolic_median', 'FIO2_median', 'admitsource', 'P/F ratio_median',
               'I_BP_mean_median', 'NI_BP_systolic_median', 'Verbal_median', 'mean_PAP_median']

time_series_header = ['id', 'time'] + drug_list + diagnosis_abbrevation_list + person_info_list + dynamic_detail_list + \
                     ['outcome']

# 画图的标记点现状
marks = ['X', 'o', 'v', 's', 'p', 'P', '*', 'h', 'H', 'D', 'd', '>', '<', '^', '_', '1', '2', '3', '4', '.', '+',
         'x']
# 常用线形
linestyle = ['-', '–', '-', ':']
colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm', 'gray', 'brown', 'tomato', 'chocolate', 'deepskyblue', 'deeppink']
# colors = ['r', 'orange', 'b', 'lawngreen', 'gold', 'olive', 'goldenrod', 'wheat', 'khaki', 'lemonchiffon',
#           'antiquewhite', 'honeydew', 'lightcyan', 'lavenderblush']
# 常见颜色
base_colors = [
    'b',
    # 蓝色
    'g',
    # 绿色
    'r',
    # 红色
    'c',
    # 青色
    'm',
    # 品红
    'y',
    # 黄色
    '#1f77b4',
    # 白色
    'lime',
    'orange',
    'cyan',
    'magenta',
    'olive',
    'dodgerblue',
    # 黑色
    'k'
]
model_names = ['GBDT', 'XGBoost', 'RF', 'adaboost', 'bagging', 'stacking', 'BayesianRidge', 'LR', 'linearRegression',
               'Perceptron']

final_item_dict = {'CPUO': '胸痛', 'Hemoglobin_changerate': '血红蛋白', 'albumin_median': '白蛋白', 'TV_median': '潮气量',
                   'glucose_variances': '葡萄糖', 'calcium_changerate': '钙', 'lactate_changerate': '乳酸',
                   'WBC_median': '白细胞', 'creatinine_variances': '肌酐', 'pH_median': '酸碱度', 'dopamine': '多巴胺',
                   'P/F ratio_median': '氧合指数', 'Mean Airway Pressure_median': '平均气道压', 'creatinine_median': '肌酐',
                   'ALT_changerate': '丙氨酸转氨酶', 'BMI': '体重指数', 'magnesium_changerate': '镁', 'vasopressor': '血管抑制剂',
                   'Base Excess_median': '碱过剩', 'Temperature_variances': '温度', 'Trauma': '创伤',
                   'glucose_changerate': '葡萄糖', 'TV_changerate': '潮气量', 'Base Excess_variances': '碱过剩',
                   'albumin_changerate': '白蛋白', 'bicarbonate_changerate': '碳酸氢盐', 'magnesium_median': '镁',
                   'bands_changerate': '中性粒细胞', 'paO2_median': '血氧分压', 'TV_variances': '潮气量', 'epinephrine': '肾上腺素',
                   'Heart Rate_variances': '心率', 'creatinine_changerate': '肌酐', 'Hemoglobin_median': '血红蛋白',
                   'BUN_changerate': '尿素氮', 'sodium_median': '钠', 'INR_changerate': '国际标准化比值', 'AST_median': '谷草转氨酶',
                   'PEEP_variances': '呼吸末正压', 'BUN_median': '尿素氮', 'ionized calcium_variances': '离子钙', 'Coma': '昏迷',
                   'ARF': '肾衰竭', 'bilirubin_median': '胆红素', 'Cardiogenic_Shock': '心源性休克', 'Asthma_Emphysema': '哮喘_肺气肿',
                   'ALT_variances': '丙氨酸转氨酶', 'dobutamine': '多巴胺', 'Temperature_changerate': '温度', 'gender': '性别',
                   'Sepsis': '败血症', 'FIO2_median': '吸入氧浓度', 'potassium_median': '钾',
                   'ionized calcium_changerate': '电离钙', 'potassium_changerate': '钾', 'ionized calcium_median': '电离钙',
                   'vasopressin': '血管加压素', 'paO2_changerate': '血氧分压', 'norepinephrine': '去甲肾上腺素',
                   'Heart Rate_median': '心率', 'bicarbonate_variances': '碳酸氢盐', 'WBC_variances': '白细胞',
                   'GI_Bleed': '消化道出血', 'albumin_variances': '白蛋白', 'basos_median': '嗜碱性粒细胞',
                   'bilirubin_changerate': '胆红素', 'milrinone': '米力农', 'Neurologic': '神经系统', 'phenylephrine': '苯肾上腺素',
                   'Heart Rate_changerate': '心率', 'PEEP_changerate': '呼吸末正压', 'BUN_variances': '尿素氮',
                   'paO2_variances': '血氧分压', 'Mean Airway Pressure_changerate': '平均气道压', 'heparin': '肝素',
                   'eos_median': '嗜酸性粒细胞', 'warfarin': '华法林', 'calcium_median': '钙', 'INR_variances': '国际标准化比值',
                   'PEEP_median': '呼吸末正压', 'Mean Airway Pressure_variances': '平均气道压', 'Temperature_median': '温度',
                   'sodium_changerate': '钠', 'AST_changerate': '谷草转氨酶', 'potassium_variances': '钾',
                   'bilirubin_variances': '胆红素', 'INR_median': '国际标准化比值', 'age': '年龄', 'calcium_variances': '钙',
                   'WBC_changerate': '白细胞', 'magnesium_variances': '镁', 'bands_median': '中性粒细胞', 'Pneumonia': '肺炎',
                   'lactate_variances': '乳酸', 'pH_changerate': '酸碱度值', 'admitsource': '入院来源', 'Valve_Disease': '瓣膜病',
                   'glucose_median': '葡萄糖', 'ALT_median': '丙氨酸转氨酶', 'AST_variances': '谷草转氨酶', 'pH_variances': '酸碱度',
                   'sodium_variances': '钠', 'bicarbonate_median': '碳酸氢盐', 'lactate_median': '乳酸'}

univariate_header = drug_list + diagnosis_abbrevation_list + person_info_list + dynamic_median_list
result_header = ['id'] + drug_list + diagnosis_abbrevation_list + person_info_list + dynamic_detail_list + extra_list
xiu_header = ['id'] + drug_list + diagnosis_abbrevation_list + person_info_list + special_list(dynamic_item_list,
                                                                                               ['median', 'std',
                                                                                                'mean']) + extra_list
outcome_dict = {
    'Spontaneous Recovery': 0,
    'Long Stay': 1,
    'Rapid Death': 2
}

apache_header = ['id', 'gender', 'teachtype', 'admitsource', 'age', 'admitdiagnosis', 'meds', 'thrombolytics',
                 'diedinhospital', 'aids', 'hepaticfailure', 'lymphoma', 'metastaticcancer', 'leukemia',
                 'immunosuppression', 'cirrhosis', 'electivesurgery', 'activetx', 'readmit', 'ima', 'midur', 'ventday1',
                 'oobintubday1', 'oobventday1', 'diabetes', 'dischargelocation', 'visitnumber', 'amilocation',
                 'sicuday', 'saps3day1', 'saps3today', 'saps3yesterday', 'verbal', 'motor', 'eyes', 'pao2', 'fio2',
                 'creatinine', 'day1meds', 'day1verbal', 'day1motor', 'day1eyes', 'day1pao2', 'day1fio2']

apacheiv_header = ['age', 'albumin', 'los', 'mortality prediction', 'natural antilog', 'bilirubin', 'bun', 'chronic',
                   'creatinine', 'fio2', 'gcseye', 'gcsmotor', 'gcs', 'gcsverbal', 'glucose', 'hematocrit', 'hr', 'ht',
                   'intubated', 'map', 'oxygen', 'pco2', 'ph', 'phpaco2', 'po2', 'rr', 'sodium', 'temperaturef',
                   'temp', 'urine', 'wbc']
