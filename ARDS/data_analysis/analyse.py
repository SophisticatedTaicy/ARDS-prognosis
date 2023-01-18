import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
from scipy import stats

import filter.param
from filter.common import read_file
from filter.param import *


class analysis_plot:
    def __init__(self, name=None):
        # 存储的图片名称？
        if name:
            self.name = name
        self.basepath = 'D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS\data_analysis\pictures'

    def num_to_dict(self, num):
        dict = {}
        i = 0
        while num > 1:
            dict[i] = num % 10
            i += 1
            num = int(num / 10)
        return dict

    # 患者死亡以及转归情况统计
    def filter_death_by_severity(self, details, severitys):
        '''
        :param severities: ARDS严重程度，1 重度，2 中度，3 轻度，4未知（）
        :param details: 患者死亡状态，-1 患者活着，0 ICU死亡，1 28天内死亡，2 医院死亡
        :return:患者死亡情况展示
        '''
        severitys = np.array(severitys)
        icu_death_mild = 0
        icu_death_moderate = 0
        icu_death_severity = 0
        day28_death_mild = 0
        day28_death_moderate = 0
        day28_death_severity = 0
        hospital_death_mild = 0
        hospital_death_moderate = 0
        hospital_death_severity = 0
        death = 0
        for severity, detail in zip(severitys, details):
            if '1' in str(detail):
                # ICU死亡
                if severity == 3:
                    icu_death_mild += 1
                elif severity == 2:
                    icu_death_moderate += 1
                elif severity == 1:
                    icu_death_severity += 1
            if '2' in str(detail):
                # 28天内死亡
                if severity == 3:
                    day28_death_mild += 1
                elif severity == 2:
                    day28_death_moderate += 1
                elif severity == 1:
                    day28_death_severity += 1
            if '4' in str(detail):
                # 医院死亡
                if severity == 3:
                    hospital_death_mild += 1
                elif severity == 2:
                    hospital_death_moderate += 1
                elif severity == 1:
                    hospital_death_severity += 1
            if '3' not in str(detail):
                death += 1
        print('total death : %s ' % death)
        mild_rate = [hospital_death_mild / death, day28_death_mild / death, icu_death_mild / death]
        mild_rate = [item * 100 for item in mild_rate]
        moderate_rate = [hospital_death_moderate / death, day28_death_moderate / death, icu_death_moderate / death]
        moderate_rate = [item * 100 for item in moderate_rate]
        severe_rate = [hospital_death_severity / death, day28_death_severity / death, icu_death_severity / death]
        severe_rate = [item * 100 for item in severe_rate]
        name_list = ['Hospital Mortality', '28-day Mortality', 'ICU Mortality']
        label_list = ['Severe', 'Moderate', 'Mild']
        width = 1
        # 误差棒属性
        # ecolor：定义误差棒的颜色
        # capsize：定义误差棒帽的大小（长度）
        # yerr ：定义y轴方向的误差棒的大小
        error_kw = {'ecolor': '0.1', 'capsize': 6}
        # 误差大小
        plt.figure(dpi=500)
        plt.grid(zorder=0)
        mild = [hospital_death_mild, day28_death_mild, icu_death_mild]
        moderate = [hospital_death_moderate, day28_death_moderate, icu_death_moderate]
        severe = [hospital_death_severity, day28_death_severity, icu_death_severity]
        yerr = np.mean(severe_rate) * 0.005
        x = [0, 3.3, 6.6]
        plt.bar(x, severe_rate, width=width, yerr=yerr, label=label_list[0], error_kw=error_kw, fc='r', zorder=100)
        for i, rate, num in zip(x, severe_rate, severe):
            plt.text(i, rate / 2, '%.0f' % rate + '%' + '\n(n=%d)' % num, ha='center', va='bottom', zorder=100)
        for i in range(len(x)):
            x[i] += width
        plt.bar(x, moderate_rate, width=width, yerr=yerr, label=label_list[1], error_kw=error_kw, tick_label=name_list,
                fc='coral', zorder=100)
        for i, rate, num in zip(x, moderate_rate, moderate):
            if rate > 10:
                plt.text(i, rate / 2, '%.0f' % rate + '%' + '\n(n=%d)' % num, ha='center', va='top', zorder=100)
            elif rate > 0:
                plt.text(i, rate + 5, '%.0f' % rate + '%', ha='center', va='top', zorder=100)
                if rate - 3 > 0:
                    plt.text(i, rate - 1, '(n=%d)' % num, ha='center', va='top', zorder=100)
                else:
                    plt.text(i, 3, '(n=%d)' % num, ha='center', va='top', zorder=100)
        for i in range(len(x)):
            x[i] += width
        plt.bar(x, mild_rate, width=width, yerr=yerr, label=label_list[2], error_kw=error_kw, fc='sandybrown',
                zorder=100)
        for i, rate, num in zip(x, mild_rate, mild):
            if rate > 10:
                plt.text(i, rate / 2, '%.0f' % rate + '%' + '\n(n=%d)' % num, ha='center', va='top', zorder=100)
            elif rate > 0:
                plt.text(i, rate + 5, '%.0f' % rate + '%', ha='center', va='top', zorder=100)
                if rate - 3 > 0:
                    plt.text(i, rate - 1, '(n=%d)' % num, ha='center', va='top', zorder=100)
                else:
                    plt.text(i, 3, '(n=%d)' % num, ha='center', va='top', zorder=100)
        plt.ylabel('Mortality Rate', fontsize=20, fontweight='bold')
        plt.text(0, -6, 'Severity', ha='center', va='top', fontweight='bold')
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc=8, ncol=10, frameon=False)
        plt.ylim(0, 82, 10)
        self.save_pic('total_death')
        plt.show()

    # 将数据转换为百分比
    def to_percent(self, num):
        return '%d' % num + '%'

    # 患者入院诊断统计
    def disease_plot(self, diseases):
        dis_sum = []
        dis_rate = []
        diseases = np.array(diseases)
        # 误差棒属性
        # ecolor：定义误差棒的颜色
        # capsize：定义误差棒帽的大小（长度）
        # yerr ：定义y轴方向的误差棒的大小
        error_kw = {'ecolor': '0.1', 'capsize': 5}
        # 误差大小横向
        xerr = 0.1
        for i in range(26):
            valid_index = np.where(diseases[:, i] == 1)
            num = np.array(valid_index).shape[1]
            rate = round((num / len(diseases)) * 100, 1)
            if rate > 0:
                dis_sum.append(num)
                dis_rate.append(rate)
        # 按照数量排序
        combines = zip(dis_sum, dis_rate, diagnosis_list)
        combines = sorted(combines, reverse=True)
        dis_sum_new, dis_rate_new, names_new = zip(*combines)
        fig, ax = plt.subplots(dpi=500, figsize=(7, 10))
        max_index = min(15, len(names_new))
        b = ax.barh(range(len(names_new[:max_index])), dis_rate_new[:max_index][::-1], xerr=xerr,
                    tick_label=names_new[:max_index][::-1],
                    error_kw=error_kw, color='gray', zorder=1)
        for i, rect in enumerate(b):
            w = rect.get_width()
            if i > 12:
                ax.text(w - 2.5, rect.get_y() + rect.get_height() / 2, '%.1f' % w + '%', ha='left', va='center',
                        zorder=2)
            else:
                ax.text(w + 0.5, rect.get_y() + rect.get_height() / 2, '%.1f' % w + '%', ha='left', va='center',
                        zorder=2)
        plt.xticks(np.arange(0, 22, 5))
        plt.grid()
        # 设置横坐标中%
        # plt.gca().xaxis.set_major_formatter(FuncFormatter(self.to_percent))
        plt.xlabel('% of ICU stays', fontsize=20, fontweight='bold')
        plt.ylabel('Admission Diagnosis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        self.save_pic('diagnosis')
        plt.show()

    # 患者年龄统计
    def age_plot(self, ages):
        '''
        :param ages: 年龄数据
        :return:
        '''
        plt.figure(dpi=500)
        plt.figure(figsize=(6, 8))
        print('average age : ' + str(np.mean(ages)))
        sns.distplot(ages, vertical=True, hist=False, kde=False, fit=stats.norm,
                     fit_kws={'color': 'black', 'linestyle': '-'})
        # 呈现图例
        plt.yticks(np.arange(20, 95, 20))
        plt.xticks(np.arange(0.01, 0.025, 0.01))
        plt.ylabel('Age at admission', fontweight='bold', fontsize=15)
        plt.xlabel('Estimated density', fontweight='bold', fontsize=15)
        plt.legend()
        plt.grid()
        self.save_pic('age')
        # 呈现图形
        plt.show()

    # 患者入院Apache IV分数统计
    def apacheIV_plot(self, apache):
        if not (apache == 0).all():
            print('average apache IV : ' + str(np.mean(apache)))
            '''
            :param ages: 入院Apache分数
            :return:
            '''
            plt.figure(dpi=500, figsize=(6, 8))
            sns.distplot(apache, vertical=True, hist=False, kde=False, fit=stats.norm,
                         fit_kws={'color': 'black', 'linestyle': '-'})
            # 呈现图例
            plt.yticks(np.arange(0, 205, 50))
            plt.ylim([-5, 205])
            plt.xticks(np.arange(0, 0.013, 0.005))
            plt.xlim([-0.001, 0.015])
            plt.ylabel('APACHE IV score at admission', fontweight='bold', fontsize=15)
            plt.xlabel('Estimated density', fontweight='bold', fontsize=15)
            plt.legend()
            plt.grid()
            self.save_pic('apache')
            # 呈现图形
            plt.show()

    # 患者住院情况统计
    def stay_boxplot(self, unit, hospital):
        '''
        :param unit:ICU住院天数
        :param hospital:医院住院天数
        :return:
        '''
        plt.figure(dpi=500, figsize=(6, 7))
        print('average unit stay : ' + str(np.mean(unit)) + ' average hospital stay : ' + str(np.mean(hospital)))
        plt.boxplot((hospital, unit), labels=('Hospital', 'ICU'), widths=0.3, sym='|')
        # 呈现图例
        plt.ylim(-2, 20)
        plt.yticks(np.arange(0, 31, 10))
        plt.xlabel('Stay type', fontweight='bold', fontsize=15)
        plt.ylabel('LOS(days)', fontweight='bold', fontsize=15)
        plt.legend()
        plt.grid()
        self.save_pic('stay')
        # 呈现图形
        plt.show()

    # 患者总体死亡率统计
    def death_plot(self, result):
        '''
        :param death:
        :return:
        '''
        details = np.array(result.iloc[:, -4])
        ICU_death = len([item for item in details if '1' in str(item)])
        day28_death = len([item for item in details if '2' in str(item)])
        Hospital_death = len([item for item in details if '4' in str(item)])
        death = len(details)
        death_rate = [Hospital_death / death, day28_death / death, ICU_death / death]
        death_rate = [item * 100 for item in death_rate]
        death_sum = [Hospital_death, day28_death, ICU_death]
        x = [0, 0.6, 1.2]
        # 误差棒属性
        error_kw = {'ecolor': '0.2', 'capsize': 6}
        # 误差大小
        yerr = np.mean(death_rate) * 0.005
        plt.figure(dpi=500)
        plt.grid(zorder=0)
        name_list = ['Hospital Mortality', '28-day Mortality', 'ICU Mortality']
        # 每根柱子显示不同颜色，不同标记
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.to_percent))
        plt.bar(x, death_rate, width=0.5, yerr=yerr, error_kw=error_kw, tick_label=name_list, color='gray', zorder=5)
        for i, rate, num in zip(x, death_rate, death_sum):
            plt.text(i, rate / 2, '%.0f' % rate + '%' + '\n(death=%d)' % num, ha='center', va='bottom', zorder=5)
        plt.ylabel('Mortality rate')
        plt.legend()
        self.save_pic('death')
        # 呈现图形
        plt.show()

    # 患者严重程度分析
    def severity_plot(self, result):
        details = np.array(result['detail'])
        severitys = np.array(result['severity'])
        mild = np.array(np.where(severitys == 3)).shape[1]
        moderate = np.array(np.where(severitys == 2)).shape[1]
        severe = np.array(np.where(severitys == 1)).shape[1]
        mild_death = 0
        moderate_death = 0
        severe_death = 0
        for severity, detail in zip(severitys, details):
            if severity == 3 and detail != 3:
                mild_death += 1
            if severity == 2 and detail != 3:
                moderate_death += 1
            if severity == 1 and detail != 3:
                severe_death += 1
        death_rates = [severe_death / severe, moderate_death / moderate, mild_death / mild]
        death_rates = [item * 100 for item in death_rates]
        deaths = [severe_death, moderate_death, mild_death]
        x = [0, 0.12, 0.24]
        name_list = ['Severe', 'Moderate', 'Mild']
        # 每根柱子显示不同颜色，不同标记
        # 误差棒属性
        error_kw = {'ecolor': '0.2', 'capsize': 6}
        # 误差大小
        plt.figure(dpi=500)
        plt.grid(zorder=0)
        yerr = np.mean(death_rates) * 0.005
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(self.to_percent))
        plt.bar(x, death_rates, width=0.1, yerr=yerr, error_kw=error_kw, tick_label=name_list,
                color=['r', 'coral', 'sandybrown'], zorder=5)
        for i, rate, num in zip(x, death_rates, deaths):
            plt.text(i, rate / 2, '%.0f' % rate + '%' + '\n(death=%d)' % num, ha='center', va='bottom', zorder=5)
        plt.ylabel('Mortality rate')
        plt.legend()
        plt.ylim(0, 42, 10)
        self.save_pic('severity')
        # 呈现图形
        plt.show()

    # 患者入院来源展示
    def admit_source(self, source_datas):
        source_datas = np.array(source_datas)
        sum = len(source_datas)
        final_dict = {}
        source_dict = {'Emergency\nDepartment': 1, 'Operating\nRoom': 2, 'Direct\nAdmit': 3, 'Floor': 4, 'other': 5}
        num_rates = {}
        for name, value in source_dict.items():
            num = len(np.argwhere(source_datas == value))
            rate = (num / sum) * 100
            num_rates[num] = rate
            final_dict[name] = rate
        final_dict = dict(
            sorted({value: key for key, value in final_dict.items()}.items(), key=lambda item: item[0], reverse=True))
        num_rates = dict(
            sorted({value: key for key, value in num_rates.items()}.items(), key=lambda item: item[0], reverse=True))
        # 误差棒属性
        error_kw = {'ecolor': '0.2', 'capsize': 6}
        # 误差大小
        yerr = 1
        # 柱状图显示
        plt.figure(dpi=500)
        plt.grid(zorder=0)
        names = [str(value) for key, value in final_dict.items() if key > 0]
        rates = [int(key) for key, value in final_dict.items() if key > 0]
        nums = [int(value) for key, value in num_rates.items() if key > 0]
        x = range(len(names))
        plt.bar(x, rates, width=0.5, yerr=yerr, error_kw=error_kw, tick_label=names, color='gray', zorder=5)
        for i, rate, num in zip(x, rates, nums):
            plt.text(i, rate / 2, '%.0f' % rate + '%' + '\n(n=%d)' % num, ha='center', va='top', zorder=5)
        plt.ylim([-2, max(rates) + 1])
        plt.yticks(np.arange(0, 46, 15))
        plt.ylabel('Patient number')
        self.save_pic('admit_source')
        # 呈现图形
        plt.show()

    # 选择所有急诊入院患者数据
    def select_emergency(self, ):
        columns = []
        with open('../eicu/pictures/0827result/data_new.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[37] == 0:
                    # 写入急诊转入数据
                    columns.append(row)
        dataframe = DataFrame(columns)
        # 数据写入
        dataframe.to_csv('../result/emergency.csv', mode='w', encoding='utf-8', index=False)
        # 表头写入
        with open('../eicu/pictures/0827result/emergency.csv', mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=filter.param.result_header)
            writer.writeheader()

    # 按照原文单变量分析结果抽取对应预后的数据信息
    def extract_paper_univariate_features(self, data):
        recovery = filter.param.recovery
        long_stay = filter.param.long_stay
        death = filter.param.death
        # 快速恢复
        recovery_data = data[recovery[0]]
        for i in range(1, len(recovery)):
            recovery_data = pd.concat([recovery_data, data[recovery[i]]], axis=1)
        recovery_data = pd.concat([recovery_data, data['status']], axis=1)
        recovery_data = np.array(recovery_data)
        # 长期住院
        long_stay_data = data[long_stay[0]]
        for i in range(1, len(long_stay)):
            long_stay_data = pd.concat([long_stay_data, data[long_stay[i]]], axis=1)
        long_stay_data = pd.concat([long_stay_data, data['status']], axis=1)
        long_stay_data = np.array(long_stay_data)
        # 快速死亡
        death_data = data[death[0]]
        for i in range(1, len(death)):
            death_data = pd.concat([death_data, data[death[i]]], axis=1)
        death_data = pd.concat([death_data, data['status']], axis=1)
        death_data = np.array(death_data)
        return recovery_data, long_stay_data, death_data

    def plot_images(self, data, name=None):
        if name:
            self.name = name
        severity = data['severity']
        details = data['detail']
        self.filter_death_by_severity(details, severity)
        self.severity_plot(data)
        ages = data['age']
        self.age_plot(ages)
        diseases = data.iloc[:, 10:36]
        self.disease_plot(diseases)
        apacheivs = data['admission_score']
        self.apacheIV_plot(apacheivs)
        unitstays = data['unit']
        hosstays = data['hospital']
        self.stay_boxplot(unitstays, hosstays)
        admitsource = data['admitsource']
        self.admit_source(admitsource)

    def save_pic(self, type):
        self.path = self.name + '_' + type + '.jpg'
        self.path = os.path.join(self.basepath, self.path)
        plt.savefig(self.path)


if __name__ == '__main__':
    base_path = 'D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS'
    # eicu_path = os.path.join('eicu', 'csvfiles')
    # mimic3_path = os.path.join('mimic', 'mimic3', 'csvfiles')
    # mimic4_path = os.path.join('mimic', 'mimic4', 'csvfiles')
    # base_name = 'result_1207'
    # eicu = read_file(path=os.path.join(base_path, eicu_path), filename=base_name)
    # mimic3 = read_file(path=os.path.join(base_path, mimic3_path), filename=base_name)
    # mimic4 = read_file(path=os.path.join(base_path, mimic4_path), filename=base_name)
    # analyser = analysis_plot(name='eicu')
    # analyser.plot_images(name='eicu', data=eicu)
    # analyser.plot_images(name='mimic3', data=mimic3)
    # analyser.plot_images(name='mimic4', data=mimic4)
    total_path = os.path.join('combine', 'csvfiles')
    total = read_file(path=os.path.join(base_path, total_path), filename='total_data')
    analyser = analysis_plot()
    analyser.plot_images(data=total, name='total')
