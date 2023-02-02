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
from filter.common import read_file, concat_array
from filter.param import *
from matplotlib import ticker

global dataset_names
dataset_names = ['eICU', 'MIMIC III', 'MIMIC IV', 'ARDset']


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
        # self.plot_stacked_histogram(data)
        # self.patient_consist(data)
        # self.plot_hospital_stay(data)
        # self.admission_source_word_cloud_plot(data)
        self.admission_source_plot(data, vertical=True)
        # severity = data['severity']
        # details = data['detail']
        # self.filter_death_by_severity(details, severity)
        # self.severity_plot(data)
        # ages = data['age']
        # self.age_plot(ages)
        # diseases = data.iloc[:, 10:36]
        # self.disease_plot(diseases)
        # apacheivs = data['admission_score']
        # self.apacheIV_plot(apacheivs)
        # unitstays = data['unit']
        # hosstays = data['hospital']
        # self.stay_boxplot(unitstays, hosstays)
        # admitsource = data['admitsource']
        # self.admit_source(admitsource)

    def admission_source_word_cloud_plot(self, data):
        import jieba  # 导入jieba分词模块
        import wordcloud
        import imageio.v2 as imageio
        from matplotlib import colors as mat_colors
        def mark_to_chinese_word(dataset_name, sub_data):
            final_txt_list = ''
            for column_mark, chinese_word in zip(diagnosis_abbrevation_list, diagnosis_chinese_list):
                sub_disease = sub_data[column_mark]
                mark_sum = len([item for item in np.array(sub_disease) if item == 1])
                final_txt_list = final_txt_list + (column_mark + ',') * mark_sum
            with open('pictures/' + dataset_name + '_diagnosis_chinese_word_cloud.txt', 'w') as f:
                f.write(final_txt_list)

        fig = plt.figure()
        sub_plots = [221, 222, 223, 224]
        pic = imageio.imread('pictures/img.png')
        pic = np.array(pic)
        for i, dataset_name, sub_plot in zip(range(len(data)), dataset_names, sub_plots):
            fig.add_subplot(sub_plot)
            sub_data = data[i][diagnosis_abbrevation_list]
            mark_to_chinese_word(dataset_name, sub_data)
            data_word_txt = open('pictures/' + dataset_name + '_diagnosis_chinese_word_cloud.txt', 'r').read()
            colormap = mat_colors.ListedColormap(colors)
            wc = wordcloud.WordCloud(mask=pic, font_path='SimHei.ttf', width=800, height=600, max_words=40, font_step=3,
                                     colormap=colormap, background_color='white', collocations=False)
            wc.generate(data_word_txt)
            plt.imshow(wc)
        plt.savefig('admission_source.eps', format='eps', dpi=600)
        self.save_pic('admission_source')
        plt.show()

    def admission_source_plot(self, data, vertical=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(zorder=0)
        from pylab import mpl
        # mpl.rcParams['font.sans-serif'] = ['SimHei']

        def trans_admission_word_count(sub_data, selected=False, n=0):
            chinese_word_dict = {}
            english_word_dict = {}
            # mpl.rcParams['font.sans-serif'] = ['Times New Roman']
            nums = []
            for item in sub_data:
                nums.append(len([item for item in np.array(sub_data[item]) if item]))
            for chinese, english, num in zip(diagnosis_chinese_list, diagnosis_abbrevation_list, nums):
                chinese_word_dict[chinese] = num
                english_word_dict[english] = num
            chinese_word_dict = {k: round(v / len(sub_data), 3) for k, v in chinese_word_dict.items()}
            english_word_dict = {k: round(v / len(sub_data), 3) for k, v in english_word_dict.items()}
            # sort by value
            chinese_word_dict = list(chinese_word_dict.items())
            english_word_dict = list(english_word_dict.items())
            chinese_word_dict.sort(key=lambda x: x[1], reverse=True)
            english_word_dict.sort(key=lambda x: x[1], reverse=True)
            # sort by key
            # chinese_word_dict.sort(key=lambda x: x[0], reverse=True)
            # english_word_dict.sort(key=lambda x: x[0], reverse=True)
            if selected:
                chinese_word_dict = chinese_word_dict[:n]
                english_word_dict = english_word_dict[:n]
            chinese_word_dict = {item[0]: item[1] for item in chinese_word_dict}
            english_word_dict = {item[0]: item[1] for item in english_word_dict}
            return chinese_word_dict, english_word_dict

        rates = [-1.5, -0.5, 0.5, 1.5]

        colors = ['red', 'orange', 'yellowgreen', 'deepskyblue']
        # mpl.rcParams['font.sans-serif'] = ['SimHei']
        if vertical:
            height = 0.2
            for sub_data, name, rate, color in zip(data, dataset_names, rates, colors):
                sub_admission = sub_data[diagnosis_abbrevation_list]
                chinese, english = trans_admission_word_count(sub_admission, True, 5)
                y = np.arange(5, 0, -1) + rate * height
                mpl.rcParams['font.sans-serif'] = ['Time New Roman']
                plt.barh(y, list(english.values()), height=height, label=name, color=color, zorder=5)
                for a, b, label in zip(y, list(english.values()), english.keys()):
                    plt.text(b, a, label, verticalalignment='center', zorder=6)
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            plt.xlabel('Percentage of Admission Diagnosis', fontweight='bold', fontsize=15,
                       fontproperties='Times New Roman')
            plt.ylabel('Admission Diagnosis', fontweight='bold', fontsize=15, fontproperties='Times New Roman')
            plt.xticks(np.arange(0, 0.85, 0.2))
        else:
            width = 0.2
            for sub_data, name, rate, color in zip(data, dataset_names, rates, colors):
                sub_admission = sub_data[diagnosis_abbrevation_list]
                chinese, english = trans_admission_word_count(sub_admission, True, 5)
                x = np.arange(len(english))
                mpl.rcParams['font.sans-serif'] = ['Time New Roman']
                plt.bar(x + rate * width, english.values(), width, yerr=0.005, label=name, color=color, zorder=5,
                        error_kw={'ecolor': '0.2', 'capsize': 6})
                for a, b, label in zip(x + rate * width, list(english.values()), english.keys()):
                    # plt.text(a, b, label, rotation=30, wrap=False)
                    plt.text(a, b, label, rotation=-90, wrap=True, verticalalignment='center', zorder=6)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
            plt.ylabel('Percentage of Admission Diagnosis', fontweight='bold', fontsize=15)
            plt.xlabel('Admission Diagnosis', fontweight='bold', fontsize=15)
        plt.title('Top 5 Admission Diagnosis for Each Dataset', fontweight='bold', fontsize=15)
        plt.legend(labels=dataset_names, loc=4)
        plt.savefig('admission.svg', format='svg')
        plt.show()

    # plot stacked histogram for each datasets and group them up into a union
    # x—axis:eICU,MIMIC III,MIMIC IV and ARDset
    # y-axis:split patient age into 18-25,25-40,40-60,60-89,and compute patient rate in each age gap
    def plot_stacked_histogram(self, data):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False

        def plot_each_stacked(stacks, stack_names, stack_label):
            """
            @param stacks: current age group
            @param stack_names: item names, such as eICU,MIMIC III, MIMIC IV and ARDset.
            @param stack_label: age group name
            """
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.grid(zorder=0)
            # color_base = np.array([128, 4, 128])
            red = 125
            origin_green = 200
            blue = 255
            colors = ['coral', 'sandybrown', 'gold', 'orange']
            colors = ['gold', 'sandybrown', 'orange', 'darkorange', 'red']
            # colors.reverse()
            # colors = ['brown', 'gray', 'olive', 'pink']
            for i, color in zip(range(len(stacks)), colors):
                if i > 0:
                    bottom = []
                    for k in range(len(stacks[i])):
                        bottom.append(sum(stacks[:i, k]))
                else:
                    bottom = [0] * len(stacks[i])
                green = origin_green - 60 * i
                # ax.bar(stack_names, stacks[i], width=0.45, bottom=bottom, label=stack_label[i],
                #        color=np.array([red, green, blue]) / 255)
                ax.bar(stack_names, stacks[i], width=0.45, bottom=bottom, label=stack_label[i],
                       color=color, zorder=3)
                for i, num, height in zip(range(len(stacks[i])), stacks[i], bottom):
                    if num > 0:
                        ax.text(i, height + num / 2, '%.0f' % num, ha='center', va='top', zorder=3, fontsize=15)
            ax.set_ylabel('Percentage of ARDS Patients', fontweight='bold', fontsize=20,
                          fontproperties='Times New Roman')
            ax.set_title('Age Distribution in ARDS Datasets', fontweight='bold', fontproperties='Times New Roman',
                         fontsize=20)
            # ax.set_xlabel('Datasets', fontweight='bold', fontsize=20, fontproperties='Times New Roman')
            # 坐标轴显示百分比，若实际数值为小于1的小数，xmax设置为1，即xmax=1
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
            plt.xticks(fontproperties='Times New Roman')
            plt.yticks(fontproperties='Times New Roman')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
            plt.savefig('age_distribution.emf', format='emf')
            # self.save_pic('age_distribution')
            plt.show()

        def compute_patient_rate_in_each_gap(data):
            """
            @param data: data group
            @return: four arrays which respectively contain four age gap groups
            """
            sum = len(data)
            group_18 = []
            group_33 = []
            group_48 = []
            group_63 = []
            group_78 = []
            for i in range(sum):
                current_group = data[i]['age']
                current_sum = len([item for item in current_group if item > 0])
                age_18 = 100 * len([item for item in current_group if item >= 18 and item < 33]) / current_sum
                age_33 = 100 * len([item for item in current_group if item >= 33 and item < 48]) / current_sum
                age_48 = 100 * len([item for item in current_group if item >= 48 and item < 63]) / current_sum
                age_63 = 100 * len([item for item in current_group if item >= 63 and item < 78]) / current_sum
                age_78 = 100 * len([item for item in current_group if item >= 78]) / current_sum
                group_18.append(round(age_18, 1))
                group_33.append(round(age_33, 1))
                group_48.append(round(age_48, 1))
                group_63.append(round(age_63, 1))
                group_78.append(round(age_78, 1))
            return group_18, group_33, group_48, group_63, group_78

        x_labels = ['eICU', 'MIMIC III', 'MIMIC IV', 'ARDset']
        group_18, group_33, group_48, group_63, group_78 = compute_patient_rate_in_each_gap(data)
        stacks = np.array([group_18, group_33, group_48, group_63, group_78])
        plot_each_stacked(stacks, x_labels,
                          ['18-33 years', '33-48 years', '48-63 years', '63-78 years', '>=78  years'])

    def patient_consist(self, data):
        def plot_pie_graph(fig, ax1_num, ax2_num, one_classes, one_data, two_classes, two_data):
            from matplotlib.patches import ConnectionPatch
            # 制画布
            explode = (0, 0.1)
            ax1 = fig.add_subplot(ax1_num)
            ax2 = fig.add_subplot(ax2_num)
            # fig.subplots_adjust(wspace=0)
            fig.subplots_adjust(left=0.08, bottom=0.0, right=0.96, top=0.99, wspace=0.01, hspace=0.05)
            # 大饼图的制作
            ax1.pie(one_data, autopct='%1.1f%%', startangle=60, radius=1.2, explode=explode,
                    labels=one_classes, colors=['lightgreen', 'lightcoral'], pctdistance=0.45,
                    textprops={'fontsize': 20, 'color': 'black'}, labeldistance=0.75)
            # l_text是饼图对着文字大小，p_text是饼图内文字大小
            # for l, p in zip(l_text, p_text):
            #     l.set_size(30)
            #     p.set_size(20)

            # 小饼图的制作
            width = 0.8
            patches, l_text, p_text = ax2.pie(two_data, autopct='%1.1f%%', startangle=90, labels=two_classes,
                                              labeldistance=1.3, radius=0.5, shadow=True,
                                              colors=['brown', 'tomato', 'coral'])
            for l, p in zip(l_text, p_text):
                l.set_size(18)
                p.set_size(15)

            # 使用ConnectionPatch画出两个饼图的间连线
            # 先得到饼图边缘的数据
            theta1, theta2 = ax1.patches[len(one_classes) - 1].theta1, ax1.patches[len(one_classes) - 1].theta2
            center, r = ax1.patches[len(one_classes) - 1].center, ax1.patches[len(one_classes) - 1].r

            # 画出上边缘的连线
            x = r * np.cos(np.pi / 180 * theta2) + center[0]
            y = np.sin(np.pi / 180 * theta2) + center[1]
            con = ConnectionPatch(xyA=(-width / 2, 0.5), xyB=(x, y), coordsA='data', coordsB='data', axesA=ax2,
                                  axesB=ax1)
            con.set_linewidth(1)
            con.set_color = ([0, 0, 0])
            ax2.add_artist(con)

            # 画出下边缘的连线
            x = r * np.cos(np.pi / 180 * theta1) + center[0]
            y = np.sin(np.pi / 180 * theta1) + center[1]
            con = ConnectionPatch(xyA=(-width / 2, -0.5), xyB=(x, y), coordsA='data', coordsB='data', axesA=ax2,
                                  axesB=ax1)
            con.set_linewidth(1)
            con.set_color = ([0, 0, 0])
            ax2.add_artist(con)

        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        fig = plt.figure(figsize=(12, 7.5))
        plt_num = [241, 242, 243, 244, 245, 246, 247, 248]
        dataset_names = ['eICU', 'MIMIC III', 'MIMIC IV', 'ARDset']
        for i, name in zip(range(len(data)), dataset_names):
            sub_data = data[i]
            one_classes = ['alive', 'expired']
            patients = len(np.array(sub_data))
            two_classes = ['severe', 'moderate', 'mild']
            alive_rate = 100 * len([item for item in sub_data['detail'] if item == 3]) / patients
            one_data = [round(alive_rate, 1), round(100 - alive_rate, 1)]
            patient_detail = np.array(sub_data[['detail', 'severity']])
            expired_severes = len([out for out, ser in patient_detail if out != 3 and ser == 1])
            expired_moderates = len([out for out, ser in patient_detail if out != 3 and ser == 2])
            expired_milds = len([out for out, ser in patient_detail if out != 3 and ser == 3])
            two_data = [expired_severes, expired_moderates, expired_milds]
            plot_pie_graph(fig, plt_num[2 * i], plt_num[2 * i + 1], one_classes, one_data, two_classes, two_data)
            plt.title(name, fontweight='bold', fontsize=30, loc='left')
        # 不显示坐标刻度
        # plt.xaxis.set_visible(False)
        # plt.yaxis.set_visible(False)
        # plt.title('Patients Condition and the Severity of Expired Patients', fontproperties='Times New Roman',
        #           fontsize=25, loc='left')
        plt.savefig('patient_condition.svg', format='svg')
        # self.save_pic('patient_condition')
        plt.show()

    def plot_hospital_stay(self, data):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        dataset_names = ['eICU', 'MIMIC III', 'MIMIC IV', 'ARDset']
        stay_type = ['Hospital stay', 'ICU stay']
        colors = ['red', 'blue', 'green', 'purple']
        fig = plt.figure()
        plt.xticks([]), plt.yticks([])
        fig.add_subplot(121)
        for i, color, name in zip(range(len(data)), colors, dataset_names):
            hospital_data = data[i][['hospital']]
            sns.distplot(hospital_data, vertical=True, hist=False, kde=False, fit=stats.norm,
                         fit_kws={'color': color, 'linestyle': '-'}, kde_kws={'label': name + '_' + stay_type[0]})
        plt.ylabel('Days', fontsize=20, fontweight='bold')
        plt.xlabel('Density', fontsize=20, fontweight='bold')
        plt.xticks(np.arange(0, 0.062, 0.02))
        plt.xlim([0, 0.062])
        plt.yticks(np.arange(0, 62, 20))
        plt.grid()
        plt.ylim([0, 62])
        plt.legend(labels=dataset_names, title='Hospital Stay')
        fig.add_subplot(122)
        for i, color, name in zip(range(len(data)), colors, dataset_names):
            unit_data = data[i][['unit']]
            sns.distplot(unit_data, vertical=True, hist=False, kde=False, fit=stats.norm,
                         kde_kws={'label': name + '_' + stay_type[1]},
                         fit_kws={'color': color, 'linestyle': 'dashed'})
        plt.xlabel('Density', fontsize=20, fontweight='bold')
        plt.xticks(np.arange(0, 0.062, 0.02))
        plt.xlim([0, 0.062])
        plt.yticks(np.arange(0, 62, 20))
        plt.ylim([0, 62])
        plt.legend(labels=dataset_names, title='ICU Stay')
        plt.grid()
        plt.savefig('hospital_stay.svg', format='svg')
        # self.save_pic('hospital_stay')
        plt.show()

    def save_pic(self, name):
        self.name = name
        self.path = self.name + '.svg'
        self.path = os.path.join(self.basepath, self.path)
        plt.savefig(self.path)
        print('%s saved in %s ' % (name, self.path))


if __name__ == '__main__':
    base_path = 'D:\pycharm\ARDS-prognosis-for-eICU-data\ARDS'
    total_path = os.path.join('combine', 'csvfiles')
    eicu = read_file(path=os.path.join(base_path, total_path), filename='total_eicu')
    mimic3 = read_file(path=os.path.join(base_path, total_path), filename='total_mimic3')
    mimic4 = read_file(path=os.path.join(base_path, total_path), filename='total_mimic4')
    total = read_file(path=os.path.join(base_path, total_path), filename='total_data')
    data = [eicu, mimic3, mimic4, total]
    ages = [eicu['age'], mimic3['age'], mimic4['age'], total['age']]
    analyser = analysis_plot()
    # analyser.plot_images(data=DataFrame(ages), name='total')
    analyser.plot_images(data=data)
