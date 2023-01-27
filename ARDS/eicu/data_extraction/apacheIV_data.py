import threading

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

import filter.param
from ARDS import init
from ARDS.eicu.data_extraction import query_sql
from dimension_reduction.univariate_analyse import plot_univariate_analysis
from train_AUC import get_mapfunction


class MyThread(threading.Thread):
    def __init__(self, name, data):
        super(MyThread, self).__init__(name=name)
        self.name = name
        self.data = data
        self.is_first = True

    def run(self):
        print('%s区间数据在运行中-----------' % str(self.name))
        for item in self.data:
            # 住院记录对应的220维数据信息抽取
            apache_header = filter.param.apache_header
            header = init.init_dict(header=apache_header)
            # 患者住院记录基本信息
            id = item[0]
            header['id'] = id
            # 获取患者apache数据
            query = query_sql.Query()
            res = query.extra_apacheIV_data_by_id(id, header)
            if res:
                DataFrame([header]).to_csv('../0907/apache.csv', mode='a', encoding='utf-8', header=False, index=False)
                # # # 数据追加写入文件
                print(str(id) + '全部数据写入成功！')
            else:
                print('id : %s not be found!' % id)


# apache异常数据补充0
def fill_with_0(apache):
    '''
    1.入院诊断特征编码
    2.none值填充，浮点值取均值，整数值取众数
    3.fio2数值转换为小数
    4.异常值转换0或均值
    '''
    columns = apache.columns
    print(apache.dtypes)
    category_columns = [x for x in columns if apache[x].dtype is np.dtype('object')]
    # 非数值型数据编码
    for i in category_columns:
        apache[i] = apache[i].apply(get_mapfunction(apache[i]))
    # 含异常值数列查找
    none_columns = [x for x in columns if True in np.isnan(np.array(apache[x]))]
    # none异常值填充
    for i in none_columns:
        sum = np.sum(np.array([x for x in apache[i] if not np.isnan(x) and x > 0]))
        num = len(np.array([x for x in apache[i] if not np.isnan(x) and x > 0]))
        mean = round(sum / num, 3)
        # nan数值替换
        apache[i] = np.nan_to_num(apache[i], nan=mean)
        # <=0数值替换
        apache[i] = [mean if x <= 0 else x for x in np.array(apache[i])]
    # 其他异常值
    float_columns = [x for x in columns if apache[x].dtype == np.float]
    for i in float_columns:
        sum = np.sum(np.array([x for x in apache[i] if not np.isnan(x) and x > 0]))
        num = len(np.array([x for x in apache[i] if not np.isnan(x) and x > 0]))
        mean = round(sum / num, 3)
        apache[i] = [mean if x <= 0 else x for x in np.array(apache[i])]
    fio2_columns = [x for x in columns if 'fio2' in str(x)]
    for i in fio2_columns:
        apache[i] = [round(x / 100, 3) if x > 1 else x for x in np.array(apache[i])]
    header = filter.param.apache_header
    header.append('status')
    apache.to_csv('../0907/apache_deal_1.csv', mode='w', encoding='utf-8', header=header, index=False)


# apache数据异常值补充均值
def fill_with_average(apache):
    '''
    年龄，入院apache以及bmi异常数据修改
    apache变量-1修改
    '''
    apache = np.array(apache)
    row = apache.shape[0]
    for j in range(39, 42):
        valid_num = 0
        sum = 0
        for i in range(row):
            item = apache[i][j]
            if item > 10:
                valid_num += 1
                sum += item
        if valid_num > 0:
            average = round(sum / valid_num, 3)
            for i in range(row):
                item = apache[i][j]
                if np.isnan(item) or item <= 10:
                    apache[i][j] = average
                else:
                    apache[i][j] = 0
        else:
            for i in range(row):
                item = apache[i][j]
                if np.isnan(item) or item <= 10:
                    apache[i][j] = 0
    for j in range(44, 64):
        valid_num = 0
        sum = 0
        for i in range(row):
            item = apache[i][j]
            if item > 0:
                # 吸入氧气浓度转换为数值型
                if j == 63:
                    if item > 1:
                        apache[i][j] = item / 100
                        item = apache[i][j]
                valid_num += 1
                sum += item
        if valid_num > 0:
            average = round(sum / valid_num, 3)
            for i in range(row):
                item = apache[i][j]
                if item < 0:
                    apache[i][j] = average
        else:
            for i in range(row):
                item = apache[i][j]
                if item < 0:
                    apache[i][j] = 0
    DataFrame(apache).to_csv('../0907/apache_average.csv', mode='w', encoding='utf-8',
                             header=filter.param.apache_header,
                             index=False)


# 提取apache分析数据
def apache_scale_data_for_univariate_analysis(data):
    '''
    :param apache:
    :return:
    '''
    new_data = data.iloc[:, 1:32]
    print(new_data.columns)
    for i in range(32, 44):
        # 当前列数据
        temp = np.array(data.iloc[:, i])
        # 计算20分位和80分位数据
        per_2 = np.percentile(temp, 20)
        per_8 = np.percentile(temp, 80)
        dif = per_8 - per_2
        if dif:
            temp = np.round(temp / dif, 4)
        new_data = pd.concat([new_data, DataFrame(temp)], axis=1)
    new_data = pd.concat([new_data, data.iloc[:, 44]], axis=1)
    header = filter.param.apache_header
    header.remove('id')
    header.append('status')
    print(new_data.columns)
    DataFrame(new_data).to_csv('../0907/apache_standard.csv', mode='w', encoding='utf-8', index=False,
                               header=header)


# 删除重复行
def drop_duplicates_and_add_head(apache, data):
    # 删除重复行
    apache = apache.drop_duplicates()
    # 添加对应患者的状态数据
    ids = apache.iloc[:, 0]
    status = []
    for id in ids:
        data_index = np.array(np.argwhere(data == id))[0][0]
        status.append(data[data_index][-5])
    apache = pd.concat([apache, DataFrame(status)], axis=1)
    header = filter.param.apache_header
    header.append('status')
    DataFrame(apache).to_csv('../0907/apache_1.csv', mode='w', encoding='utf-8', header=header, index=False)


# 使用apache数据，查找出各结果的最相关特征，apache单变量分析
def apache_univariate_analysis(apache):
    header = filter.param.apache_header
    header.remove('id')
    labels = np.array(apache.iloc[:, -1])
    data = np.array(apache.iloc[:, :-1])
    outcomes = {'Spontaneous recovery': 0, 'Long stay': 1, 'Rapid death': 2}
    for name, label in outcomes.items():
        label_new = []
        for item in labels:
            if item == label:
                label_new.append(1)
            else:
                label_new.append(0)
        label_new = np.array(label_new)
        # 单变量分析
        x_train, x_test, y_train, y_test = train_test_split(data, label_new, shuffle=True, random_state=42)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        # 计算优势比
        coefs = np.exp(model.coef_)
        # 保留优势比参数
        coef_lr = pd.DataFrame({'var': header, 'coef': coefs.flatten()})
        coef_lr.to_csv('../0907/apache_coef_' + str(name) + '.csv', mode='w', encoding='utf-8', index=False,
                       header=['var', 'coef'])
        index_sort = np.abs(coef_lr['coef'] - 1).sort_values().index
        # 提取前十五个特征
        coef_lr = coef_lr.iloc[index_sort, :][-15:]
        vars = coef_lr['var']
        coefs = coef_lr['coef']
        max = np.max(coefs)
        min = np.min(coefs)
        # 绘制单变量分析图
        plot_univariate_analysis(vars, coefs, 'apache', name, max, min)
        XGB = XGBClassifier(max_depth=7, min_child_weight=5)
        XGB_param = {
            # 'max_depth': [7, 9],
            # 'min_child_weight': [1, 3, 5],
            # 'subsample': [i / 10.0 for i in range(6, 10)],
            # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
            # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
            # # 'gamma': [i / 10.0 for i in range(0, 5)],
            'eta': [0.00001, 0.0001, 0.0002, 0.0005, 0.001],
            'n_estimators': [25, 50, 100, 200, 500, 800, 1000],
            'learning_rate': [0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2],
        }
        # 绘制全部数据的曲线图
        clf = GridSearchCV(XGB, XGB_param, cv=5, n_jobs=5, scoring='roc_auc')
        clf.fit(x_train, y_train)
        best_estimator = clf.best_estimator_
        y_pred = best_estimator.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
        test_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=r'all(area=%f)' % test_auc, color='r')
        print('outcome : %s, select features : %s, all best params : %s, auc: %s' % (name, vars, clf.best_params_,
                                                                                     test_auc))
        # 选取最重要的特征对应的数据
        vars = [item for item in vars]
        new_data = apache[vars[0]]
        for i in range(1, len(vars)):
            new_data = pd.concat([new_data, apache[vars[i]]], axis=1)
        new_data = np.array(new_data)
        x_train, x_test, y_train, y_test = train_test_split(new_data, label_new, test_size=0.2, shuffle=True,
                                                            random_state=42)
        XGB = XGBClassifier()
        clf = GridSearchCV(XGB, XGB_param, cv=5, n_jobs=5, scoring='roc_auc', )
        clf.fit(x_train, y_train)
        best_estimator = clf.best_estimator_
        y_pred = best_estimator.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
        test_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=r'top 15(area=%f)' % test_auc, color='b')
        print('outcome : %s, select features : %s, top 15 best params : %s, auc: %s' % (
            name, vars, clf.best_params_, test_auc))
        plt.title(r'%s' % name, fontweight='bold')
        plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
        plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
        plt.legend(loc='lower right', fontsize=10)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('../0907/apache_roc_xgb_' + str(name) + '.png')
        plt.show()


if __name__ == '__main__':
    # 所有ARDS患者的住院记录id
    # result = np.array(pd.read_csv('../0907/final_unitstayid.csv').iloc[:, 0])
    # for id in result:
    #     # 住院记录对应的220维数据信息抽取
    #     apache_header = filter.param.apache_header
    #     header = init.init_dict(header=apache_header)
    #     # 患者住院记录基本信息
    #     # id = item[0]
    #     header['id'] = id
    #     # 获取患者apache数据
    #     query = query_sql.Query()
    #     res = query.extra_apacheIV_data_by_id(id, header)
    #     if res:
    #         DataFrame([header]).to_csv('../0907/apache.csv', mode='a', encoding='utf-8', header=False, index=False)
    #         # # # 数据追加写入文件
    #         print(str(id) + '全部数据写入成功！')
    #     else:
    #         print('id : %s not be found!' % id)
    # 去重+添加状态
    # data = np.array(pd.read_csv('../0907/data.csv'))
    # apache = pd.read_csv('../0907/apache.csv')
    # drop_duplicates_and_add_head(apache, data)
    # 异常值处理
    # data = pd.read_csv('../0907/apache.csv', low_memory=False)
    # fill_0 = fill_with_0(data)
    # 数据归一化
    # apache_deal = pd.read_csv('../0907/apache_deal.csv')
    # apache_scale_data_for_univariate_analysis(apache_deal)
    standard = pd.read_csv('../0907/apache_standard.csv')
    apache_univariate_analysis(standard)
