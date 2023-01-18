import threading

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

import filter.param
from ARDS import init
from ARDS.eicu.data_extraction import query_sql
from dimension_reduction.univariate_analyse import plot_univariate_analysis


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
            aps_header = filter.param.aps_header
            header = init.init_dict(header=aps_header)
            # 患者住院记录基本信息
            id = item[0]
            identification = item[1]
            severity = item[2]
            enrollment = item[3]
            header['id'] = id
            header['identification'] = identification
            header['severity'] = severity
            header['enrollment'] = enrollment
            # 查询相关住院记录静态数据
            query = query_sql.Query()
            query.filter_static(item, header)
            # 获取患者aps数据
            query = query_sql.Query()
            aps = query.filter_with_aps(id)
            if aps:
                for i in range(len(aps[0])):
                    header[aps_header[i + 42]] = aps[0][i]
            else:
                print('不存在患者%s的aps信息' % id)
                continue
            # 获取患者额外信息
            query = query_sql.Query()
            status, detail, unitstay, hospitalstay = query.access_outcome(id, enrollment)
            header['status'] = status
            header['detail'] = detail
            header['unit'] = unitstay
            header['hospital'] = hospitalstay
            # print('patient %s info : %s' % (id, str(header)))
            DataFrame([header]).to_csv('../0907/aps.csv', mode='a', encoding='utf-8', header=False, index=False)
            # # # 数据追加写入文件
            print(str(item) + '全部数据写入成功！')


# aps异常数据补充0
def fill_with_0(aps):
    '''
    年龄，入院apache以及bmi异常数据修改
    aps变量-1修改
    '''
    aps = np.array(aps)
    row = aps.shape[0]
    for j in range(39, 42):
        valid_num = 0
        sum = 0
        for i in range(row):
            item = aps[i][j]
            if item > 10:
                valid_num += 1
                sum += item
        if valid_num > 0:
            average = round(sum / valid_num, 3)
            for i in range(row):
                item = aps[i][j]
                if np.isnan(item) or item <= 10:
                    aps[i][j] = average
        else:
            for i in range(row):
                if np.isnan(item) or item <= 10:
                    aps[i][j] = 0
    for j in range(44, 64):
        for i in range(row):
            item = aps[i][j]
            if item > 0:
                # 吸入氧气浓度转换为数值型
                if j == 63:
                    if item > 1:
                        aps[i][j] = item / 100
            elif item < 0:
                aps[i][j] = 0
    DataFrame(aps).to_csv('../0907/aps_0.csv', mode='w', encoding='utf-8', header=filter.param.aps_header,
                          index=False)


# aps数据异常值补充均值
def fill_with_average(aps):
    '''
    年龄，入院apache以及bmi异常数据修改
    aps变量-1修改
    '''
    aps = np.array(aps)
    row = aps.shape[0]
    for j in range(39, 42):
        valid_num = 0
        sum = 0
        for i in range(row):
            item = aps[i][j]
            if item > 10:
                valid_num += 1
                sum += item
        if valid_num > 0:
            average = round(sum / valid_num, 3)
            for i in range(row):
                item = aps[i][j]
                if np.isnan(item) or item <= 10:
                    aps[i][j] = average
                else:
                    aps[i][j] = 0
        else:
            for i in range(row):
                item = aps[i][j]
                if np.isnan(item) or item <= 10:
                    aps[i][j] = 0
    for j in range(44, 64):
        valid_num = 0
        sum = 0
        for i in range(row):
            item = aps[i][j]
            if item > 0:
                # 吸入氧气浓度转换为数值型
                if j == 63:
                    if item > 1:
                        aps[i][j] = item / 100
                        item = aps[i][j]
                valid_num += 1
                sum += item
        if valid_num > 0:
            average = round(sum / valid_num, 3)
            for i in range(row):
                item = aps[i][j]
                if item < 0:
                    aps[i][j] = average
        else:
            for i in range(row):
                item = aps[i][j]
                if item < 0:
                    aps[i][j] = 0
    DataFrame(aps).to_csv('../0907/aps_average.csv', mode='w', encoding='utf-8', header=filter.param.aps_header,
                          index=False)


# 提取aps分析数据
def aps_scale_data_for_univariate_analysis(data, mark):
    '''
    :param aps:
    :return:
    '''
    columns = filter.param.aps_header[42:-6]
    new_data = data.iloc[:, 42:44]
    for i in range(44, 64):
        # 当前列数据
        temp = np.array(data.iloc[:, i])
        # 计算20分位和80分位数据
        per_2 = np.percentile(temp, 20)
        per_8 = np.percentile(temp, 80)
        dif = per_8 - per_2
        if dif:
            temp = np.round(temp / dif, 4)
        new_data = pd.concat([new_data, DataFrame(temp)], axis=1)
    new_data = pd.concat([new_data, data.iloc[:, 64]], axis=1)
    print('data : %s' % new_data.columns)
    DataFrame(new_data).to_csv('../0907/aps_univariate_' + str(mark) + '.csv', mode='w', encoding='utf-8', index=False,
                               header=columns)


# 删除重复行
def drop_duplicates_and_add_head(data):
    data = data.drop_duplicates()
    DataFrame(data).to_csv('../0907/apache.csv', mode='w', encoding='utf-8', header=filter.param.apache_header,
                           index=False)


# 使用aps数据，查找出各结果的最相关特征，aps单变量分析
def aps_univariate_analysis(aps):
    header = filter.param.aps_header[42:-7]
    labels = np.array(aps.iloc[:, -1])
    data = np.array(aps.iloc[:, :-1])
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
        coef_lr.to_csv('../0907/pictures/aps_coef_' + str(name) + '.csv', mode='w', encoding='utf-8', index=False,
                       header=['var', 'coef'])
        index_sort = np.abs(coef_lr['coef'] - 1).sort_values().index
        # 提取前十五个特征
        coef_lr = coef_lr.iloc[index_sort, :][-15:]
        vars = coef_lr['var']
        coefs = coef_lr['coef']
        max = np.max(coefs)
        min = np.min(coefs)
        # 绘制单变量分析图
        plot_univariate_analysis(vars, coefs, 'aps', name, max, min)
        XGB = XGBClassifier(max_depth=7, min_child_weight=5)
        XGB_param = {
            # 'max_depth': [7, 9],
            # 'min_child_weight': [1, 3, 5],
            # 'subsample': [i / 10.0 for i in range(6, 10)],
            # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
            # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
            'gamma': [i / 10.0 for i in range(0, 5)],
            # 'eta': [0.0001, 0.001, 0.005, 0.01, ],
            'n_estimators': [25, 50, 80, 100, 150, 200, ],
            'learning_rate': [0.0001, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.2],
        }
        GBDT = GradientBoostingClassifier(max_depth=7, max_features=7, min_samples_leaf=70, min_samples_split=1600)
        GBDT_param = {
            # 'max_depth': [7, 9],
            # 'min_samples_split': range(800, 1900, 200),
            # 'min_samples_leaf': range(60, 101, 10),
            # 'max_features': range(7, 20, 2),
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'n_estimators': [25, 50, 100, 200, 500],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.01, 0.015, 0.02]
        }
        clf = GridSearchCV(XGB, XGB_param, cv=5, n_jobs=5, scoring='roc_auc')
        clf.fit(x_train, y_train)
        best_estimator = clf.best_estimator_
        y_pred = best_estimator.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
        auc = metrics.auc(fpr, tpr)
        print('outcome : %s, select features : %s, all best params : %s, auc: %s' % (name, vars, clf.best_params_, auc))
        plt.plot(fpr, tpr, label=r'all(area=%f)' % auc, color='r')
        # 选取最重要的特征对应的数据
        vars = [item for item in vars]
        new_data = aps[vars[0]]
        for i in range(1, len(vars)):
            new_data = pd.concat([new_data, aps[vars[i]]], axis=1)
        new_data = np.array(new_data)
        x_train, x_test, y_train, y_test = train_test_split(new_data, label_new, test_size=0.2, shuffle=True,
                                                            random_state=42)
        XGB = XGBClassifier()
        clf = GridSearchCV(XGB, XGB_param, cv=5, n_jobs=5, scoring='roc_auc', )
        clf.fit(x_train, y_train)
        best_estimator = clf.best_estimator_
        y_pred = best_estimator.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=r'top 15(area=%f)' % auc, color='b')
        print('outcome : %s, select features : %s, top 15 best params : %s, auc: %s' % (
            name, vars, clf.best_params_, auc))
        plt.title(r'%s' % name, fontweight='bold')
        plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
        plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
        plt.legend(loc='lower right', fontsize=10)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('../0907/aps_roc_XGBoost_' + str(name) + '.png')
        plt.show()


if __name__ == '__main__':
    # 所有ARDS患者的住院记录id
    # result = np.array(pd.read_csv('../0907/final_unitstayid.csv'))
    # # 分线程运行
    # df_list = []
    # for i in range(13):
    #     df_list.append(result[i * 1000:i * 1000 + 1000])
    # df_list.append(result[10000:-1])
    # for data in tqdm(df_list):
    #     name = str(data[0][0]) + '-' + str(data[-1][0])
    #     # 分线程运行
    #     MyThread(name=name, data=data).start()
    # # 删除重复行
    # drop_duplicates_and_add_head(pd.read_csv('../0907/apache.csv'))
    # data = pd.read_csv('../0907/apache.csv', low_memory=False)
    # average = fill_with_average(data)
    # fill_0 = fill_with_0(data)
    # fill_0 = pd.read_csv('../0907/aps_0.csv')
    # aps_scale_data_for_univariate_analysis(fill_0, '0')
    # average = pd.read_csv('../0907/aps_average.csv')
    # aps_scale_data_for_univariate_analysis(average, 'average')
    # fill_0 = pd.read_csv('../0907/aps_0.csv')
    average = pd.read_csv('../0907/aps_univariate_0.csv')
    aps_univariate_analysis(average)
