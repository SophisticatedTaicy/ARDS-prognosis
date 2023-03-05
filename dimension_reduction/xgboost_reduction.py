import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from pandas import DataFrame
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pylab import mpl

from filter.common import standard_data_by_white, read_file, judge_label_balance, format_label, concat_array
from filter.param import outcome_dict


def xgboost_selective(data, label, columns, x_test=None, y_test=None):
    '''
    :param data: 数据
    :param label: 标签
    :param name: 标签名
    :return: 特征重要性
    '''
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    if x_test == None and y_test == None:
        x_train, x_test, y_train, y_test = train_test_split(np.array(data), label, test_size=0.3, random_state=1232,
                                                            shuffle=True)
    else:
        x_train = data
        y_train = label
    x_train, x_test = standard_data_by_white(x_train, x_test)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.001,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'lambda': 0.0001,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.5,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'eta': 0.0001,  # 如同学习率
        'seed': 1000,
        'nthread': 4,  # cpu 线程数
        'eval_metric': 'logloss'
    }
    # 使显示图标自适应
    plt.rcParams['figure.autolayout'] = True
    bst = xgb.train(params, dtrain, 100)
    importances = xgb.get_score()
    coef_xgboosr = DataFrame({'var': columns, 'coef': importances})
    index_sort = np.abs(coef_xgboosr['coef']).sort_values().index
    coef_lr = coef_xgboosr.iloc[index_sort, :]
    vars = coef_lr['var']
    coefs = coef_lr['coef']
    mean = np.mean(coefs)
    # 分别绘制单变量分析结果图
    plt.grid(zorder=0)
    error_kw = {'ecolor': '0.1', 'capsize': 5}
    # 绘制柱状图
    for var, coef in zip(vars, coefs):
        plt.barh(var, coef, error_kw=error_kw, xerr=mean * 0.01, facecolor='b', zorder=5)
    # 文本
    for i, coef in enumerate(coefs):
        plt.text(coef + 0.003, i, '%.3f' % coef, fontproperties='Times New Roman')
    plt.grid()
    # if x_test:
    #     plt.savefig('xgboost.svg', bbox_inches='tight', format='svg')
    # else:
    #     plt.savefig('../clinical_exam/xgboost.svg', bbox_inches='tight', format='svg')
    plt.show()
    return bst


def XGBoost_selective(data, label, columns):
    # 数据归一化
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    model = XGBClassifier()
    model.fit(data, label)
    # xgboost重要性派选，筛选出前十五个
    importances = model.feature_importances_
    coef_xgboosr = DataFrame({'var': columns, 'coef': importances})
    index_sort = np.abs(coef_xgboosr['coef']).sort_values().index
    coef_lr = coef_xgboosr.iloc[index_sort, :][-15:]
    vars = coef_lr['var']
    coefs = coef_lr['coef']
    mean = np.mean(coefs)
    # 分别绘制单变量分析结果图
    plt.figure(dpi=500)
    plt.grid(zorder=0)
    error_kw = {'ecolor': '0.1', 'capsize': 5}
    # 绘制柱状图
    for var, coef in zip(vars, coefs):
        plt.barh(var, coef, error_kw=error_kw, xerr=mean * 0.01, facecolor='b', zorder=5)
    # 文本
    for i, coef in enumerate(coefs):
        plt.text(coef + 0.003, i, '%.3f' % coef)
    # plt.title('feature importance', fontweight='bold', fontproperties='Times New Roman')
    # plt.xlabel('F score', fontweight='bold', fontproperties='Times New Roman')
    plt.title('特征重要性', fontweight='bold')
    plt.xlabel('F分数', fontweight='bold')
    plt.tight_layout()
    if data.shape[0] > 50:
        plt.savefig('XGBoost.svg', bbox_inches='tight', format='svg')
    else:
        plt.savefig('../clinical_exam/XGBoost.svg', bbox_inches='tight', format='svg')
    plt.show()


def catboost_selective(data, label, columns, x_test=None, y_test=None):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    from catboost import CatBoostClassifier
    if x_test == None and y_test == None:
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1232, shuffle=True)
    else:
        x_train = data
        y_train = label
    category_features = np.where(x_train.dtypes != np.float)
    # model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.5, cat_features=category_features,
    #                            loss_function='Logloss', logging_level='Verbose')
    model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.5, loss_function='Logloss',
                               logging_level='Verbose')
    model.fit(x_train, y_train, eval_set=(x_test, y_test), plot=True)
    importances = model.feature_importances_
    coef_xgboosr = DataFrame({'var': columns, 'coef': importances})
    index_sort = np.abs(coef_xgboosr['coef']).sort_values().index
    coef_lr = coef_xgboosr.iloc[index_sort, :]
    vars = coef_lr['var']
    coefs = coef_lr['coef']
    mean = np.mean(coefs)
    # 分别绘制单变量分析结果图
    # plt.figure(dpi=500)
    plt.grid(zorder=0)
    error_kw = {'ecolor': '0.1', 'capsize': 5}
    # 绘制柱状图
    for var, coef in zip(vars, coefs):
        print('var : %s coef : %s' % (var, coef))
        if coef > 0:
            plt.barh(var, coef, error_kw=error_kw, xerr=mean * 0.05, facecolor='b', zorder=5)
    # 文本
    for i, coef in enumerate(coefs):
        print('coef : %s i : %s' % (coef, i))
        if coef > 0:
            plt.text(coef + 0.1, i - 23, '%.1f' % coef, fontproperties='Times New Roman')
    plt.tight_layout()
    plt.savefig('../clinical_exam/catboost.svg', bbox_inches='tight', format='svg')
    plt.show()


if __name__ == '__main__':
    total = pd.read_csv('../ARDS/combine/csvfiles/merge_data.csv')
    coulmns = list(total.columns)
    coulmns.remove('outcome')
    labels = np.array(total['outcome'])
    common_columns = coulmns
    print(common_columns)
    data = total[common_columns]
    XGBoost_selective(data, labels, coulmns)
    catboost_selective(data, labels, coulmns)
