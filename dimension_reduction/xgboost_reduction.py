import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt, pyplot
from pandas import DataFrame
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve
from xgboost import plot_importance, XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from pylab import mpl
from filter.common import standard_data_by_white


def xgboost_selective(data, label, columns):
    '''
    :param data: 数据
    :param label: 标签
    :param name: 标签名
    :return: 特征重要性
    '''
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), label, test_size=0.25, shuffle=True,
                                                        random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
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
    importances = xgb.get_score(importance_type='gain')
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
    # y_pred = bst.predict(dtest)
    # accuracy = accuracy_score(y_test, y_pred)
    # print('accuracy : ' + str(accuracy))
    # plt.figure(dpi=500)
    # plot_importance(bst, title='feature importance', max_num_features=15, importance_type='gain', grid=False,
    #                 height=0.5)
    plt.grid()
    plt.savefig('../clinical_exam/xgboost.svg', bbox_inches='tight', format='svg')
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
    max = np.max(coefs)
    min = np.min(coefs)
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
    plt.title('feature importance', fontweight='bold', fontproperties='Times New Roman')
    plt.xlim([min - 0.01, max + 0.01])
    plt.xlabel('F score', fontweight='bold', fontproperties='Times New Roman')
    plt.tight_layout()
    plt.savefig('../clinical_exam/XGBoost.svg', bbox_inches='tight', format='svg')
    pyplot.show()


def catboost_selective(data, label, columns):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    from catboost import CatBoostClassifier
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=1232, shuffle=True)
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
    fill_aver = pd.read_csv('../ARDS/eicu/pictures/0907/fill_with_0.csv', sep=',', encoding='utf-8')
    static = fill_aver.iloc[:, 1:42]
    dynamic = fill_aver.iloc[:, 42:205:3]
    data = pd.concat([static, dynamic], axis=1)
    labels = fill_aver.iloc[:, -5]
    results = {'Spontaneous recovery': 0, 'Long stay': 1, 'Rapid death': 2}
    for name, label in results.items():
        label_new = []
        for item in labels:
            if item == label:
                label_new.append(1)
            else:
                label_new.append(0)
        label_new = np.array(label_new)
        xgboost_selective(data, label_new, name)
