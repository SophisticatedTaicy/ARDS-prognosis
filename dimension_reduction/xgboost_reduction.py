import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from pandas import DataFrame
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance, XGBClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import shap
import filter.param
from dimension_reduction.univariate_analyse import plot_univariate_analysis


def xgboost_selective(data, label, name):
    '''
    :param data: 数据
    :param label: 标签
    :param name: 标签名
    :return: 特征重要性
    '''
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, shuffle=True, random_state=42)
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
    y_pred = bst.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy : ' + str(accuracy))
    title = str(name) + ' feature importance'
    plt.figure(dpi=500)
    plot_importance(bst, title=title, max_num_features=15, importance_type='gain', grid=False, height=0.5, )
    plt.grid()
    plt.savefig('../0907/pictures/xgboost_' + str(name) + '_feature importance.png')
    plt.show()
    return bst


def XGBoost_selective(data, label, name):
    model = XGBClassifier()
    cols = data.columns
    model.fit(np.array(data), label)
    # xgboost重要性派选，筛选出前十五个
    header = filter.param.univariate_header[:-1]
    importances = model.feature_importances_
    coef_xgboosr = DataFrame({'var': header, 'coef': importances})
    index_sort = np.abs(coef_xgboosr['coef']).sort_values().index
    coef_lr = coef_xgboosr.iloc[index_sort, :][-15:]
    vars = coef_lr['var']
    coefs = coef_lr['coef']
    print('vars : %s coefs : %s ' % (vars, coefs))
    max = np.max(coefs)
    min = np.min(coefs)
    mean = np.mean(coefs)
    # 分别绘制单变量分析结果图
    plt.figure(dpi=500)
    plt.grid(zorder=0)
    error_kw = {'ecolor': '0.1', 'capsize': 5}
    for var, coef in zip(vars, coefs):
        plt.barh(var, coef, error_kw=error_kw, xerr=mean * 0.01, facecolor='b')
    plt.title('%s feature importance' % name, fontweight='bold')
    plt.xlim([min - 0.01, max + 0.01])
    plt.xlabel('F score', fontweight='bold')
    plt.tight_layout()
    plt.savefig('../0907/pictures/XGBoost_' + str(name) + '.png')
    pyplot.show()
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), label_new)
    # 绘制所有变量结果图
    XGB = XGBClassifier(max_depth=7, min_child_weight=5, colsample_bytree=0.8, subsample=0.9)
    XGB_param = {
        # 'max_depth': [7, 9],
        # 'min_child_weight': [1, 3, 5],
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        # 'gamma': [i / 10.0 for i in range(0, 5)],
        # 'eta': [0.0001, 0.001, 0.005, 0.01, ],
        # 'n_estimators': [25, 50, 100, 200, 500, 800, 1000, 1500],
        # 'learning_rate': [0.0001, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1],
    }
    clf = GridSearchCV(XGB, XGB_param, cv=5, n_jobs=5, scoring='roc_auc')
    clf.fit(x_train, y_train)
    best_estimator = clf.best_estimator_
    y_pred = best_estimator.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
    test_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=r'all(area=%f)' % test_auc, color='r')
    print('outcome : %s, select features : %s, all best params : %s, auc: %s' % (
        name, vars, clf.best_params_, test_auc))
    # 选取最重要的特征对应的数据
    vars = [item for item in vars]
    print(vars)
    new_data = data[vars[0]]
    for i in range(1, len(vars)):
        new_data = pd.concat([new_data, data[vars[i]]], axis=1)
    cols = new_data.columns
    # new_data = np.array(new_data)
    # 各特征重要性系数
    model.fit(np.array(new_data), label_new)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.array(new_data))
    # 按照shap绝对值排序，筛选出最大的前五个
    player_explainer = pd.DataFrame()
    player_explainer['feature'] = cols
    player_explainer['shap_value'] = shap_values[20]
    shap_index_sort = np.abs(player_explainer['shap_value']).sort_values().index

    x_train, x_test, y_train, y_test = train_test_split(np.array(new_data), label_new, test_size=0.2, shuffle=True,
                                                        random_state=42)
    XGB = XGBClassifier(max_depth=7, min_child_weight=5, subsample=0.9)

    clf = GridSearchCV(XGB, XGB_param, cv=5, n_jobs=5, scoring='roc_auc')
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
    shap.summary_plot(shap_values, new_data)


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
        # bst = xgboost_selective(data, label_new, name)
        XGBoost_selective(data, label_new, name)
