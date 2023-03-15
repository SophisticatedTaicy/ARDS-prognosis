import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from pandas import DataFrame
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from pylab import mpl

from clinical_exam.param import english_columns, columns_dict
from filter.common import standard_data_by_white
from ml.classification.classify_parameter import XGBoost_none


def plot_decision_curve_analysis_on_test_set(model, data, label, name, flag, x_test=None, y_test=None):
    if x_test is None or y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(label), test_size=0.2,
                                                            shuffle=True,
                                                            random_state=42)
        x_train, x_test = standard_data_by_white(x_train, x_test)
    else:
        x_train = data
        y_train = label
    model.fit(x_train, y_train)
    if name == 'Perceptron':
        test_predict_proba = model._predict_proba_lr(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        net_benefit_model = calculate_net_benefit_model(threshold, test_predict_proba[:, 1], y_test)
    elif name == 'LinearRegression' or name == 'BayesianRidge':
        test_predict_proba = model.predict(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba, pos_label=1)
        net_benefit_model = calculate_net_benefit_model(threshold, test_predict_proba, y_test)
    else:
        test_predict_proba = model.predict_proba(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        net_benefit_model = calculate_net_benefit_model(threshold, test_predict_proba[:, 1], y_test)
    net_benefit_all = calculate_net_benefit_all(threshold, y_test)
    fig, ax = plt.subplots()
    plot_DCA(ax, threshold, net_benefit_model, net_benefit_all, flag)
    plt.show()


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, flag):
    # Plot
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    ax.plot(thresh_group, net_benefit_model, color='crimson', label='Model')
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
    # Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)
    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation
    if flag:
        ax.set_xlabel(xlabel='阈概率', fontdict={'family': 'SimSun', 'fontsize': 15})
        ax.set_ylabel(ylabel='净获益', fontdict={'family': 'SimSun', 'fontsize': 15})
    else:
        ax.set_xlabel(xlabel='Threshold Probability', fontdict={'family': 'Times New Roman', 'fontsize': 15})
        ax.set_ylabel(ylabel='Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    if flag:
        plt.savefig('DCA_chi.svg', bbox_inches='tight', format='svg')
    else:
        plt.savefig('DCA.svg', bbox_inches='tight', format='svg')
    return ax


def XGBoost_selective(data, label, flag):
    """
    @param columns: feature names
    @param data: feature data
    @param label: labels
    @param flag: 0/1 --> english/chinese
    """
    # 数据归一化
    model = XGBClassifier()
    model.fit(np.array(data), label)
    # xgboost重要性派选，筛选出前十五个
    importances = model.feature_importances_
    if flag:
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['SimSun']
        coef_xgboosr = DataFrame({'var': data.columns, 'coef': importances})
    else:
        mpl.rcParams['font.family'] = 'Times New Roman'
        coef_xgboosr = DataFrame({'var': english_columns, 'coef': importances})
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
        plt.text(coef + 0.003, i, '%.2f' % coef)
    if flag:
        plt.title('特征重要性', fontweight='bold')
        plt.xlabel('F分数', fontweight='bold')
    else:
        plt.title('feature importance', fontweight='bold', fontproperties='Times New Roman')
        plt.xlabel('F score', fontweight='bold', fontproperties='Times New Roman')
    plt.tight_layout()
    if flag:
        plt.savefig('XGBoost_chi.svg', bbox_inches='tight', format='svg')
    else:
        plt.savefig('XGBoost.svg', bbox_inches='tight', format='svg')
    plt.show()
    # prepear top 15 data
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), label, test_size=0.2, shuffle=True,
                                                        random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    model = XGBoost_none
    model.fit(x_train, y_train)
    y_predict_proba = model.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_predict_proba[:, 1])
    mean_auc = auc(fpr, tpr)
    if flag:
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['SimSun']
        plt.plot(fpr, tpr, label='所有 (area=%.3f)' % mean_auc, color='red')
    else:
        plt.plot(fpr, tpr, label='All (area=%.3f)' % mean_auc, color='red')
    if flag == 0:
        new_vars = []
        for item in list(vars):
            for k, v in columns_dict.items():
                if v == item:
                    new_vars.append(k)
                    print('key : %s,  value : %s ' % (k, v))
        vars = new_vars
    new_data = data[vars]
    x_train, x_test, y_train, y_test = train_test_split(np.array(new_data), label, test_size=0.2, shuffle=True,
                                                        random_state=42)
    x_train, x_test = standard_data_by_white(x_train, x_test)
    model.fit(x_train, y_train)
    y_predict_proba = model.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_predict_proba[:, 1])
    mean_auc = auc(fpr, tpr)
    if flag:
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['SimSun']
        plt.plot(fpr, tpr, label='前15 (area=%.3f)' % mean_auc, color='blue')
    else:
        plt.plot(fpr, tpr, label='Top 15 (area=%.3f)' % mean_auc, color='blue')
    if flag:
        plt.xlabel('1-特异性', fontsize=9, fontweight='bold')
        plt.ylabel('敏感度', fontsize=9, fontweight='bold')
    else:
        plt.xlabel('1-specificity', fontsize=9, fontweight='bold', fontproperties='Times New Roman')
        plt.ylabel('sensitivity', fontsize=9, fontweight='bold', fontproperties='Times New Roman')
    plt.yticks(np.arange(0, 1.05, 0.2), fontsize=7, fontproperties='Times New Roman')
    plt.xticks(np.arange(0, 1.05, 0.2), fontsize=7, fontproperties='Times New Roman')
    plt.grid()
    plt.legend(loc=4)
    if flag:
        plt.savefig('compare_chi.svg', bbox_inches='tight', format='svg')
    else:
        plt.savefig('compare.svg', bbox_inches='tight', format='svg')
    plt.show()
    plot_decision_curve_analysis_on_test_set(model, x_train, y_train, 'XGBoost', flag, x_test, y_test)


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
    data = total[common_columns]
    XGBoost_selective(data, labels, 0)
    # catboost_selective(data, labels, coulmns)
