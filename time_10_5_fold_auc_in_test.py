from matplotlib import pyplot as plt, ticker
from matplotlib.ticker import MultipleLocator
from numpy import interp
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from ml.classification import classify_parameter


def model_test_in_five_fold(model, data, label, model_name, label_name):
    KF = KFold(n_splits=5)
    mean_tpr = []
    mean_fpr = np.linspace(0, 1, 100)
    t_mean_tpr = []
    t_mean_fpr = np.linspace(0, 1, 100)
    t_tprs = []
    tprs = []
    scale = MinMaxScaler()
    for train_index, test_index in KF.split(ards):
        # 每种模型训练时进行五折交叉检验
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        # 数据归一化
        x_train = scale.fit_transform(x_train)
        x_test = scale.fit_transform(x_test)
        # 模型训练
        model.fit(x_train, y_train)
        # 训练集测试
        t_predict_proba = model.predict_proba(x_train)
        t_fpr, t_tpr, t_threshold = metrics.roc_curve(y_train, t_predict_proba[:, 1], pos_label=1)
        t_tprs.append(interp(t_mean_fpr, t_fpr, t_tpr))
        t_tprs[-1][0] = 0
        t_mean_tpr = np.mean(t_tprs, axis=0)
        t_mean_tpr[-1] = 1
        # 测试集模型预测
        test_predict_proba = model.predict_proba(x_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    t_mean_auc = metrics.auc(t_mean_fpr, t_mean_tpr)
    plt.plot(t_mean_fpr, t_mean_tpr, label=r'Training (area=%0.3f)' % t_mean_auc, lw=1, color='r')
    plt.plot(mean_fpr, mean_tpr, label=r'Internal Validation (area=%0.3f)' % mean_auc, lw=1, color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='gray', label='Luck', alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    title = label_name + ' ROC'
    plt.title(title, fontsize=10)
    plt.legend(loc='lower right', fontsize=7)
    plt.savefig('0907/' + str(label_name) + ' ' + str(model_name) + '.png')
    plt.show()


def model_train_validate_and_test(model, data, label, model_name, label_name, params):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)
    # 使用五折交叉验证
    research = GridSearchCV(model, params, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
    research.fit(x_train, y_train)
    print(str(model_name) + ' ' + str(label_name) + ' best params ' + str(research.best_params_) + ' best_estimator_' +
          str(research.best_estimator_))
    best_estimator = research.best_estimator_
    train_pred = best_estimator.predict_proba(x_train)
    test_pred = best_estimator.predict_proba(x_test)
    train_fpr, train_tpr, train_threshold = metrics.roc_curve(y_train, train_pred[:, 1])
    test_fpr, test_tpr, test_threshold = metrics.roc_curve(y_test, test_pred[:, 1])
    train_roc = metrics.auc(train_fpr, train_tpr)
    test_roc = metrics.auc(test_fpr, test_tpr)
    plt.plot(train_fpr, train_tpr, label=r'Training (area=%0.3f)' % train_roc, lw=1, color='r')
    plt.plot(test_fpr, test_tpr, label=r'Internal Validation (area=%0.3f)' % test_roc, lw=1, color='b')
    plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle='-', lw=3, color='gray', alpha=0.8)
    ax = plt.gca()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    # 设置坐标轴刻度显示
    ax.xaxis.set_major_locator(ticker.FixedLocator((0.00, 0.25, 0.50, 0.75, 1.00)))
    # 把x轴的主刻度设置为0.25的倍数
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    plt.xticks(np.arange(0, 1.05, 0.125))
    plt.yticks(np.arange(0, 1.05, 0.125))
    plt.xlabel('1 - specificity', fontsize=10)
    plt.ylabel('sensitivity', fontsize=10)
    title = model_name + ' ROC'
    plt.title(title, fontsize=10)
    plt.legend(loc='lower right', fontsize=7)
    plt.grid()
    plt.savefig('0907/' + str(label_name) + ' ' + str(model_name) + '.png')
    plt.show()


# https://blog.csdn.net/DoReAGON/article/details/89290691
if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    # 解决RuntimeWarning: invalid value encountered in true_divide
    np.seterr(divide='ignore', invalid='ignore')
    dataframe = shuffle(pd.read_csv('ARDS/eicu/pictures/0907/fill_with_average.csv', encoding='utf-8'))
    ards = np.array(dataframe.iloc[:, 1:-5])
    label = np.array(dataframe.iloc[:, -5])
    outcome = {'Spontaneous recovery': 0, 'Long stay': 1, 'Rapid death': 2}
    models = {'XGBoost': classify_parameter.XGB, 'GBDT': classify_parameter.GBDT}
    # 数据类型转换
    params = [classify_parameter.XGB_param, classify_parameter.GBDT_param]
    for outcome_name, outcome_label in outcome.items():
        label_new = []
        for item in label:
            if item == outcome_label:
                label_new.append(1)
            else:
                label_new.append(0)
        label_new = np.array(label_new)
        for name, model, param in zip(models.keys(), models.values(), params):
            print('model : %s outcome :%s ' % (name, outcome_name))
            model_train_validate_and_test(model, ards, label_new, name, outcome_name, param)
