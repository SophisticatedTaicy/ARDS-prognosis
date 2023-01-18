import os
import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from matplotlib import pyplot as plt
from numpy import interp
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter('ignore')
import filter.param
from ml.classification import classify_parameter
from ml.classification.classify_parameter import XGB

os.environ['PATH'] += os.pathsep + 'D:/graphviz/bin/'
from tqdm import tqdm

param = {
    'n_estimators': [100, 150, 200, 250, 300],
    # 'gamma': [0, 0.05, 0.1, 0.5, 1, 10, 100]
}
GBDT_param = {'n_estimators': [160, 165, 168, 200], 'learning_rate': [0.006, 0.0063, 0.0065, 0.0068, 0.007, 0.0075]}


def AUC(y_true, y_pred):
    pred_score = 1.0 / (1.0 + np.exp(-y_pred[:, 1]))
    pred = [1 if p > 0.5 else 0 for p in pred_score]
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    return [('accuracy', acc), ('auc', auc), ('f1', f1)]

def pca_plot(data, label_new, model, name, param, pca_size):
    warnings.simplefilter('ignore')
    colors = filter.param.base_colors[:len(pca_size)]
    marks = filter.param.marks[:len(pca_size)]
    # pca不需要归一化
    # 划分训练集和测试集
    scaler = MinMaxScaler()
    for n_component, color, mark in tqdm(zip(pca_size, colors, marks)):
        if n_component != 206:
            pca = PCA(n_components=n_component)
            data = pca.fit_transform(data)
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(data, label_new, test_size=0.2, shuffle=True)
        # 网格搜索五折验证
        if n_component == 206:
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
        research = GridSearchCV(model, param, scoring='roc_auc', n_jobs=10, refit=True, cv=5, verbose=0)
        # 划分训练集和测试集
        research.fit(x_train, y_train)
        # 最优模型
        model = research.best_estimator_
        # 测试数据上性能表现
        y_test_pred = model.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_test_pred[:, 1])
        auc = metrics.auc(fpr, tpr)
        print('model : ' + str(name) + ' n components : ' + str(n_component) + ' best param :' + str(
            research.best_params_) + ' auc : ' + str(auc))
        # 测试集上roc曲线绘制
        plt.plot(fpr[:-1:35], tpr[:-1:35], label='testing n = %d (area = %.4f)' % (n_component, auc),
                 lw=1, color=color, marker=mark, markersize=1)
    plt.title('PCA ROCs of %s ' % 'long stay', fontsize=20)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Luck', alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    plt.legend(loc='lower right', fontsize=7)
    # 绘制网格
    plt.grid()
    plt.savefig('0907/' + str(name) + str('_pca_3') + 'long stay' + '.png')
    plt.show()


def lda_plot(data, label, model, name, param):
    # colors = filter.param.base_colors
    lda = LinearDiscriminantAnalysis(n_components=2)
    data = lda.fit_transform(data, label)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)
    y_pred = lda.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr[:-1:35], tpr[:-1:35], label='n = %d (area = %.4f)' % (2, auc),
             lw=1, color='r', marker='o', markersize=3)
    print(str(data) + ' predict : ' + str(y_pred[:, 1]) + ' true : ' + str(y_test))


def plot(data, label, model, name, param):
    KF = KFold(n_splits=5, shuffle=True, random_state=42)
    scale = MinMaxScaler()
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    t_tprs = []
    t_mean_fpr = np.linspace(0, 1, 100)
    for train_index, test_index in KF.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        x_train = scale.fit_transform(x_train)
        x_test = scale.fit_transform(x_test)
        clf = RandomizedSearchCV(model, param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
        clf.fit(x_train, y_train)
        best_estimator = clf.best_estimator_
        y_train_pred = best_estimator.predict_proba(x_train)
        y_pred = best_estimator.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
        t_fpr, t_tpr, t_threshold = roc_curve(y_train, y_train_pred[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0
        t_tprs.append(interp(t_mean_fpr, t_fpr, t_tpr))
        t_tprs[-1][0] = 0
    mean_tpr = np.mean(tprs, axis=0)
    t_mean_tpr = np.mean(t_tprs, axis=0)
    mean_tpr[-1] = 1
    t_mean_tpr[-1] = 1
    t_mean_auc = metrics.auc(t_mean_fpr, t_mean_tpr)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr[:-1:5], mean_tpr[:-1:5], label='test (area = %.4f)' % mean_auc,
             lw=1, color='b', marker='o', markersize=3)
    plt.plot(mean_fpr[:-1:5], mean_tpr[:-1:5], label='training (area = %.4f)' % t_mean_auc,
             lw=1, color='r', marker='o', markersize=3)
    plt.title('LDA ROCs of %s ' % 'long stay', fontsize=20)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Luck', alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend(loc='lower right', fontsize=7)
    plt.grid()
    plt.savefig('pictures/' + str(name) + ' long stay' + '.png')
    plt.show()


def shapBoruta(data, label, model):
    importances = np.zeros((data.reshape[1]))
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, shuffle=True,
                                                              random_state=42)
        research = GridSearchCV(model, param, cv=5)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        model = model.best_estimator_
        y_predict = model.predict_proba(y_test)
        fpr, tpr, threshold = roc_curve(y_test, y_predict[:, 1])
        auc = metrics.auc(fpr, tpr)
        plt.plot()


def boruta(data, label, model):
    fea_selector = BorutaPy(model, n_estimators=106, verbose=3, random_state=42)
    fea_selector.fit(data, label)
    labels = filter.param.result_header[1:-5]
    sel_feas = {}
    print('查看选择的特征： ' + str(fea_selector.support_))
    for label, selected, rank in zip(labels, fea_selector.support_, fea_selector.ranking_):
        if selected:
            sel_feas[label] = rank
            print(str(label) + ' is selected and rank is ' + str(rank))


if __name__ == '__main__':
    dataframe = pd.read_csv('../ARDS/eicu/pictures/0907/fill_with_0.csv', sep=',', encoding='utf-8')
    data = np.array(dataframe.iloc[:, 1:-5])
    label = np.array(dataframe.iloc[:, -5])
    results = {'Spontaneous recovery': 0, 'Long stay': 1, 'Rapid death': 2}
    model_dict = {'GBDT': classify_parameter.GBDT_none, 'XGBoost': classify_parameter.XGBoost_none}
    params = [classify_parameter.GBDT_param, classify_parameter.XGB_param]
    colors = filter.param.base_colors[:7]
    label_new = []
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    boruta(data, label_new, XGB)
