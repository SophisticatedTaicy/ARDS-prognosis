# GBDT_param = {'n_estimators': [140, 150, ], 'learning_rate': [0.003, 0.005]}
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBClassifier

param = {
    'n_estimators': [1000, 1100, 2000],
    'gamma': [0.5, 1, ]
}


def pca_plot(data, label, dimensions):
    '''
    :param data: 特征数据
    :param label: 标签数据
    :param dimensions: PCA维度
    :return:
    '''
    warnings.simplefilter('ignore')
    # pca不需要归一化
    # 划分训练集和测试集
    scaler = MinMaxScaler()
    plt.figure(dpi=500)
    model = XGBClassifier(max_depth=9, min_child_weight=1)
    param = {
        # 'max_depth': [7, 9],
        # 'min_child_weight': [1, 3, 5],
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        # 'gamma': [i / 10.0 for i in range(0, 5)],
        # 'eta': [0.0001, 0.001, 0.005, 0.01, ],
        'n_estimators': [25, 50, 100, 200, 500],
        'learning_rate': [0.001, 0.01, 0.02, 0.05, 0.1],
    }
    results = {'Spontaneous recovery': 0, 'Long stay': 1, 'Rapid death': 2}
    colors = ['y', 'g', 'b']
    marks = ['p', 's', 'o']
    for key, value, color, mark in zip(results.keys(), results.values(), colors, marks):
        label_new = []
        aucs = []
        # 按照不同预后划分标签
        for item in label:
            if item == value:
                label_new.append(1)
            else:
                label_new.append(0)
        label_new = np.array(label_new)
        for n_component in tqdm(dimensions):
            if n_component < 206:
                pca = PCA(n_components=n_component)
                data_new = pca.fit_transform(data)
            else:
                data_new = np.array(data)
            x_train, x_test, y_train, y_test = train_test_split(data_new, label_new, test_size=0.2, shuffle=True,
                                                                random_state=1000)
            if n_component == 206:
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.fit_transform(x_test)
            research = GridSearchCV(model, param, scoring='roc_auc', n_jobs=10, refit=True, cv=5)
            # 划分训练集和测试集
            research.fit(x_train, y_train)
            # 最优模型
            model = research.best_estimator_
            # 测试数据上性能表现
            y_test_pred = model.predict_proba(x_test)
            fpr, tpr, threshold = roc_curve(y_test, y_test_pred[:, 1])
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            print('XGBoost dimensions are  : ' + str(n_component) + ' best param :' + str(
                research.best_params_) + ' auc : ' + str(auc))
        plt.plot(dimensions, aucs, marker=mark, lw=2, color=color, label=key)
    plt.title('AUC Performance on PCA', fontsize=15, fontweight='bold')
    plt.xlim([dimensions[-1] - 1, dimensions[0] + 1])
    plt.ylim([0.6, 1])
    plt.xticks(np.arange(0, 201, 10))
    plt.xlabel('dimensions', fontweight='bold', fontsize=15)
    plt.ylabel('Auc', fontweight='bold', fontsize=15)
    plt.legend(loc='lower right', fontsize=7)
    # 绘制网格
    plt.grid()
    plt.savefig('../0907/pca_2.png')
    plt.show()


if __name__ == '__main__':
    dataframe = pd.read_csv('../ARDS/eicu/pictures/0907/fill_with_0.csv', sep=',', encoding='utf-8')
    data = np.array(dataframe.iloc[:, 1:-5])
    label = np.array(dataframe.iloc[:, -5])
    XGBoost_size = [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
    pca_plot(data, label, XGBoost_size)
