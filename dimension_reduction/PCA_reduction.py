import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier
from filter.common import standard_data_by_white
from pylab import mpl

from filter.param import outcome_dict

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
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun']
    warnings.simplefilter('ignore')
    # pca不需要归一化
    # 划分训练集和测试集
    model = XGBClassifier(max_depth=9, min_child_weight=1)
    param = {
        # 'max_depth': [7, 9],
        # 'min_child_weight': [1, 3, 5],
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        # 'gamma': [i / 10.0 for i in range(0, 5)],
        # 'eta': [0.0001, 0.001, 0.005, 0.01, ],
        # 'n_estimators': [25, 50, 100, 200, 500],
        # 'learning_rate': [0.001, 0.01, 0.02, 0.05, 0.1],
    }
    colors = ['y', 'g', 'b']
    marks = ['p', 's', 'o']
    for key, value, color, mark in zip(outcome_dict.keys(), outcome_dict.values(), colors, marks):
        new_data = np.array(data)
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
            if n_component < dimensions[-1]:
                pca = PCA(n_components=n_component)
                data_new = pca.fit_transform(new_data)
            else:
                data_new = new_data
            x_train, x_test, y_train, y_test = train_test_split(data_new, label_new, test_size=0.2, shuffle=True,
                                                                random_state=1000)
            if n_component == dimensions[-1]:
                x_train, x_test = standard_data_by_white(x_train, x_test)
            research = GridSearchCV(model, param, scoring='roc_auc', n_jobs=10, refit=True, cv=5)
            # 划分训练集和测试集
            research.fit(x_train, y_train)
            # 最优模型
            best_estimator = research.best_estimator_
            # 测试数据上性能表现
            y_test_pred = best_estimator.predict_proba(x_test)
            fpr, tpr, threshold = roc_curve(y_test, y_test_pred[:, 1])
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            print('XGBoost dimensions are  : ' + str(n_component) + ' best param :' + str(
                research.best_params_) + ' auc : ' + str(auc))
        plt.plot(dimensions, aucs, marker=mark, lw=1, color=color, label=key)
    plt.title('PCA特征降维时AUC变化', fontsize=12, fontweight='bold')
    plt.yticks(np.arange(0.6, 1.05, 0.1), fontsize=7, fontproperties='Times New Roman')
    plt.xticks(np.arange(0, dimensions[-1] + 1, 10), fontsize=8, fontproperties='Times New Roman')
    plt.xlabel('特征维度', fontsize=12, fontweight='bold')
    plt.ylabel('AUC', fontsize=12, fontweight='bold', fontproperties='Times New Roman')
    labelss = plt.legend(loc='lower right', fontsize=10).get_texts()
    [label.set_fontname('Times New Roman') for label in labelss]
    plt.grid()
    plt.savefig('pca.svg', bbox_inches='tight', format='svg')
    plt.show()


if __name__ == '__main__':
    # 通过查看ARDset数据集PCA降维过程中，性能变化，说明特征降维对于减轻特征采集难度，加快预后效率有着非凡的意义
    total = pd.read_csv('../ARDS/combine/csvfiles/merge_data.csv', encoding='utf-8')
    labels = np.array(total['outcome'])
    del total['outcome']
    data = total.iloc[:, :]
    dimensions = np.arange(4, 105, 5)
    # second_dimensions = np.arange(5, 20, 5)
    pca_plot(data, labels, dimensions)
    # pca_plot(data, labels, second_dimensions)
