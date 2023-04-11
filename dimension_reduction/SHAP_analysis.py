from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# https://www.cnblogs.com/cgmcoding/p/15339638.html
from matplotlib import pyplot as plt


def shap_analysis(data, label, name):
    '''
    :param data: 特征数据
    :param label: 标签数据
    :param name: 预后结果名称
    :return:
    '''
    cols = data.columns
    model = xgb.XGBRegressor()
    model.fit(data, label)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    j = 15
    # 展示前15个特征的数据信息
    # 特征名称、特征值、shap值、预测平均分数、特征shap和
    player_explainer = pd.DataFrame()
    player_explainer['feature'] = cols
    player_explainer['feature_value'] = data.iloc[j].values
    player_explainer['shap_value'] = shap_values[j]
    player_explainer['base'] = model.predict(data).mean()  # 就是预测的分数的均值
    player_explainer['sum'] = player_explainer['shap_value'].sum()  # 特征的shap和
    player_explainer['base+sum'] = player_explainer['base'] + player_explainer['sum']
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, data)


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
        shap_analysis(data, label_new, name)
