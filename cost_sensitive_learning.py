import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.utils import shuffle

default_params = {'max_depth': 6, 'eta': 0.1, 'silent': 1, 'eval_metric': 'auc', "objective": "binary:logistic"}


# https://blog.csdn.net/zwqjoy/article/details/109311133
def focal_loss(p, dtrain):
    alpha = 2
    gamma = 0.0009
    # y, p, alpha, gamma = symbols('y p alpha gamma')
    y = dtrain.get_label()
    p = 1 / (1 + np.exp(-p))
    loss = alpha * ((1 - p) ** gamma) * (-y) * np.log(p) - (1 - alpha) * (p ** gamma) * (1 - y) * np.log(1 - p)
    print('loss : '+str(loss))
    grad = p * (1 - p) * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (
                1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p + p ** gamma * (
                                      1 - alpha) * (1 - y) / (1 - p))
    hess = p * (1 - p) * (p * (1 - p) * (
            -alpha * gamma ** 2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + alpha * gamma * y * (
            1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + 2 * alpha * gamma * y * (1 - p) ** gamma / (
                    p * (1 - p)) + alpha * y * (1 - p) ** gamma / p ** 2 - gamma ** 2 * p ** gamma * (
                    1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + 2 * gamma * p ** gamma * (1 - alpha) * (
                    1 - y) / (p * (1 - p)) + gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
        1 - p) / p ** 2 + p ** gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2) - p * (
                                  alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (
                                  1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
                              1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)) + (1 - p) * (
                                  alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (
                                  1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
                              1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)))

    # print('grad : ' + str(grad) + ' hess : ' + str(hess))
    return grad, hess


if __name__ == '__main__':
    dataframe = shuffle(pd.read_csv('ARDS/eicu/result/0801_fill_with_0.csv', sep=','))
    data = np.array(dataframe.iloc[:, 1:-5])
    label = np.array(dataframe.iloc[:, -5])
    label_new = []
    for item in label:
        if item == 0:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    KF = KFold(n_splits=5)
    model = xgb
    for train_index, test_index in KF.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label_new[train_index], label_new[test_index]
        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)
        x_train = model.DMatrix(x_train, label=y_train)
        x_test = model.DMatrix(x_test, label=y_test)
        model.train(params=default_params, dtrain=x_train, num_boost_round=3, early_stopping_rounds=50,
                    evals=[(x_train, 'train'), (x_test, 'eval')], verbose_eval=1)
        print('focal loss---------------')
        model.train(params=default_params, dtrain=x_train, num_boost_round=3, early_stopping_rounds=50,
                    evals=[(x_train, 'train'), (x_test, 'eval')], verbose_eval=1, obj=focal_loss)
