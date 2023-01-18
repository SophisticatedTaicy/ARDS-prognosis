from math import floor

import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import xgboost as xgb

import numpy as np
from numpy.random import shuffle

from scipy.stats import uniform, randint

from utils import load_data

# HYPER_PARAMETERS = {
#     "learning_rate": [0.04, 0.06, 0.08, 0.10, 0.12],
#     "n_estimators":  [200, 300, 400],
#     "max_depth":     [5, 6, 7]
# }
# """
# n_components: 120, trainnig auc: 0.731769, testing auc: 0.625646, param: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 400}
# n_components: 130, trainnig auc: 0.727230, testing auc: 0.627540, param: {'learning_rate': 0.08, 'max_depth': 7, 'n_estimators': 400}
# n_components: 140, trainnig auc: 0.730589, testing auc: 0.630464, param: {'learning_rate': 0.08, 'max_depth': 7, 'n_estimators': 400}
# n_components: 150, trainnig auc: 0.733399, testing auc: 0.608625, param: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 400}
# n_components: 160, trainnig auc: 0.731458, testing auc: 0.626899, param: {'learning_rate': 0.12, 'max_depth': 7, 'n_estimators': 400}
# n_components: 170, trainnig auc: 0.733820, testing auc: 0.618797, param: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 400}
# n_components: 180, trainnig auc: 0.733493, testing auc: 0.625665, param: {'learning_rate': 0.12, 'max_depth': 7, 'n_estimators': 300}
# """

# HYPER_PARAMETERS = {
#     "colsample_bytree": uniform(0.7, 0.3),
#     "gamma":            uniform(0, 0.5),
#     "learning_rate":    uniform(0.03, 0.3), # default 0.1 
#     "max_depth":        randint(2, 6),      # default 3
#     "n_estimators":     randint(100, 150),  # default 100
#     "subsample":        uniform(0.6, 0.4)
# }

# HYPER_PARAMETERS = {
#     "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7], 
#     "gamma":            [0.1, 0.2, 0.3, 0.4, 0.5], 
#     "learning_rate":    [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], 
#     "max_depth":        [2, 3, 4, 5, 6, 7], 
#     "n_estimators":     [20, 40, 60, 80, 150, 200, 300, 400], 
#     "subsample":        [0.4, 0.6]
# }
# """
# trainnig auc: 0.991373, testing auc: 0.645960, param: {'colsample_bytree': 0.4, 'gamma': 0.2, 'learning_rate': 0.06, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.6}
# """

HYPER_PARAMETERS = {
    "colsample_bytree": [0.1, 0.2, 0.3, 0.4], 
    "gamma":            [0.2], 
    "learning_rate":    [0.04, 0.06, 0.08], 
    "max_depth":        [6, 7], 
    "n_estimators":     [300, 400], 
    "subsample":        [0.6]
}
"""
trainnig auc: 0.991373, testing auc: 0.645960, param: {'colsample_bytree': 0.4, 'gamma': 0.2, 'learning_rate': 0.06, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.6}
"""


# HYPER_PARAMETERS = {
#     "learning_rate": [0.1],
#     "n_estimators":  [25],
# }

def train(X, Y, n_components):
    # if n_components < 200:
    #     pca = PCA(n_components=n_components)
    #     X = pca.fit_transform(X)
    model = GridSearchCV(xgb.XGBClassifier(objective="binary:logistic", tree_method='gpu_hist'), HYPER_PARAMETERS, scoring='roc_auc', n_jobs=12, cv=5)
    # model = GridSearchCV(xgb.XGBClassifier(objective="binary:logistic"), HYPER_PARAMETERS, scoring='roc_auc', n_jobs=12, cv=5)
    model.fit(X, Y)
    return model.best_estimator_, model.best_params_, model.best_score_, None # pca

if __name__ == "__main__":
    data = load_data("data.csv")[..., :-4]
    for i in range(len(data)): 
        if data[i, -1] != 1: data[i, -1] = 0
    shuffle(data)
    data_train, data_test = data[:floor(0.8 * len(data))], data[floor(0.8 * len(data)):]
    X_train, Y_train = data_train[..., :-1], data_train[..., -1]
    X_test,  Y_test  = data_test[..., :- 1], data_test[...,  -1]

    # for n_components in [25, 50, 75, 100, 125, 150, 175, 200]:
    # for n_components in [100, 120, 140, 160, 180]:
    # for n_components in [120, 130, 140, 150, 160, 170, 180]:

    model, param, score_train, pca = train(X_train, Y_train, 200)
    score_train = roc_auc_score(Y_train, model.predict(X_train))
    # score_test  = roc_auc_score(Y_test,  model.predict(pca.transform(X_test)), multi_class='ovr')
    score_test  = roc_auc_score(Y_test,  model.predict(X_test), multi_class='ovr')
    print(f"trainnig auc: {score_train:.06f}, testing auc: {score_test:.06f}, param: {param}")
        # print(f"n_components: {n_components:3}, trainnig auc: {score_train:.06f}, testing auc: {score_test:.06f}, param: {param}")