import numpy as np
import xgboost
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import BayesianRidge

xgboost.set_config(verbosity=0)

# 进行以下14个模型的训练

# 基础模型
GBDT_none = GradientBoostingClassifier(n_estimators=100)
XGBoost_none = XGBClassifier(n_estimators=400)
RF_none = RandomForestClassifier()
adaboost_none = AdaBoostClassifier()
linearRegression_none = LinearRegression()
bagging_none = BaggingClassifier()
BayesRidge_none = BayesianRidge()
LR_none = LogisticRegression()
KNN_none = KNeighborsClassifier()
stacking_none = StackingCVClassifier(classifiers=[KNN_none, GBDT_none, XGBoost_none], meta_classifier=LR_none)
DT_none = DecisionTreeClassifier()
SVM_none = SVC(probability=True)
NaiveBayes_none = MultinomialNB()
Perceptron_none = Perceptron()
base_param = {'n_estimators': [25],
              'learning_rate': [0.01]}
# 手动参数
GBDT = GradientBoostingClassifier(max_depth=6, max_features='log2')
XGB = XGBClassifier(gamma=0.3, max_depth=6)
RF = RandomForestClassifier(random_state=42, max_features='sqrt', oob_score=False, criterion='gini')
adaboost = AdaBoostClassifier(random_state=42)
LinearRe = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
bagging = BaggingClassifier(base_estimator=DT_none, max_samples=0.8, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=42, n_estimators=9)
BayesRidge = BayesianRidge(tol=0.001, alpha_1=1e-06,
                           alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None,
                           compute_score=False, fit_intercept=True, copy_X=True, verbose=False)
GBDT_Grid = GradientBoostingClassifier(max_depth=6, max_features='log2')
LR = LogisticRegression(max_iter=10, multi_class='auto', penalty='l2')
LR_ = LogisticRegression(C=1, penalty='l1', solver='liblinear')
KNN = KNeighborsClassifier()
DT = DecisionTreeClassifier(splitter="best", max_depth=3, max_features='sqrt', random_state=42)  # 选择决策树为基本分类器
SVM = SVC(probability=True, kernel='linear')
Perceptron_ = Perceptron(fit_intercept=False, shuffle=False, max_iter=10000)
stacking = StackingCVClassifier(classifiers=[KNN, GBDT, XGB], meta_classifier=LR)
# 调优参数搜索
GBDT_param = {
    'n_estimators': [25, 50, 100, 200, 500, 1000],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.01, 0.015, 0.02]
    # 'n_estimators': [25],
    # 'learning_rate': [0.01]
}
GBDT_Grid_param = {'n_estimators': [700, 750, 800],
                   'learning_rate': [0.008, 0.01, 0.015, 0.02]}
LR_param = {'C': [np.linspace(0.1, 1, 5)],
            # 'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs', 'sag'],
            # 'max_iter': range(100, 1000, 100)
            }
KNN_param = {'n_neighbors': list(range(10, 70, 10)), 'weights': ['uniform', 'distance'], 'p': range(1, 7, 1)}
bagging_param = {'n_estimators': range(1, 10, 1)}
adaboost_param = {'n_estimators': [700, 750, 800], 'learning_rate': [0.015, 0.018, 0.02]}
RF_param = {'n_estimators': range(100, 1000, 100), 'max_depth': range(2, 10, 2), 'max_features': ['sqrt', 'log2', 0.75]}
DT_param = {'criterion': ['gini', 'entropy'], 'max_features': ['log2', 'sqrt'], 'max_depth': range(2, 10, 2)}
XGB_param = {
    # 'max_depth': [7, 9],
    # 'min_child_weight': [1, 3, 5],
    # 'gamma': [i / 10.0 for i in range(0, 5)],
    # 'subsample': [i / 10.0 for i in range(6, 10)],
    # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'eta': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
    'n_estimators': [10, 25, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
}
SVM_param = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': [0.001, 0.0001, 0.01, 0.1, 0.5, 1]
}
Perceptron_param = {'penalty': ['none', 'l2'], 'max_iter': range(100, 1000, 100)}
BayesRidge_param = {'n_iter': range(100, 1000, 100)}
boosting_param = {'n_estimators': range(10, 100, 10), 'learning_rate': [0.001, 0.01, 0.005, 0.1, 0.05, 0.15], }
linearreg_param = {}

# 网格搜索
GBDT_GridSearchCV = GridSearchCV(GBDT_Grid, GBDT_Grid_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
XGB_RandomizedSearchCV = RandomizedSearchCV(XGB, XGB_param, n_iter=13, cv=5)
RF_RandomizedSearchCV = RandomizedSearchCV(RF, RF_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
boosting_RandomizedSearchCV = RandomizedSearchCV(adaboost, boosting_param, scoring='roc_auc', n_jobs=3,
                                                 refit=True, cv=5, verbose=0)
linearreg_RandomizedSearchCV = RandomizedSearchCV(linearRegression_none, linearreg_param, scoring='roc_auc', n_jobs=3,
                                                  refit=True, cv=5, verbose=0)
bagging_GridSearchCV = GridSearchCV(bagging, bagging_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
BayesianRidge_RandomizedSearchCV = RandomizedSearchCV(BayesRidge, BayesRidge_param, scoring='roc_auc', n_jobs=3,
                                                      refit=True, cv=5, verbose=0)
GBDT_RandomizedSearchCV = RandomizedSearchCV(GBDT, GBDT_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
LR_RandomizedSearchCV = RandomizedSearchCV(LR, LR_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
KNN_GridSearchCV = GridSearchCV(KNN_none, KNN_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
KNN_RandomizedSearchCV = RandomizedSearchCV(KNN_none, KNN_param, n_iter=13, cv=5)
bagging_RandomizedSearchCV = RandomizedSearchCV(bagging, bagging_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5,
                                                verbose=0)
DT_RandomizedSearchCV = RandomizedSearchCV(DT, DT_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
SVM_RandomizedSearchCV = RandomizedSearchCV(SVM, SVM_param, scoring='roc_auc', n_jobs=3, refit=True, cv=5, verbose=0)
Perceptron_RandomizedSearchCV = RandomizedSearchCV(Perceptron_, Perceptron_param, scoring='roc_auc', n_jobs=3,
                                                   refit=True, cv=5, verbose=0)
stacking_RandomizedSearchCV = RandomizedSearchCV(stacking, {}, scoring='roc_auc', n_jobs=3,
                                                 refit=True, cv=5, verbose=0)
models = [GBDT_RandomizedSearchCV, XGB_RandomizedSearchCV, RF_RandomizedSearchCV, boosting_RandomizedSearchCV,
          linearreg_RandomizedSearchCV, bagging_RandomizedSearchCV, BayesianRidge_RandomizedSearchCV,
          LR_RandomizedSearchCV, KNN_RandomizedSearchCV, stacking_RandomizedSearchCV, DT_RandomizedSearchCV,
          SVM_RandomizedSearchCV, Perceptron_RandomizedSearchCV]
searchCVnames = ['GBDT', 'XGBoost', 'Random Forest', 'Boosting', 'Linear Regression', 'Bagging', 'Bayesian Ridge',
                 'Logistic Regression', 'KNN', 'Stacking', 'Decision Tree', 'SVM', 'Perceptron']
searchCVnames_ab = ['GBDT', 'XGBoost', 'RF', 'Boosting', 'LinR', 'Bagging', 'BR', 'LogR', 'KNN', 'Stacking', 'DT',
                    'SVM', 'Perceptron']
searchCVnames_ab_clinical = ['GBDT', 'XGBoost', 'RF', 'AdaBoost', 'LinR', 'Bagging', 'BR', 'KNN', 'Stacking', 'DT',
                             'SVM', 'Perceptron']
GBDT_tunned = GradientBoostingClassifier(learning_rate=0.01, n_estimators=750, max_depth=6, max_features='log2',
                                         subsample=0.5)
GBDT_origin = GradientBoostingClassifier(max_depth=6, max_leaf_nodes=1, n_estimators=25)
LR_tunned = LogisticRegression(solver='lbfgs', max_iter=1000)
KNN_tunned = KNeighborsClassifier(weights='uniform', p=1, n_neighbors=40)
bagging_tunned = BaggingClassifier(n_estimators=9)
RF_tunned = RandomForestClassifier(n_estimators=800, max_features='sqrt', max_depth=8, random_state=42)
DT_tunned = DecisionTreeClassifier(max_features='sqrt', max_depth=8, criterion='entropy', splitter='best')
XGB_tunned = XGBClassifier(booster='gbtree', n_estimators=900, learning_rat=0.01, reg_lambda=0.05, reg_alpha=0.5,
                           max_depth=6)
SVM_tunned = SVC(probability=True, gamma=0.0001, C=0.01)
NaiveBayes_tunned = MultinomialNB()
Perceptron_tunned = Perceptron(random_state=42, fit_intercept=False, shuffle=False)
BayesRidge_tunned = BayesianRidge(n_iter=100)
stacking_tunned = StackingCVClassifier(classifiers=[KNN_none, RF_none, SVM_none],  # 第一层分类器
                                       meta_classifier=LR_none)
adaboost_tunned = AdaBoostClassifier(n_estimators=90, learning_rate=0.15)
linearreg_tunned = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
tunned_model = [GBDT_tunned, XGB_tunned, RF_tunned, adaboost_tunned, bagging_tunned, stacking_tunned,
                BayesRidge_tunned, LR_tunned, linearreg_tunned, Perceptron_tunned]
base_models = [GBDT_none, XGBoost_none, RF_none, adaboost_none, linearRegression_none, bagging_none, BayesRidge_none,
               LR_none, KNN_none, stacking_none, DT_none, SVM_none, Perceptron_none]
base_models_clical = [GBDT_none, XGBoost_none, RF_none, adaboost_none, linearRegression_none, bagging_none,
                      BayesRidge_none, KNN_none, stacking_none, DT_none, SVM_none, Perceptron_none]
base_models_sec = [XGBoost_none, RF_none, linearRegression_none, DT_none, SVM_none, BayesRidge_none, LR_none, KNN_none]
model_names_sec = ['XGBoost', 'RF', 'LinR', 'DT', 'SVM', 'BR', 'LogR', 'KNN']
params = [GBDT_param, XGB_param, RF_param, adaboost_param, bagging_param, LR_param, SVM_param, Perceptron_param]
GBDT_SR = GradientBoostingClassifier(learning_rate=0.015, n_estimators=700, max_depth=6, max_features='log2',
                                     subsample=0.5)
names = ['GBDT', 'XGBoost', 'RandomForest', 'Adaboost', 'Bagging', 'LogisticRegression', 'SVM', 'Perceptron']
GBDT_LS = GradientBoostingClassifier(learning_rate=0.02, n_estimators=750, max_depth=6, max_features='log2',
                                     subsample=0.5)
GBDT_RD = GradientBoostingClassifier(learning_rate=0.02, n_estimators=800, max_depth=6, max_features='log2',
                                     subsample=0.5)
XGBoost_SR = XGBClassifier(booster='gbtree', n_estimators=1000, learning_rate=0.01, reg_lambda=0.05, reg_alpha=0.5,
                           max_depth=6)
XGBoost_LS = XGBClassifier(booster='gbtree', n_estimators=1000, learning_rat=0.01, reg_lambda=0.05, reg_alpha=0.5,
                           max_depth=6)
XGBoost_RD = XGBClassifier(booster='gbtree', n_estimators=900, learning_rate=0.01, reg_lambda=0.05, reg_alpha=0.5,
                           max_depth=6)
