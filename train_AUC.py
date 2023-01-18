import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

## 把所有的相同类别的特征编码为同一个值
import ml.classification.classify_parameter


def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(), range(len(x.unique().tolist()))))

    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1

    return mapfunction


if __name__ == "__main__":
    # 1. 载入数据
    data = pd.read_csv('ARDS/eicu/pictures/0907/fill_with_average.csv').iloc[:, 1:-4]
    print(data.head())

    # 2. 对类别特征编码   
    numerical_features = [x for x in data.columns if data[x].dtype == np.float]
    category_features = [x for x in data.columns if data[x].dtype != np.float]
    for i in category_features:
        data[i] = data[i].apply(get_mapfunction(data[i]))
    print(data.head())

    # 3. 准备训练集和测试集
    label = np.array(data.iloc[:, -1])
    Y = []
    for item in label:
        if item == 1:
            Y.append(1)
        else:
            Y.append(0)
    Y = np.array(Y)
    X = np.array(data.iloc[:, :-1])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1224)
    print('x_train shape : %s,y_train shape : %s' % (x_train.shape, y_train.shape))

    # 4. 训练 
    from xgboost.sklearn import XGBClassifier
    from sklearn import metrics

    # Xgboost的参数及调参 
    # https://blog.csdn.net/iyuanshuo/article/details/80142730
    # https://blog.csdn.net/qq_43468807/article/details/106238362
    # https://blog.csdn.net/u010657489/article/details/51952785?

    # 4.1 定义模型 
    clf = XGBClassifier(n_estimators=25)

    # 4.2 使用5折交叉验证评估模型参数
    params = {
        'eta': [0.3],
        'max_depth': [6],
        'n_estimators': [25],
        'min_child_weight': [1, 2]
    }
    gsearch = GridSearchCV(clf, param_grid=ml.classification.classify_parameter.XGB_param, scoring='roc_auc', cv=5)
    gsearch.fit(x_train, y_train)
    print("Best score: %0.3f" % gsearch.best_score_)

    clf = gsearch.best_estimator_

    # 5. 评估  
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    test_predict_proba = clf.predict_proba(x_test)
    print('The train accuracy of the XGBoost is:', metrics.accuracy_score(y_train, train_predict))
    print('The test accuracy of the XGBoost is:', metrics.accuracy_score(y_test, test_predict))

    # 6. AUC曲线绘制
    fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid()
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    plt.legend(loc='lower right', fontsize=7)
    plt.show()
