import numpy as np
import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas_profiling.utils import dataframe
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

from dimension_reduction.univariate_analyse import univariate_analysis
from filter.common import standard_data_by_white
from filter.param import *
from ml.classification.classify_parameter import *
import matplotlib as mpl


def calculate_metrics(gt, pred):
    """
    :param gt: 数据的真实标签，一般对应二分类的整数形式，例如:y=[1,0,1,0,1]
    :param pred: 输入数据的预测值，因为计算混淆矩阵的时候，内容必须是整数，所以对于float的值，应该先调整为整数
    :return: 返回相应的评估指标的值
    """
    """
        confusion_matrix(y_true,y_pred,labels,sample_weight,normalize)
        y_true:真实标签；
        y_pred:预测概率转化为标签；
        labels:用于标签重新排序或选择标签子集；
        sample_weight:样本权重；
        normalize:在真实（行）、预测（列）条件或所有总体上标准化混淆矩阵；
    """
    print("starting!!!-----------------------------------------------")
    sns.set()
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=2)
    confusion = confusion_matrix(gt, pred)
    # 打印具体的混淆矩阵的每个部分的值
    print(confusion)
    # 从左到右依次表示TN、FP、FN、TP
    print(confusion.ravel())
    # 绘制混淆矩阵的图
    sns.heatmap(confusion, annot=True, cmap='Blues', linewidths=0.5, ax=ax1)
    ax2.set_title('sns_heatmap_confusion_matrix')
    ax2.set_xlabel('y_pred')
    ax2.set_ylabel('y_true')
    fig.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
    # 混淆矩阵的每个值的表示
    (TN, FP, FN, TP) = confusion.ravel()
    # 通过混淆矩阵计算每个评估指标的值
    print('AUC:', roc_auc_score(gt, pred))
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))
    print('PPV:', TP / float(TP + FP))
    print('Recall:', TP / float(TP + FN))
    print('Precision:', TP / float(TP + FP))
    # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    P = TP / float(TP + FP)
    R = TP / float(TP + FN)
    print('F1-score:', (2 * P * R) / (P + R))
    print('True Positive Rate:', round(TP / float(TP + FN)))
    print('False Positive Rate:', FP / float(FP + TN))
    print('Ending!!!------------------------------------------------------')

    # 采用sklearn提供的函数验证,用于对比混淆矩阵方法与这个方法的区别
    print("the result of sklearn package")
    auc = roc_auc_score(gt, pred)
    print("sklearn auc:", auc)
    accuracy = accuracy_score(gt, pred)
    print("sklearn accuracy:", accuracy)
    recal = recall_score(gt, pred)
    precision = precision_score(gt, pred)
    print("sklearn recall:{},precision:{}".format(recal, precision))
    print("sklearn F1-score:{}".format((2 * recal * precision) / (recal + precision)))


# 展示真实值和预测值
def plot_test_and_predict(y_test, y_pred):
    # 为了正常显示中文
    mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    mpl.rcParams['font.size'] = 12  # 字体大小
    plt.plot(y_pred, label='预测预后')
    plt.plot(y_test, label='真实预后')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    # 未归一化数据
    # 解决RuntimeWarning: invalid value encountered in true_divide
    data = pd.read_csv('clinical_exam/test.csv').iloc[:, 2:]
    new_data = []
    for row in np.array(data):
        new_row = []
        for item in row:
            if type(item) == np.str:
                if '<' in item:
                    item = item[1:]
                new_row.append(float(item))
            else:
                new_row.append(round(item, 3))
        new_data.append(new_row)
    DataFrame(new_data).to_csv('clinical_exam/test_1.csv', encoding='utf-8', index=False)
    columns = data.columns[1:]
    new_data = np.array(new_data)
    label = np.array(new_data[:, 0])
    data = np.array(new_data[:, 1:])
    test_analyser = univariate_analysis(columns, datatype='test')
    for model, name, color, mark in zip(base_models, searchCVnames, colors[:len(base_models)],
                                        marks[:len(base_models)]):
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True,
                                                            random_state=42)
        # test_analyser.univariate_analysis(1, x_train, x_test, y_train, y_test)
        x_train, x_test = standard_data_by_white(x_train, x_test)
        # model_research = GridSearchCV(model, scoring='roc_auc', n_jobs=10, refit=True, cv=5, verbose=0)
        model.fit(x_train, y_train)
        if name == 'Perceptron':
            test_predict_proba = model._predict_proba_lr(x_test)
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1])
        elif name == 'Linear Regression' or name == 'Bayesian Ridge':
            test_predict_proba = model.predict(x_test)
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba)
        else:
            test_predict_proba = model.predict_proba(x_test)
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1])
        y_predict = model.predict(x_test)
        if name != 'Linear Regression' and name != 'Bayesian Ridge':
            accuracy = round(accuracy_score(y_test, y_predict), 3)
            precision = round(precision_score(y_test, y_predict), 3)
            recall = round(recall_score(y_test, y_predict), 3)
            F1 = round(f1_score(y_test, y_predict), 3)
        else:
            accuracy = 0
            precision = 0
            recall = 0
            F1 = 0
        # 求auc时，我们必须用predict_proba。因为roc曲线的阀值是根据其正样本的概率求的。
        auc = round(metrics.auc(fpr, tpr), 3)
        plt.plot(fpr, tpr, color=color, label=r'%s (area=%0.3f)' % (name, auc), lw=1, marker=mark, markersize=2)
        print('model : %s auc : %s  accuracy : %s  precision : %s  recall : %s  F1 : %s ' % (
            name, auc, accuracy, precision, recall, F1))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - specificity', fontweight='bold', fontsize=15)
    plt.ylabel('sensitivity', fontweight='bold', fontsize=15)
    plt.title('ROCs', fontsize=17)
    plt.legend(loc='lower right', fontsize=7)
    plt.grid()
    plt.savefig('multiple_model.png')
    plt.show()





