import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


def find_best_k(data, label):
    k_error = []
    k_range = range(1, 100)
    current_min_error = 1
    current_k = 1
    # 最佳56 0.2630824525605133，20可考虑 0.26725486726041103
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, data, label, cv=5, scoring='accuracy')
        error = 1 - score.mean()
        k_error.append(error)
        if error < current_min_error:
            current_min_error = error
            current_k = k
    plt.plot(k_range, k_error)
    plt.plot(current_k, current_min_error, color='r')
    plt.xlabel('Value of k for KNN')
    plt.ylabel('error')
    plt.show()
    return current_k


if __name__ == '__main__':
    dataframe = shuffle(pd.read_csv('../../ARDS/eicu/result/0801_fill_with_0.csv', sep=',', encoding='utf-8'))
    # 先找出最佳k值
    data = np.array(dataframe.iloc[:, 1:-5])
    result = np.array(dataframe.iloc[:, -5])
    find_best_k(data, result)
    data = MinMaxScaler().fit_transform(data)
