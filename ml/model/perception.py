import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Perception(object):
    def __init__(self, x, y, eta):
        if x.shape[0] != y.shape[0]:
            raise ValueError('Error!.x and y must be same when axis =0')
        else:
            self.x = x
            self.y = y
            self.eta = eta

    def ini_percept(self):
        # 权重和偏移数据初始化
        weight = np.zeros(self.x.shape[1])
        b = 0
        mistake = True
        Vis = Plotting(self.x, self.y)
        number = 0
        while mistake:
            mistake = False
            Vis.open_in()
            Vis.vis_plot(weight, b, number)
            for i in range(self.x.shape[0]):
                if self.y[i] * (weight @ self.x[i] + b) <= 0:
                    weight += self.eta * self.y[i] * self.x[i]
                    b += self.eta * self.y[i]
                    number += 1
                    mistake = True
                    print('weight is : ' + str(weight) + ' b is : ' + str(b))
                    break
        Vis.close()
        return weight, b


class Plotting(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def open_in(self):
        plt.ion()

    def close(self):
        plt.ioff()
        plt.show()

    def vis_plot(self, weight, b, number):
        plt.cla()
        plt.xlim(0, np.max(self.X.T[0]) + 1)
        plt.ylim(0, np.max(self.X.T[1]) + 1)
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        if True in list(weight == 0):
            plt.plot(0, 0)
        else:
            x1 = -b / weight[0]
            x2 = -b / weight[1]
            plt.plot([x1, 0], [0, x2])
        plt.title('change time:%d' % number)
        number1 = "%05d" % number
        plt.savefig(r'result/%s.png' % number1)
        plt.pause(0.01)

    def just_plot_result(self, weight, b):
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        x1 = -b / weight[0]
        x2 = -b / weight[1]
        plt.plot([x1, 0], [0, x2])
        plt.show()


if __name__ == '__main__':
    dataframe = np.array(pd.read_csv('../../ARDS/eicu/result/fill_with_0.csv', sep=','))
    data = dataframe[:, 1:-5]
    label = dataframe[:, -5]
    label_new = []
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(-1)
    label_new = np.array(label_new)
    perception = Perception(data, label_new, 1)
    perception.ini_percept()
