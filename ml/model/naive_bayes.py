import pandas as pd
import numpy as np
import json


class NaiveBayes:
    def __init__(self):
        self.model = {}  # key 为类别名 val 为字典PClass表示该类的该类，PFeature:{}对应对于各个特征的概率

    def calEntropy(self, y):  # 计算熵
        valRate = y.value_counts().apply(lambda x: x / y.size)  # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain=pd.Series()):
        # if not yTrain.empty:  # 如果不传，自动选择最后一列作为分类标签
        #     xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain, yTrain)
        return self.model

    def buildNaiveBayes(self, xTrain, yTrain):
        # yTrain = xTrain.iloc[:, -1]

        yTrainCounts = yTrain.value_counts()  # 频次汇总 得到各个特征对应的概率

        yTrainCounts = yTrainCounts.apply(lambda x: (x + 1) / (yTrain.size + yTrainCounts.size))  # 使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature': {}}

        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)
        # print(allPropByFeature)
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = propDatas.value_counts()  # 频次汇总 得到各个特征对应的概率
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0  # 如果有属性灭有，那么自动补0
                Ni = len(allPropByFeature[nameFeature])
                propClassSummary = propClassSummary.apply(lambda x: (x + 1) / (propDatas.size + Ni))  # 使用了拉普拉斯平滑
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature

        return retModel

    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']

            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                rate += np.log(propsRate.get(val, 0))  # 使用log加法避免很小的小数连续乘，接近零
                # print(nameFeature, val, propsRate.get(val, 0))
            # print(nameClass, rate)
            if curMaxRate is None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass

        return curClassSelect

    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)


if __name__ == '__main__':
    dataframe = pd.read_csv('../../ARDS/eicu/result/fill_with_0.csv', sep=',', encoding='utf-8')
    data = dataframe.iloc[:, 1: -5]
    label = dataframe.iloc[:, -5]
    model = NaiveBayes()
    tree = model.fit(data, label)
    print(json.dumps(tree, ensure_ascii=False))
    pd = pd.DataFrame({'预测值': model.predict(data), '正取值': label})
    print(pd)
    print('正确率:%f%%' % (pd[pd['预测值'] == pd['正取值']].shape[0] * 100.0 / pd.shape[0]))
