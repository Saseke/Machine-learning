# coding=utf-8
from math import log


# 计算给定数据集合的香农熵
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 计算数据集中实例的总数
    labelCounts = {}  # 构建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 取数据的最后一列的数值作为键值
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 每个键值都记录下当前类别出现的次数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires  # 通过已知的频率去求类别出现的概率
        shannonEnt -= prob * log(prob, 2)  # base 2  利用概率计算香农熵
    return shannonEnt


# 自定义的数据集合
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按照给定特征进行划分
# dataSet是需要划分的数据集，axis是划分的列的名字,value是列的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建新的list对象,python在函数中传递的列表的引用，由于需要多次进行调用，所以在内部新定义了一个list集合
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉分类列添加到新的集合中
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# choose the best feature to split
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据当前的信息熵
    bestInfoGain = 0.0  # 定义最大的信息增益
    bestFeature = -1  # 定义分割后信息增益最大的特征
    for i in range(numFeature):  # 遍历数据集合中的所有特征值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 将特征对应的值放在一个集合中，使得特征列的数据具有唯一性
        newEntropy = 0.0
        # 遍历特征列d额所有值，分别计算信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
