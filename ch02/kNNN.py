# coding=utf-8
import operator  # 运算符模块
from os import listdir
from numpy import *  # 导入科学计算NumPy


# 定义一个函数，创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 具体的kNN算法
# 思想：计算已经知道类别数据集中的点和当前点之间的距离；
# 按照距离递增次序排序，选取与当前点距离最小的k个点，
# 确定前k个点所在类别的出现频率
# 返回前k个点出现频率最高的类别作为当前点的预测分类
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 得到矩阵大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 做差值
    sqDiffMat = diffMat ** 2  # 差值进行平方
    sqlDistances = sqDiffMat.sum(axis=1)
    distances = sqlDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 测试kNN算法
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # 去掉尾部的回车符
        listFromLine = line.split('\t')  # 将第一行数据按空格进行分割
        returnMat[index, :] = listFromLine[0:3]  # 0:3列为数据集的数据
        classLabelVector.append(int(listFromLine[-1]))  # 最后一列为分类标签
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # 构建新的全是0的新矩阵行数和列数与dataSet一致
    m = dataSet.shape[0]  # 获取列的长度
    normDataSet = dataSet - tile(minVals, (m, 1))  # 将变量内容复制c成输入矩阵同样大小的矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


# 计算错误率,查看分类的效果
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 得到数据集和标签
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 得到归一化后的矩阵
    m = normMat.shape[0]  # 得到行数
    numTestVecs = int(m * hoRatio)  # 确定用来做测试的样本数量
    errorCount = 0.0  # 错误数量
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print "the classifer came back with %d,the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print "the total error rate is %f" % (errorCount / float(numTestVecs))


# change image to vector
def img2vector(filename):
    returnVect = zeros((1, 1024))  # create 1*1024 size of array
    fr = open(filename)  # open file
    for i in range(32):  # cycle 32 row
        lineStr = fr.readline()  # read one line
        for j in range(32):  # cycle 32 column
            returnVect[0, 32 * i + j] = int(lineStr[j])  # storage data
    return returnVect


# handwritten digit recognition system of test code
def handlewritingClassTest():
    hwLabels = []
    trainingFileList = listdir(
        'trainingDigits')  # get directory content
    m = len(trainingFileList)  # get number of directory
    trainingMat = zeros((m, 1024))  # create m row 1024 column of matrix
    for i in range(m):  # every row storage a image
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)  # 从文件名中解析出分类数字
        trainingMat[i:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnterTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnterTest, trainingMat, hwLabels, 3)
        print "the classfier came back with: %d,the real answer is %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is %d" % errorCount
    print "\nthe total error rate is :%f" % (errorCount / float(mTest))
