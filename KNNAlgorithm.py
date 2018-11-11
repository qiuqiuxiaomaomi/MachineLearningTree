#coding=utf-8
# 算法基本流程
#     1）计算已知类别数据集中的点与当前点之间的距离
#     2）按照距离递增次序排序
#     3）选取与当前点距离最小的k个点
#     5）确定前k个点所在类别的出现频率
#     6）返回前k个点出现评率最高的类别作为当前点的预测分类


from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0],[0,0.1]])
    labels = ['a', 'a','b','b']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffNat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffNat = diffNat ** 2
    sqDistances = sqDiffNat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDisIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + i

    sortedClassCount = sorted(classCount.iteritems(),
                              key =operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]