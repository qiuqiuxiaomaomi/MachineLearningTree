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