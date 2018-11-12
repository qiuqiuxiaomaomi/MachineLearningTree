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

# 使用k-近邻算法改进约会网站的配对效果
# 海伦一直使用在线与会网站寻找适合自己的约会对象，尽管约会网站会推荐不同的人选，但她没有从中找到喜欢的人
#     。经过一番总结，她发现曾交往过三种类型的人：
#            1）不喜欢的人
#            2）魅力一般的人
#            3）机具魅力的人
#      尽管发现上述规律，但海伦依然无法将约会网站推荐的匹配对象归入恰当的分类。她觉得可以再周一到周五约会那些魅力
#      一般的人，而周末则更喜欢与那些极具魅力的人为伴。海伦希望我们的分类软件可以更好的帮助她将匹配对象划分到确切
#     的分类中，此外，海伦还收集了一些约会网站未曾记录的数据信息，她认为这些数据更有助于匹配对象的归类。
#
#     在约会网站上使用k-近邻算法
#          1）收集数据：提供文本文件
#          2）准备数据：使用Python解析文本文件
#          3）分析数据：使用MatplotLib画二维扩散图
#          5）训练算法：此步骤不适用于k-近邻算法
#          6）测试算法：使用海伦提供的部分数据作为测试样本
#          7）使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros(numberOfLines, 3)
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index =+ 1

    return returnMat.classLabelVector

#    数值归一化
#            在处理这种不同取值范围的特征值时，我们通常采用的方法是将数值归一化，如将取值范围处理为-到1或者-1
#        到1之间，下面的共识将任意取值范围内的特征值转化为0到1区间内的值：
#                    newValue = (oldValue - min/(max -min))
#        其中min和max分别是数据集中的最小特征值和最大特征值。虽然改变数值取值范围增加了分类器的复杂度，但为了得到准确结果，必须这样做。

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    norDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = norDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 测试程序

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m.hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with ：%d, the real result is : %d"  % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

#使用算法：构建完整可用系统
    def classifyPerson():
        resultList = ['not at all', 'in small doses', 'in large doses']
        percentTats = float(raw_input("percentage of time spent playing video games?"))
        ffMiles = float_(raw_input("frequent flier miles earned per year?"))
        iceCream = float(raw_input("liters of ice cream consumed per year?"))
        datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
        normMat, ranges, minVals = autoNorm(datingDataMat)
        inArr = array([ffMiles, percentTats, iceCream])
        classifierResult = classify0((inArr - minVals)/ ranges, normMat, datingLabels, 3)
        print "You will probably like this person: " resultList[classifierResult - 1]
