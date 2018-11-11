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