import numpy as np
import pandas as pd
from math import log2


def calcEntropy(X,Y):
    numEntris=len(X)
    labelCounts={}
    for y in Y:
        if labelCounts.get(y)==None:
            labelCounts[y]=0
        labelCounts[y]+=1
    entropy=0.0
    for key in labelCounts.keys():
        prob=float(labelCounts[key])/numEntris
        entropy-=prob*log2(prob)
    return entropy
 
 
def splitDataSet(X,Y,axis,value):
    """
    按照value切分数据集
    """
    retDataSet=[]
    retY=[]
    for i,x in enumerate(X):
        if x[axis]==value:
            reducedx=x[:axis]
            reducedx.extend(x[axis+1:])
            retDataSet.append(reducedx)
            retY.append(Y[i])
    return retDataSet,retY

def chooseBestFeatureToSplit(X,Y):
    num_features=len(X[0])
    baseEntropy=calcEntropy(X,Y)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(num_features):
        featList=[example[i] for example in X]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subSet,subY=splitDataSet(X,Y,i,value)
            prob=len(subSet)/float(len(X))
            newEntropy+=prob*calcEntropy(subSet,subY)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
    




if __name__ == "__main__":
    X=[[1,1,1]for i in range(10)]
    Y=[0 for i in range(1)]+[1 for i in range(7)]+[2 for i in range(2)]
    print(calcEntropy(X,Y))
    X=[[1,1],[1,1],[1,0],[0,1],[0,1]]
    Y=['yes','yes','no','no','no']
    print(splitDataSet(X,Y,0,1))
    print(splitDataSet(X,Y,0,0))
    print(chooseBestFeatureToSplit(X,Y))