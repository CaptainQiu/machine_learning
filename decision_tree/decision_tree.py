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





if __name__ == "__main__":
    X=[[1,1,1]for i in range(10)]
    Y=[0 for i in range(1)]+[1 for i in range(7)]+[2 for i in range(2)]
    print(calcEntropy(X,Y))