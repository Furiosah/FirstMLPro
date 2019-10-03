import numpy as np
import pandas as pd

def logistic(a):
    return 1/(1+np.exp(-a))
def preVal(x, w):
    return w.dot(x)
#here define w and x as series in pandas
def prob(x, w):
    return logistic(preVal(x,w))

def gradientD(wi, yi, x):
    #yi is a number which will be given in the LogReg class
    return yi-(prob(x, wi))
def bias(X):
    #X is a 2d array
    #print(X.shape)
    X = np.insert(X,(X.shape[1]),1,axis=1)
    return X
 
def evaluate_acc(Y, result):
    acc = 0.00
    i = 0
    count = 0

    while i<(Y.size):
        if Y[i]==result[i]:
            count = count+1
        i=i+1

    acc = count/Y.size
    return acc

class LogisticRegression:
    def __init__(self, flr, slr, num_it):
        self.flr=flr
        self.slr=slr
        self.num_it=num_it
    def fit_with_l2_regularization(self,X,Y,l2c):
        X = bias(X)
        sum = np.arange(X.shape[1],dtype=float)
        sum = np.zeros_like(sum)
        weight = self.w
        i=0
        while (i<self.num_it):
            for row,y in zip(X,Y):
                sum = np.add(row*(gradientD(weight,y,row)),sum)
            weight = np.add(weight,self.flr*sum)+2*(l2c*weight)
            i=i+1
        j=0
        while (j<self.num_it):
            for row,y in zip(X,Y):
                sum = np.add(row*(gradientD(weight,y,row)),sum)
            weight = np.add(weight,self.slr*sum)+2*(l2c*weight)
            j=j+1

        self.w = weight
        return
    def fit(self,X,Y):
        #X is the 2d array, Y is the set of actural value, w is the initial weight(all numpy array)
        X = bias(X)
        sum = np.arange(X.shape[1],dtype=float)
        sum = np.zeros_like(sum)
        self.w = np.zeros([1, X.shape[1]])
        weight = self.w
        i=0
        while (i<self.num_it):
            for row,y in zip(X,Y):
                sum = np.add(row*(gradientD(weight,y,row)),sum)
            weight = np.add(weight,self.flr*sum)
            i=i+1
        j=0
        while (j<self.num_it):
            for row,y in zip(X,Y):
                sum = np.add(row*(gradientD(weight,y,row)),sum)
            weight = np.add(weight,self.slr*sum)
            j=j+1
        self.w = weight
        return

    def predict(self, X):
        X = bias(X)
        result = np.zeros(X.shape[0])
        index=0
        #empty series for our predictions
        for row in X:
            preProb = prob(row,self.w)
            if preProb>0.5:
                result[index]=1
            else:
                result[index]=0
            index=index+1
        return result
