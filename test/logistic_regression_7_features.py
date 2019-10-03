import numpy as np
import pandas as pd

# Clean the data
def cleanData(fileName):
    dataset = pd.read_csv(fileName, ';')
    index = 0
    for q in dataset['quality']:
        if q > 5:
            dataset.at[index, 'quality'] = 1
        else:
            dataset.at[index, 'quality'] = 0
        index = index + 1
    dataset.dropna()
    
# Drop the irrelevant features
    cleaned_data = dataset.drop(['residual sugar', 'free sulfur dioxide', 'density', 'pH'], axis=1)

    pos = cleaned_data[cleaned_data['quality']==1]
    neg = cleaned_data[cleaned_data['quality']==0]
#    print(pos.describe())
#    print(neg.describe())
    cleaned_pos = dropData(pos)
    cleaned_neg = dropData(neg)
    result = pd.concat([cleaned_pos, cleaned_neg])
    return result
    

# Drop the irrelevant data (out of two-standard-deviation)
def dropData(dataset):
    (row_size, col_size) = dataset.shape
    describe = dataset.describe()
    outOfBound = False
    badData = []
    # Do not consider quality here
    for col_index in range(col_size-1):
        mean = describe.iat[1, col_index]
        std = describe.iat[2, col_index]
#        print(mean, std)
        upper_bound = mean + 2*std
        lower_bound = mean - 2*std
        for row_index in range(row_size):
            cell_value = dataset.iat[row_index, col_index]
            if(cell_value>upper_bound or cell_value<lower_bound):
                outOfBound = True
            if(outOfBound == True):
                if(not (row_index in badData)):
                    badData = badData + [row_index]
                outOfBound = False
    cleaned_dataset = dataset.drop(dataset.index[badData])
    return cleaned_dataset

def logistic(a):
    return 1/(1+np.exp(-a))
def preVal(x, w):
    print(x)
    print(w)
    return w.dot(x)
# Here define w and x as series in pandas
def prob(x, w):
    return logistic(preVal(x,w))
def gradientD(wi, yi, x):
    #yi is a number which will be given in the LogReg class
    return yi-(prob(x, wi))
def bias(X):
    #X is a 2d array
    y = np.arange(X.shape[0],dtype=float)
    y = np.ones_like(y)
    X = np.insert(X,(X.shape[1]),y,axis=1)
    return X

class LogisticRegression:
    def __init__(self, w, X, flr=1, slr=0.1, num_it=50):
        self.flr=flr
        self.slr=slr
        self.num_it=num_it
        w = np.arange(X.shape[1]+1,dtype=float)
        w = np.zeros_like(w)
        self.w = w
    def __add_biasTerm(self, X):
        # intercept has been added here
        return bias(X)
    
    def fit_with_l2_regularization(self,X,Y,l2c):
        X = self.__add_biasTerm(X)
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
# X is the 2d array, Y is the set of actural value, w is the initial weight(all numpy array)
        X = self.__add_biasTerm(X)
        sum = np.arange(X.shape[1],dtype=float)
        sum = np.zeros_like(sum)
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
        X = self.__add_biasTerm(X)
        result = np.arange(X.shape[0])
        result = np.zeros_like(result)
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
    
data = cleanData('../winequality-red.csv')
print(data.shape)
