import time, numpy, random

def test(model, X, y):
        hits = 0
        result = model.predict(X)
        for i in range(len(X)):
            if result[i] == y[i]:
                hits = hits + 1
        return float(hits)/float(len(X))

def validate(model, k, X, y):
    splits=[]
    accuracies=[]
    times=[]
    shuffledX=[]
    shuffledy=[]
    shuffledIndices=list(range(len(X)))
    random.shuffle(shuffledIndices)
    for i in shuffledIndices:
        shuffledX.append(X[i])
        shuffledy.append(y[i])
    for i in range(k):
        start = int(i*len(shuffledX)/k)
        end = int((i+1)*len(shuffledX)/k)
        if i == k-1:
            end = len(shuffledX)
        split = SplitSet(shuffledX[0:start] + shuffledX[end:], shuffledy[0:start] + shuffledy[end:], shuffledX[start:end], shuffledy[start:end])
        splits.append(split)
    
    for split in splits:
        start_time = time.process_time()
        model.fit(numpy.array(split.Xtrain), numpy.array(split.ytrain))
        accuracies.append(test(model, numpy.array(split.Xvalidation), numpy.array(split.yvalidation)))
        times.append(time.process_time() - start_time)

    return {
        'accuracy' : sum(accuracies)/k,
        'time' : sum(times)/k,
    }

class SplitSet:
    def __init__(self, Xtrain, ytrain, Xvalidation, yvalidation):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xvalidation = Xvalidation
        self.yvalidation = yvalidation
