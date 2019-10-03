import numpy

class LDA:

    def fit(self, X, y):
        self.n = len(y)
        if (self.n > 0):
            self.m = len(X[0])
        else:
            self.m = 0
        self.X0 = []
        self.y0 = []
        self.X1 = []
        self.y1 = []
        for i in range(self.n):
            if (y[i] == 0):
                self.X0.append(X[i])
                self.y0.append(y[i])
            else:
                self.X1.append(X[i])
                self.y1.append(y[i])
        self.py0 = float(len(self.y0))/float(self.n)
        self.py1 = float(len(self.y1))/float(self.n)
        if (len(self.X0) == 0):
            self.u0 = numpy.zeros([1, self.m])
        else:
            self.u0 = numpy.mean(self.X0, axis=0)
        if (len(self.X1) == 0):
            self.u1 = numpy.zeros([1, self.m])
        else:
            self.u1 = numpy.mean(self.X1, axis=0)
        self.covariance = numpy.zeros([self.m, self.m])
        for i in range(len(self.X0)):
            self.covariance = self.covariance + numpy.multiply((self.X0[i] - self.u0), numpy.transpose([self.X0[i] - self.u0]))
        for i in range(len(self.X1)):
            self.covariance = self.covariance + numpy.multiply((self.X1[i] - self.u1), numpy.transpose([self.X1[i] - self.u1]))
        self.covariance = self.covariance/(self.n-2)
        return

    def predictOne(self, x):
        if (numpy.log(self.py1/self.py0) - numpy.matmul(numpy.matmul(self.u1, numpy.linalg.inv(self.covariance)), self.u1) / 2 + numpy.matmul(numpy.matmul(self.u0, numpy.linalg.inv(self.covariance)), self.u0) / 2 + numpy.matmul(numpy.matmul(x, numpy.linalg.inv(self.covariance)), self.u1 - self.u0)) > 0:
            return 1
        return 0

    def predict(self, X):
        result = []
        for i in range(len(X)):
            result.append(self.predictOne(X[i]))
        return result

    def test(self, X, y):
        hits = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                hits = hits + 1
        return float(hits)/float(len(X))
