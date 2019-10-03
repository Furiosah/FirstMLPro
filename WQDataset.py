import csv

class WQDataset:
    def load(self, path='winequality-red.csv'):
        self.X = [];
        self.y = [];
        with open(path) as data:
            dataReader = csv.reader(data, delimiter=';')
            headers = next(dataReader, None)
            for row in dataReader:
                x = [];
                for i in range(len(row)):
                    if i == 11:
                        if int(row[i]) in [6, 7, 8, 9, 10]:
                            self.y.append(1)
                        else:
                            self.y.append(0)
                    else:
                        x.append(float(row[i]))
                self.X.append(x)
