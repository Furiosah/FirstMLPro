import csv

class BCWDataset:

    def load(self, path='breast-cancer-wisconsin.data'):
        self.X = []
        self.y = []
        with open(path) as data:
            for row in csv.reader(data, delimiter=','):
                x = []
                skip = False
                for i in range(len(row)):
                    if (i == 0):
                        continue
                    if (row[i] == '?'):
                        skip = True
                        break
                    if (i == 10):
                        if (int(row[i]) == 2):
                            self.y.append(0.)
                        else:
                            self.y.append(1.)
                    else:
                        x.append(float(row[i]))
                if (skip):
                    continue
                self.X.append(x)


