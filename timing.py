import time
import BCWDataset, WQDataset
import LDA, LogisticRegression
import KFoldCrossValidator

bcwd = BCWDataset.BCWDataset()
bcwd.load()
wqd = WQDataset.WQDataset()
wqd.load()

print("LDA, BCW")
print(KFoldCrossValidator.validate(LDA.LDA(), 5, bcwd.X, bcwd.y))
print("LogReg, BCW")
print(KFoldCrossValidator.validate(LogisticRegression.LogisticRegression(flr=0.6, slr=0.1, num_it=100), 5, bcwd.X, bcwd.y))
print("LDA, WQ")
print(KFoldCrossValidator.validate(LDA.LDA(), 5, wqd.X, wqd.y))
print("LogReg, WQ")
print(KFoldCrossValidator.validate(LogisticRegression.LogisticRegression(flr=0.6, slr=0.1, num_it=100), 5, wqd.X, wqd.y))
