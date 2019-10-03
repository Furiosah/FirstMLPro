import pandas as pd

# Clean the data
def shallowCleanData():
    dataset = pd.read_csv('winequality-red.csv', ';')
    index = 0
    for q in dataset['quality']:
        if q > 5:
            dataset.at[index, 'quality'] = 1
        else:
            dataset.at[index, 'quality'] = 0
        index = index + 1
    dataset.dropna()
    
    pos = dataset[dataset['quality']==1]
    neg = dataset[dataset['quality']==0]

    result = pd.concat([pos, neg])
    return result

# Calculate the correlation in 'winequality-red.csv'
def correlation():
    dataset = shallowCleanData()
    corr_dataframe = dataset.corr(method = 'pearson')
    corr_row = corr_dataframe.loc['quality']
    print(corr_row)
    return

correlation()