# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
#os.chdir('/Users/ishitasharma/Desktop/ECS171-Project')

# Function to get labels of duplicate columns
def duplicate_columns(frame): #copied from github
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = frame[v].to_dict(orient="list")

        vs = dcols.values()
        ks = dcols.keys()
        lvs = len(vs)

        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]:
                    dups.append(ks[i])
                    break

    return dups

# Open training data set
X = pd.read_csv('data_train.csv')
X = pd.DataFrame(X)

# GET the labels with duplicate values
indices = duplicate_columns(X)

# DELETE the columns with duplicate values
for item in indices:
    del X[item]

# Save loss column in variable Y
Y = X['loss']
Y = pd.DataFrame(Y)
# Save ID column in variable idf
idf = X['id']
idf = pd.DataFrame(idf)

# Additional features
X['f527_528'] = X['f527']-X['f528']
X['f274_527'] = X['f274']-X['f527']
X['f274_528'] = X['f274']-X['f528']
X['Log 271'] = np.log(X['f271']+1)

# Processing of invalid data
for item in X:
    if X[item].dtype not in ['float64']:
        X[item] = X[item].astype(float)
    #if test_X[item].dtype != train_X[item].dtype:
    #    test_X[item] = test_X[item].astype(train_X[item].dtype)
    if not all(X[item].notnull()):
        X[item] = X[item].fillna(X[item].median())
    #if not all(test_X[item].notnull()):
    #    test_X[item] = test_X[item].fillna(train_X[item].median())

# Deleting loss and id columns
del X['loss']
del X['id']
del X['f534']
del X['f472']

Train_X = X[0:37500]
Train_Y = Y[0:37500]

Test_X = X[37500:50000]
Test_Y = Y[37500:50000]

Train_X.to_csv("Train_X.csv",index=False)
Train_Y.to_csv("Train_Y.csv",index=False)
Test_X.to_csv("Test_X.csv",index=False)
Test_Y.to_csv("Test_Y.csv",index=False)
idf.to_csv("idf.csv",index=False)
