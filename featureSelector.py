# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from time import time
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

os.chdir('/Users/ishitasharma/Desktop/ECS171-Project')

X = pd.read_csv('data_train.csv')
X = pd.DataFrame(X)


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

Index = duplicate_columns(X)

for item in Index:
    del X[item]

    
Y= X['loss']

X['f527_528'] = X['f527']-X['f528']
X['f274_527'] = X['f274']-X['f527']
X['f274_528'] = X['f274']-X['f528']
X['Log 271'] = np.log(X['f271']+1)


del X['loss']
del X['id']

train_X = X[0:37500]
train_Y = Y[0:37500]

test_X = X[37500:50000]
test_Y = Y[37500:50000]


np.savetxt('Train_X.csv',train_X, delimiter=',')
np.savetxt('Train_Y.csv',train_Y, delimiter=',')
np.savetxt('Test_X.csv',test_X, delimiter=',')
np.savetxt('Test_Y.csv',train_Y, delimiter=',')
