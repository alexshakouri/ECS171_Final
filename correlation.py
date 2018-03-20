import pandas as pd
import numpy as np
import sys
import csv
from sklearn.preprocessing import StandardScaler

epsilon = sys.float_info.epsilon

scale = StandardScaler()
full_train_X = pd.read_csv('Full_Train_X.csv')
full_train_X = pd.DataFrame(full_train_X)

rows, cols = full_train_X.shape
flds = list(full_train_X.columns)

corr = full_train_X.corr().values

count = 0
my_list = []
for i in range(cols):
    for j in range(i+1, cols):
        if np.absolute(corr[i,j]) > 0.99999 and np.absolute(corr[i,j]) < (1-epsilon):
            #print flds[i], ' ', flds[j], ' ', corr[i,j]
            values = [flds[i], flds[j]]
            my_list.append(values)
            count += 1
#print "Number above threshold:", count
print my_list

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(my_list)
