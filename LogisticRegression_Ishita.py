import numpy as np
from sklearn import linear_model
from time import time
import os
os.chdir('/Users/ishitasharma/Desktop/ECS171-Project')


trainData = np.load('ecs171train.npy')
testData = np.load('ecs171test.npy')

parsed_trainData = np.zeros([50000,771])
parsed_testData = np.zeros([55470,770])

for i in range(50000):
    temp1 = trainData[i+1].decode('utf-8')
    temp1 = temp1.replace("NA","0")
    parsed_trainData[i] = [float(s1) for s1 in temp1.split(',')]

    
for j in range(55470):
    temp2 = testData[j+1].decode('utf-8')
    temp2 = temp2.replace("NA","0")
    parsed_testData[j] = [float(s2) for s2 in temp2.split(',')]


train_X = np.array(parsed_trainData[0:50000,1:770])
train_y = np.array(parsed_trainData[0:50000,770])
test_X = np.array(parsed_testData[0:55470,1:770])

classifier = linear_model.LogisticRegression()
t0 = time()
classifier.fit(train_X, train_y)
print ("training time:", round(time()-t0, 3), "s")

t1 = time()
prediction = classifier.predict(test_X)
print ("testing time:", round(time()-t1, 3), "s")
