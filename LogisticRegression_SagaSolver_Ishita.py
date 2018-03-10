import numpy as np
from sklearn import linear_model
from time import time
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

os.chdir('/Users/ishitasharma/Desktop/ECS171-Project')

trainData = np.load('ecs171train.npy')

parsed_trainData = np.zeros([50000, 771])


for i in range(50000):
    temp1 = trainData[i + 1].decode('utf-8')
    temp1 = temp1.replace("NA", "0")
    parsed_trainData[i] = [float(s1) for s1 in temp1.split(',')]


X = np.array(parsed_trainData[0:50000, 1:770])
y = np.array(parsed_trainData[0:50000, 770])

train_X = X[0:37501]
train_y = y[0:37501]

test_X = X[37501:50000]
test_y = y[37501:50000]

classifier = linear_model.LogisticRegression(tol=0.003, max_iter=750, solver='saga')
t0 = time()
classifier.fit(train_X, train_y)
print ("training time:", round(time() - t0, 3), "s")

t1 = time()
prediction = classifier.predict(test_X)
print ("testing time:", round(time() - t1, 3), "s")

print "accuracy", accuracy_score(test_y, prediction)
print "MAE", mean_absolute_error(test_y, prediction)


