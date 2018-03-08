from sklearn import linear_model
import numpy as np
from time import time
import os

# declaring global variables
global train_w
global train_x
global train_y
global test_x
global test_y
np.set_printoptions(threshold='nan')

nan = float("NaN") # defining nan

# initializing weights with uniform distribution
# from -0.01 to +0.01
train_w = np.random.uniform(-0.01, 0.01, 770)

def createFiles():
    global train_x
    global train_y
    global test_x
    global test_y

    train_data = np.load('ecs171train.npy') #data[0,] is the column headers
    test_data = np.load('ecs171test.npy')
    size_train = train_data.size # 50001
    size_test = test_data.size
    train_y = []
    test_y = []
    id_values_train = []
    id_values_test = []

    #train_x = np.empty((0,770), int)
    #test_x = np.empty((0,770), int)

    # open the files
    train_x_file = open("train_x.dat","a")
    test_x_file = open("test_x.dat","a")
    train_y_file = open("train_y.dat","a")
    #test_y_file = open("test_y.dat","a")
    train_id_file = open("train_id.dat","a")
    test_id_file = open("test_id.dat","a")

    # the feature set

    for i in range(1,size_train):
        print i

        train_x_list = train_data[i,].split(",")
        train_x_list = train_x_list[:-1]

        id_val = train_x_list.pop(0)
        id_values_train.append(int(id_val))

        train_x_list = [float(a) if a != 'NA' else 0 for a in train_x_list]
        for item in train_x_list:
            train_x_file.write("%s " % item)
        train_x_file.write("\n")
        #train_x = np.append(train_x, np.array([train_x_list]), axis=0)

    for item in id_values_train:
        train_id_file.write("%s " % item)
        train_id_file.write("\n")

    for j in range(1,size_test):
        print j

        test_x_list = test_data[j-1,].split(",") # i-1 b/c test set doesn't have column headers
        #test_x_list = test_x_list[:-1]

        id_val = test_x_list.pop(0)
        id_values_test.append(int(id_val))

        test_x_list = [float(b) if b != 'NA' else 0 for b in test_x_list]
        for item in test_x_list:
            test_x_file.write("%s " % item)
        test_x_file.write("\n")
        #test_x = np.append(test_x, np.array([test_x_list]), axis=0)

    for item in id_values_test:
        test_id_file.write("%s " % item)
        test_id_file.write("\n")

    # close the files
    train_x_file.close()
    test_x_file.close()
    train_id_file.close()
    test_id_file.close()

    print "Finished feature set"

    # the loss column values
    for i in range(1,size_train):
        train_y_list = train_data[i,].split(",")
        train_y.append(int(train_y_list[770]))

    for item in train_y:
        train_y_file.write("%s " % item)
        train_y_file.write("\n")

    '''
    for j in range(1,size_test):
        test_y_list = test_data[i-1,].split(",") # i-1 b/c test set doesn't have column headers
        test_y.append(int(test_y_list[769])) # no loss col?

    for item in test_y: #DO WE EVEN HAVE A TEST SET LOSS COL?
        test_y_file.write("%s " % item)
        test_y_file.write("\n")
    '''
    train_y_file.close()
    #test_y_file.close()
    #train_y = np.array(train_y)
    #test_y = np.array(test_y)
    print "Finished output set"


#createFiles()

# load the data as matrices
train_x = np.loadtxt('train_x.dat')
train_y = np.loadtxt('train_y.dat')
test_x = np.loadtxt('test_x.dat')
#test_y = np.loadtxt('test_y.dat')

#training
sub_train_x = train_x[0:40000,]
sub_test_x = train_x[40000:50000,]
sub_train_y = train_y[0:40000,]
sub_test_y = train_y[40000:50000,]

print train_x.shape
print train_y.shape
print test_x.shape

#classifier = linear_model.LogisticRegression(max_iter=500)
classifier = linear_model.SGDClassifier(loss='log', max_iter=1000)

t0 = time()
train_accuracy = classifier.fit(sub_train_x, sub_train_y).score(sub_test_x, sub_test_y)
print ("training time:", round(time()-t0, 3), "s")
print ("Training Accuracy", train_accuracy)

t1 = time()
prediction = classifier.predict(test_x)
print ("testing time:", round(time()-t1, 3), "s")
#print ("Prediction:", prediction)
prediction_file = open("prediction.dat","a")
for item in prediction:
    prediction_file.write("%s " % item)
    prediction_file.write("\n")
prediction_file.close()
