import numpy as np
import time

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
        test_x_list = test_x_list[:-1]

        id_val = train_x_list.pop(0)
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
    test_y_file.close()
    #train_y = np.array(train_y)
    #test_y = np.array(test_y)
    print "Finished output set"



# I am using the unison shuffle provided from Stack Overflow
# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def testDataAccuracy(test_w, test_x, test_y, alpha):
    print '\n'
    prediction = 0
    for n in range(0, np.size(test_y)):
        if (np.sign(np.matmul(test_x[n], np.transpose(test_w))) == test_y[n]):
            prediction += 1
        else:
            prediction += -1

    test_accuracy = prediction / float(np.size(test_y)) * 100

    print 'Accuracy of test data:', test_accuracy, "with a step size of", alpha

def stochasticGradient():
    # declaring global variables
    global train_w
    global train_x
    global train_y
    global test_x
    global test_y

    # initializing variables
    N = 50000
    d = 770
    batch = 300
    alpha = 0.001
    epoch = 0 # number of iterations
    prediction = 0
    vary = True

    print train_x.shape
    print train_y.shape

    while (True):
        epoch += 1
        prediction = 0

        # Get batch of samples
        batch_train_x, batch_train_y = unison_shuffled_copies(train_x, train_y)
        batch_train_x = batch_train_x[0:batch]
        batch_train_y = batch_train_y[0:batch]

        # function to minimize
        yxw = batch_train_y * np.matmul(batch_train_x, np.transpose(train_w))
        min_function = np.sum( np.log( 1 + np.exp(-yxw) ) ) / batch # not sure

        # numerator of the gradient function
        resize_y = np.transpose([batch_train_y, ]*d)
        gradient_yx = resize_y*batch_train_x

        # computer gradient
        resize_yxw = np.transpose([yxw, ]*d)
        gradient = np.sum( np.divide(-gradient_yx, (1+np.exp(resize_yxw))), axis=0 ) / batch

        # Problem 3d
        # conditions to vary the step size
        if (vary):
            alpha = np.power(epoch, -0.5)

        #if (alpha < 0.000001 and vary):
        #    alpha = 0.000001

        print('Alpha: %.20f' % alpha)

        train_w -= alpha * gradient

        for b in range(0, batch):
            if (np.sign(np.matmul(batch_train_x[b], np.transpose(train_w))) == batch_train_y[b]):
                prediction += 1
            else:
                prediction += -1

        train_accuracy = prediction / float(batch) * 100

        norm_gradient = np.linalg.norm(gradient)

        print 'Epoch:', epoch, '|| Min Function:', min_function, '|| Accuracy:', train_accuracy, '|| Gradient:', np.sum(gradient), '|| Norm:', norm_gradient

        if (train_accuracy > 98.0 and norm_gradient < 5):
            break

    test_w = train_w # passing in w for test...easier to follow
    testDataAccuracy(test_w, test_x, test_y, alpha)


#createFiles()

# load the data as matrices
train_x = np.loadtxt('train_x.dat')
train_y = np.loadtxt('train_y.dat')
test_x = np.loadtxt('test_x.dat')
test_y = np.loadtxt('test_y.dat')


print train_x.shape
print test_x.shape

stochasticGradient()
