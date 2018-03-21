import os
import pandas as pd
import numpy as np
import time
from itertools import chain
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

def loadDat():

    #feature select done here before!!!!
    train_X = pd.read_csv('Train_X.csv')
    train_Y = pd.read_csv('Train_Y.csv')
    test_X = pd.read_csv('Test_X.csv')
    test_Y = pd.read_csv('Test_Y.csv')

    #UPLOAD TO KAGGLE
    ##train_X = pd.read_csv('Full_Train_X.csv')
    ##train_Y = pd.read_csv('Full_Train_Y.csv')
    ##test_X = pd.read_csv('Real_Test_X.csv')
    ##test_Y = pd.read_csv('Test_Y.csv')

    #choose only a certain amount of features that have a high correlation to loss
    #TOP 10 features
    #featselect = ['f281', 'f282', 'f400', 'f471', 'f536', 'f612', 'f675', 'f527_528', 'f274_527', 'f274_528'];

     #ADD FEATURES DIFFERENCES
    train_X['f674_294'] = train_X['f674'] - train_X['f294'];
    train_X['f755_294'] = train_X['f755'] - train_X['f294'];
    train_X['f674_319'] = train_X['f674'] - train_X['f319'];
    train_X['f755_674'] = train_X['f755'] - train_X['f674'];
    train_X['f274_f528_div'] = (train_X['f274']-train_X['f528']) / (train_X['f528']-train_X['f527']+1)
    train_X['f271_div'] = (train_X['f271']) / (train_X['f528']-train_X['f527']+1)

    test_X['f674_294'] = test_X['f674'] - test_X['f294'];
    test_X['f755_294'] = test_X['f755'] - test_X['f294'];
    test_X['f674_319'] = test_X['f674'] - test_X['f319'];
    test_X['f755_674'] = test_X['f755'] - test_X['f674'];
    test_X['f274_f528_div'] = (test_X['f274']-test_X['f528']) / (test_X['f528']-test_X['f527']+1)
    test_X['f271_div'] = (test_X['f271']) / (test_X['f528']-test_X['f527']+1)

    #featselect = ['f2', 'f4', 'f5', 'f7', 'f8', 'f27', 'f67', 'f68', 'f140', 'f219', 'f220', 'f221', 'f229', 'f230', 'f271',
    #'f274', 'f332', 'f336', 'f376', 'f515',    'f523',    'f526', 'f527', 'f528', 'f532', 'f533',    'f536',    'f556',    'f592',
    #'f596', 'f608', 'f609',    'f612',    'f620',    'f621', 'f630', 'f647', 'f670', 'f767',    'f775',    'f776', 'f777', 'f778',
    #'f527_528',    'f274_527',    'f274_528',    'Log 271', 'f274_f528_div', 'f271_div', 'f674_294','f755_294', 'f674_319',
    #'f755_674']

    #466, 529] , [271, 520]

    train_X['f466_529'] = train_X['f466'] + train_X['f529'];
    train_X['f664_759'] = train_X['f664'] + train_X['f759'];
    train_X['Log 282'] = np.log(train_X['f282']+1)

    test_X['f466_529'] = test_X['f466'] + test_X['f529'];
    test_X['f664_759'] = test_X['f664'] + test_X['f759'];
    test_X['Log 282'] = np.log(test_X['f282']+1)

    #logsP = ['f673','f282']
    # log f282
    featselect = ['f2', 'f4', 'f7', 'f8', 'f13', 'f27', 'f67', 'f68', 'f140',
    'f219', 'f220', 'f221', 'f229', 'f230', 'f271', 'f274', 'f332', 'f336', 'f376', 'f515', 'f523', 'f526', 'f527',
    'f528', 'f532', 'f533', 'f592', 'f596', 'f598', 'f608', 'f609', 'f612', 'f620', 'f621', 'f630', 'f647', 'f652',
    'f670', 'f767', 'f775', 'f778', 'f527_528', 'f274_527', 'f274_528', 'Log 271', 'f274_f528_div',
    'f271_div','f755_294', 'f674_319', 'f755_674', 'f466_529', 'f664_759', 'Log 282']


    print(np.shape(train_X))
    train_X = train_X[featselect];
    test_X = test_X[featselect];
    print (np.shape(test_X))

    idf = pd.read_csv('idf.csv')
    scale = StandardScaler()
    train_X = pd.DataFrame(scale.fit_transform(train_X))
    test_X = pd.DataFrame(scale.fit_transform(test_X))
    #realtest_X = pd.DataFrame(scale.fit_transform(realtest_X))
    #return train_X, train_Y, test_X, test_Y, realtest_X,idf
    return train_X, train_Y, test_X, test_Y, idf

def train_model(train_X,train_Y):
    #classifier = linear_model.LogisticRegression(tol=0.00003, max_iter=10000, solver='saga')

    classifier = GradientBoostingClassifier(n_estimators=65, learning_rate=0.3, max_depth=6, max_features='sqrt')
    classifier.fit(train_X, train_Y)
    print "Finished Class"
    class_loss_y = classifier.predict(train_X)
    print "Class Loss Y:", np.shape(class_loss_y)
    class_loss_x = np.transpose([class_loss_y, ] * 53)
    class_loss_x = pd.DataFrame(class_loss_x)
    print "Class Loss X:", np.shape(class_loss_x)

    class_loss_y = pd.DataFrame(class_loss_y)

    #print classifier.feature_importances_
    class_X = pd.DataFrame.multiply(train_X, class_loss_x)
    print "Class X:", np.shape(class_X)
    class_Y = pd.DataFrame.multiply(train_Y, class_loss_y)
    print "Class Y:", np.shape(class_Y)


    '''COPY THIS PART'''
    for item in class_X:
        if class_X[item].dtype not in ['float64']:
            class_X[item] = class_X[item].astype(float)
        if not all(class_X[item].notnull()):
            class_X[item] = class_X[item].fillna(class_X[item].median())
    for item in class_Y:
        if class_Y[item].dtype not in ['float64']:
            class_Y[item] = class_Y[item].astype(float)
        if not all(class_Y[item].notnull()):
            class_Y[item] = class_Y[item].fillna(class_Y[item].median())
    '''STOP COPYING'''

    regressor = GradientBoostingRegressor(n_estimators=900, learning_rate=0.0075, max_depth=10 , max_features='sqrt', min_samples_leaf=20, max_leaf_nodes=30)
    regressor.fit(class_X, class_Y)
    print "Finished GBR"
    return regressor

def predictor(classifier, test_X, test_Y, idf):
    loss = cross_val_predict(classifier, test_X, test_Y, cv=5)
    "Finished CV"
    #loss=classifier.predict(test_X)
    indicene = loss < 1
    loss[indicene] = 0
    test_X['loss'] = loss
    test_X['id'] = idf
    #test_X = test_X.join(loss)
    #test_X[['id','loss']].to_csv("predictions.csv",index=False)
    print "MAE", mean_absolute_error(test_Y, loss)


def main():
    """ Combine functions to make predictions """
    #load data
    #train_X, train_Y, test_X, test_Y, realtest_X,idf= loadDat()
    train_X, train_Y, test_X, test_Y, idf = loadDat()
    #make default models
    clf_model = train_model(train_X, train_Y)
    #look at predictions
    predictor(clf_model, train_X, train_Y, idf)
    #KAGGLE
   # predictor(clf_model, test_X, test_Y, idf)

    #save prediction
    #test[['id','loss']].to_csv("pred.csv",index=False)




# run everything when calling script from CLI
if __name__ == "__main__":
    main()
