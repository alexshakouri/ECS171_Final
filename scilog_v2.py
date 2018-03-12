import os
import pandas as pd
import numpy as np
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
    '''
    train = pd.read_csv('data_train.csv')
    test  = pd.read_csv('data_test.csv')
    #featselect=['f2','f528_diff_f274','f332','f67','f25','f120','f766','f376','f39','f670','f228','f652','f415','f596','f406','f13','f355','f777','f2','f527_diff_f528','f528_diff_f274','f222','f332','f67','f25','f120','f766','f376','f670','f228','f652','f761','f4','f13','f386','f596','f9','f355','f406','f518','f328','f696','f674','f718'];
    #featselect=list(set(featselect));
    train['f527_diff_f528'] = train.f527 - train.f528
    train['f528_diff_f274'] = train.f528 - train.f274
    test['f527_diff_f528'] = train.f527 - train.f528
    test['f528_diff_f274'] = train.f528 - train.f274

    train_X = train#[featselect]
    train_Y = train.loss
    '''
    train_X = pd.read_csv('Train_X.csv')
    train_Y = pd.read_csv('Train_Y.csv')
    test_X = pd.read_csv('Test_X.csv')
    test_Y = pd.read_csv('Test_Y.csv')
    '''
    test_X  = train[35000:49999]#[featselect]
    test_Y  = train.loss[35000:49999]

    del train_X['loss']
    del train_X['id']
    del test_X['loss']
    del test_X['id']
    realtest_X=test;

    for x in train_X:
        if train_X[x].dtype not in ['float64']:
            train_X[x] = train_X[x].astype(float)
        if realtest_X[x].dtype not in ['float64']:
            realtest_X[x] = realtest_X[x].astype(float)
        if test_X[x].dtype != train_X[x].dtype:
            test_X[x] = test_X[x].astype(train_X[x].dtype)
        if realtest_X[x].dtype != train_X[x].dtype:
            realtest_X[x] = realtest_X[x].astype(train_X[x].dtype)
        if not all(train_X[x].notnull()):
            train_X[x] = train_X[x].fillna(train_X[x].median())
        if not all(test_X[x].notnull()):
            test_X[x] = test_X[x].fillna(train_X[x].median())
        if not all(realtest_X[x].notnull()):
            realtest_X[x] = realtest_X[x].fillna(train_X[x].median())
    '''
    idf = pd.read_csv('idf.csv')
    #idf = realtest_X.id
    #del realtest_X['id']
    scale = StandardScaler()
    train_X = pd.DataFrame(scale.fit_transform(train_X))
    test_X = pd.DataFrame(scale.fit_transform(test_X))
    #realtest_X = pd.DataFrame(scale.fit_transform(realtest_X))
    #return train_X, train_Y, test_X, test_Y, realtest_X,idf
    return train_X, train_Y, test_X, test_Y, idf

def train_model(train_X,train_Y):
#	indicepo = train_Y>0
#	indicene = train_Y==0
#	train_Y[indicepo]=1
#	train_Y[indicene]=0
    #classifier = linear_model.LogisticRegression(tol=0.00003, max_iter=10000, solver='saga')
    #classifier = GradientBoostingClassifier(n_estimators=65, learning_rate=0.3, max_depth=6, max_features='sqrt')
    classifier = GradientBoostingRegressor(n_estimators=160, learning_rate=0.109, max_depth=5, max_features='sqrt')
    classifier.fit(train_X, train_Y)
    return classifier

def predictor(classifier, test_X, test_Y, idf):
    cv = KFold(n_splits=10)
    loss = cross_val_predict(classifier, test_X, test_Y, cv=cv)
    #loss=classifier.predict(realtest_X)
    indicene = loss < 1
    loss[indicene] = 0
    test_X['loss'] = loss
    test_X['id'] = idf
    #test_X = test_X.join(loss)
    test_X[['id','loss']].to_csv("predictions.csv",index=False)
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
    #save prediction
    #test[['id','loss']].to_csv("pred.csv",index=False)




# run everything when calling script from CLI
if __name__ == "__main__":
    main()
