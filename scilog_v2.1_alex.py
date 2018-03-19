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

    #feature select done here before!!!!
    train_X = pd.read_csv('Train_X.csv')
    train_Y = pd.read_csv('Train_Y.csv')
    test_X = pd.read_csv('Test_X.csv')
    test_Y = pd.read_csv('Test_Y.csv')
    
    #choose only a certain amount of features that have a high correlation to loss
    #TOP 10 features
    #featselect = ['f281', 'f282', 'f400', 'f471', 'f536', 'f612', 'f675', 'f527_528', 'f274_527', 'f274_528'];

    #f220, f316, f589, f592, f523
    featselect =['f140','f219','f220', 'f221', 'f251', 'f281', 'f282', 'f290', 'f291', 'f292', 'f294', 'f314', 'f315','f316',  'f319', 'f322',  'f323', 'f335', 'f400', 'f404', 'f405', 'f414', 'f415', 'f421', 'f428', 'f471', 'f515', 'f523', 'f526', 'f533', 'f536', 'f556', 'f589', 'f591', 'f592', 'f609', 'f612', 'f620', 'f621', 'f674', 'f675', 'f676', 'f699', 'f755', 'f765', 'f766', 'f767', 'f775',  'f776', 'f527_528', 'f274_527', 'f274_528', 'Log 271'];

#featselect=list(set(featselect));  
    
    print(np.shape(train_X))
    train_X = train_X[featselect];
    test_X = test_X[featselect];
    print(np.shape(train_X))
   
    #ADD FEATURES DIFFERENCES
#    train_X['f319_294'] = train_X['f319'] - train_X['f294'];
    train_X['f674_294'] = train_X['f674'] - train_X['f294'];
    train_X['f755_294'] = train_X['f755'] - train_X['f294'];
#    train_X['f612_609'] = train_X['f612'] - train_X['f609'];   
    train_X['f755_699'] = train_X['f755'] - train_X['f699'];
    print(np.shape(train_X));

#    test_X['f319_294'] = test_X['f319'] - test_X['f294'];
    test_X['f674_294'] = test_X['f674'] - test_X['f294'];
    test_X['f755_294'] = test_X['f755'] - test_X['f294'];
#    test_X['f612_609'] = test_X['f612'] - test_X['f609'];   
    test_X['f755_699'] = test_X['f755'] - test_X['f699'];
 
 
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
    scale = StandardScaler()
    train_X = pd.DataFrame(scale.fit_transform(train_X))
    test_X = pd.DataFrame(scale.fit_transform(test_X))


    #realtest_X = pd.DataFrame(scale.fit_transform(realtest_X))
    #return train_X, train_Y, test_X, test_Y, realtest_X,idf
    return train_X, train_Y, test_X, test_Y, idf

def train_model(train_X,train_Y):
    #classifier = linear_model.LogisticRegression(tol=0.00003, max_iter=10000, solver='saga')
    #classifier = GradientBoostingClassifier(n_estimators=65, learning_rate=0.3, max_depth=6, max_features='sqrt')
    classifier = GradientBoostingRegressor(n_estimators=700, learning_rate=0.0075, max_depth=5, max_features='sqrt', min_samples_leaf=20, max_leaf_nodes=30)
    classifier.fit(train_X, train_Y)
    return classifier

def predictor(classifier, test_X, test_Y, idf):
    loss = cross_val_predict(classifier, test_X, test_Y, cv=5)
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
