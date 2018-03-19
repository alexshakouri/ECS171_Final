import os
os.chdir('/Users/ishitasharma/Desktop/ECS171-Project/Data')
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
    train_X = pd.read_csv('Train_X.csv')
    train_Y = pd.read_csv('Train_Y.csv')
    test_X = pd.read_csv('Test_X.csv')
    test_Y = pd.read_csv('Test_Y.csv')
    #real_test_X = pd.read_csv('Real_Test_X.csv')
    #full_train_X = pd.read_csv('Full_Train_X.csv')
    #full_train_Y = pd.read_csv('Full_Train_Y.csv')
    feat = ['f140',	'f219',	'f220',	'f221',	'f251',	'f281',	'f282',	'f290',	'f291',	'f292',	'f294',	'f314',	'f315',	'f316',		'f322',	'f323',	'f335',	'f400',
    'f404',	'f405',	'f414',	'f415',	'f421', 'f428',	'f471',	'f515',	'f523',	'f526',	'f533',	'f536',	'f556',	'f589',	'f591',	'f592',	'f609',	'f612',	'f620',	'f621',	
    'f675',	'f676',	'f699',	'f765',	'f766',	'f767',	'f775',	'f776', 'f527_528',	'f274_527',	'f274_528',	'Log 271', 'f674_319', 'f755_294','f674_294']

    #'f319_294','f319', 'f674','f755','f612_609','f755_699','f755_674','f755_319',, 

    train_X = train_X[feat]
    test_X = test_X[feat]

    idf = pd.read_csv('idf.csv')
    scale = StandardScaler()
    train_X = pd.DataFrame(scale.fit_transform(train_X))
    test_X = pd.DataFrame(scale.fit_transform(test_X))
    #real_test_X = pd.DataFrame(scale.fit_transform(real_test_X))
    #full_train_X = pd.DataFrame(scale.fit_transform(full_train_X))
    #return train_X, train_Y, test_X, test_Y, realtest_X,idf
    return train_X, train_Y, test_X, test_Y, idf

def train_model(train_X,train_Y):
#	indicepo = train_Y>0
#	indicene = train_Y==0
#	train_Y[indicepo]=1
#	train_Y[indicene]=0
    #classifier = linear_model.LogisticRegression(tol=0.00003, max_iter=10000, solver='saga')
    #classifier = GradientBoostingClassifier(n_estimators=65, learning_rate=0.3, max_depth=6, max_features='sqrt')
    classifier = GradientBoostingRegressor(loss='ls', n_estimators=500, learning_rate=0.0075, max_depth=10, max_features='sqrt', random_state=9753, verbose=0, min_samples_leaf=20, max_leaf_nodes=30, min_samples_split=20)
    classifier.fit(train_X, train_Y)
    return classifier

def predictor(classifier, test_X, test_Y, idf):
    cv = KFold(n_splits=5)
    loss = cross_val_predict(classifier, test_X, test_Y, cv=cv)
    #loss = classifier.predict(real_test_X)
    indicene = loss < 1
    loss[indicene] = 0
    test_X['loss'] = loss
    test_X['id'] = idf
    test_X[['id','loss']].to_csv("predictions.csv",index=False)
    print "MAE Loss Cross", mean_absolute_error(test_Y, loss)



def main():
    """ Combine functions to make predictions """
    #load data
    train_X, train_Y, test_X, test_Y, idf = loadDat()
    #make default models
    clf_model = train_model(train_X, train_Y)
    #look at predictions
    predictor(clf_model, test_X, test_Y, idf)



# run everything when calling script from CLI
if __name__ == "__main__":
    main()
