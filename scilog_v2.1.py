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
    train_X = pd.read_csv('Train_X.csv')
    train_Y = pd.read_csv('Train_Y.csv')
    test_X = pd.read_csv('Test_X.csv')
    test_Y = pd.read_csv('Test_Y.csv')
    #real_test_X = pd.read_csv('Real_Test_X.csv')
    #full_train_X = pd.read_csv('Full_Train_X.csv')
    #full_train_Y = pd.read_csv('Full_Train_Y.csv')
    feat = pd.read_csv('top_50.csv')
    feat = feat['feat'].tolist()
    train_X = train_X[feat]

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
