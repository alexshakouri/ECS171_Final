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
    #feat = ['f8','f140',	'f219',	'f220',	'f221',	'f251',	'f281',	'f282',	'f290',	'f291',	'f292',	'f294',	'f314',	'f315',	'f316',		'f322',	'f323',	'f335',	'f400',
    #'f404',	'f405',	'f414',	'f415',	'f421', 'f428',	'f471',	'f515',	'f523',	'f526',	'f533',	'f536',	'f556',	'f589',	'f591',	'f592',	'f609',	'f612',	'f620',	'f621',
    #'f675',	'f676',	'f699',	'f765',	'f766',	'f767',	'f775',	'f776', 'f527_528',	'f274_527',	'f274_528',	'Log 271']

    #'f319_294','f319', 'f674','f755','f612_609','f755_699','f755_674','f755_319',,

    feat = ['f7', 'f8', 'f36', 'f58', 'f74', 'f86', 'f98', 'f105', 'f117', 'f225', 'f236', 'f247', 'f254', 'f265', 'f274', 'f389', 'f406', 'f452', 'f478',
    'f483', 'f493', 'f503', 'f527', 'f532', 'f538', 'f548', 'f558', 'f568', 'f577', 'f729', 'f608', 'f543', 'f770', 'f452',
    'f494', 'f504', 'f484', 'f539', 'f549', 'f559', 'f578', 'f528', 'f625', 'f768', 'f453', 'f569', 'f644',
    'f140',	'f219',	'f220',	'f221',	'f251',	'f281',	'f282',	'f290',	'f291',	'f292',	'f294',	'f314',	'f315',	'f316',		'f322',	'f323',	'f335',	'f400',
    'f404',	'f405',	'f414',	'f415',	'f421', 'f428',	'f471',	'f515',	'f523',	'f526',	'f533',	'f536',	'f556',	'f589',	'f591',	'f592',	'f609',	'f612',	'f620',	'f621',
    'f675',	'f676',	'f699',	'f765',	'f766',	'f767',	'f775',	'f776', 'Log 271']

    train_X = train_X[feat]
    test_X = test_X[feat]

    feat = pd.read_csv('time_depend.csv')
    feat = feat.values.tolist()

    train_X['f7_f608'] = train_X['f7']-train_X['f608']
    train_X['f8_f532'] = train_X['f8']-train_X['f532']
    train_X['f8_f543'] = train_X['f8']-train_X['f543']
    train_X['f36_f74'] = train_X['f36']-train_X['f74']
    train_X['f36_f729'] = train_X['f36']-train_X['f729']
    train_X['f36_f770'] = train_X['f36']-train_X['f770']
    train_X['f58_f644'] = train_X['f58']-train_X['f644']
    train_X['f74_f770'] = train_X['f74']-train_X['f770']
    #train_X['f86_f452'] = train_X['f86']-train_X['f452']

    train_X['f86_f453'] = train_X['f86']-train_X['f453']
    train_X['f98_f493'] = train_X['f98']-train_X['f493']
    train_X['f98_f494'] = train_X['f98']-train_X['f494']
    train_X['f105_f503'] = train_X['f105']-train_X['f503']
    train_X['f105_f504'] = train_X['f105']-train_X['f504']
    train_X['f117_f483'] = train_X['f117']-train_X['f483']
    train_X['f117_f484'] = train_X['f117']-train_X['f484']
    train_X['f225_f538'] = train_X['f225']-train_X['f538']
    train_X['f225_f539'] = train_X['f225']-train_X['f539']
    train_X['f236_f548'] = train_X['f236']-train_X['f548']
    train_X['f236_f549'] = train_X['f236']-train_X['f549']
    train_X['f247_f558'] = train_X['f247']-train_X['f558']
    train_X['f247_f559'] = train_X['f247']-train_X['f559']
    train_X['f254_f568'] = train_X['f254']-train_X['f568']
    train_X['f254_f569'] = train_X['f254']-train_X['f569']
    train_X['f265_f577'] = train_X['f265']-train_X['f577']
    train_X['f265_f578'] = train_X['f265']-train_X['f578']
    train_X['f274_f527'] = train_X['f274']-train_X['f527']
    train_X['f274_f528'] = train_X['f274']-train_X['f528']
    train_X['f389_f625'] = train_X['f389']-train_X['f625']
    train_X['f406_f768'] = train_X['f406']-train_X['f768']
    #train_X['f452_f453'] = train_X['f452']-train_X['f453']
    train_X['f478_f608'] = train_X['f478']-train_X['f608']

    train_X['f483_f484'] = train_X['f483']-train_X['f484']
    train_X['f493_f494'] = train_X['f493']-train_X['f494']
    train_X['f503_f504'] = train_X['f503']-train_X['f504']
    train_X['f527_f528'] = train_X['f527']-train_X['f528']

    train_X['f532_f543'] = train_X['f532']-train_X['f543']
    train_X['f538_f539'] = train_X['f538']-train_X['f539']
    train_X['f548_f549'] = train_X['f548']-train_X['f549']
    train_X['f558_f559'] = train_X['f558']-train_X['f559']
    train_X['f568_f569'] = train_X['f568']-train_X['f569']
    train_X['f577_f578'] = train_X['f577']-train_X['f578']
    train_X['f729_f770'] = train_X['f729']-train_X['f770']
    '''
    count = 0
    for item in feat:
        count += 1
        print count
        s = ""
        s = item[0]+"_"+item[1]
        train_X[s] = train_X[item[0]] - train_X[item[1]]
        test_X[s] = test_X[item[0]] - test_X[item[1]]
    #print train_X
    '''


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
    print "MAE Loss:", mean_absolute_error(test_Y, loss)



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
