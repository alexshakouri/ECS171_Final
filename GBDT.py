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

def loadDat():
	train = pd.read_csv('data_train.csv')

	test  = pd.read_csv('data_test.csv')
	featselect=['f2','f528_diff_f274','f332','f67','f25','f120','f766','f376','f39','f670','f228','f652','f415','f596','f406','f13','f355','f777','f2','f527_diff_f528','f528_diff_f274','f222','f332','f67','f25','f120','f766','f376','f670','f228','f652','f761','f4','f13','f386','f596','f9','f355','f406','f518','f328','f696','f674','f718'];
	featselect=list(set(featselect));	
	train['f527_diff_f528'] = train.f527 - train.f528
	train['f528_diff_f274'] = train.f528 - train.f274
	test['f527_diff_f528'] = train.f527 - train.f528
	test['f528_diff_f274'] = train.f528 - train.f274

	#use 4/5 for training and 1/5 for validation
	train_X = (train[featselect])[0:40000]
	train_Y = train.loss[0:40000]
	test_X  = (test[featselect])[40000:50000]
	test_Y = train.loss[40000:50000]
       
        #delete the loss and the id
        #del train_X['loss']
        #del train_X['id']
        #del test_X['id']
        #del test_X['loss']

	for x in featselect:
		if train_X[x].dtype not in ['float64']:
			train_X[x] = train_X[x].astype(float)
		if test_X[x].dtype != train_X[x].dtype:
			test_X[x] = test_X[x].astype(train_X[x].dtype)
		if not all(train_X[x].notnull()):
			train_X[x] = train_X[x].fillna(train_X[x].median())
		if not all(test_X[x].notnull()):
			test_X[x] = test_X[x].fillna(train_X[x].median())
	idf = test.id
	scale = StandardScaler()
	train_X = pd.DataFrame(scale.fit_transform(train_X))
	test_X = pd.DataFrame(scale.fit_transform(test_X))
	return train_X, train_Y, test_X, test_Y, idf

def train_model(train_X,train_Y):
#	indicepo = train_Y>0
#	indicene = train_Y==0
#	train_Y[indicepo]=1
#	train_Y[indicene]=0
	#classifier = linear_model.LogisticRegression(tol=0.00003, max_iter=10000, solver='saga')
	#classifier = GradientBoostingClassifier(n_estimators=65, learning_rate=0.3, max_depth=6, max_features='sqrt')
	classifier = GradientBoostingRegressor(n_estimators= 100, learning_rate=0.05, max_depth=10, max_features='sqrt')
	#train data here
        classifier.fit(train_X, train_Y)
	return classifier

def predictor(classifier,test_X,test_Y,idf):
#	indicepo = test_Y>0
#	indicene = test_Y==0
#	test_Y[indicepo]=1
#	test_Y[indicene]=0
	
        #cross validation
	loss= cross_val_predict(classifier, test_X, test_Y, cv=10)
        #classifier.predict(test_X)
        #make losssuch that if its below 1 than zero it
	indicene = loss<1
	loss[indicene]=0
	test_X['loss'] = loss
	test_X['id'] = idf
	#test_X = test_X.join(loss)
	test_X[['id','loss']].to_csv("pred.csv",index=False)
	#f1=f1_score(test_Y, prediction)
	np.savetxt("prediction.csv", loss, delimiter=" ")
	np.savetxt("test_y.csv", test_Y, delimiter=" ")
	#print ("accuracy", accuracy_score(test_Y, loss))
	#print ("f1 score", f1)
	print ("MAE", mean_absolute_error(test_Y, loss))


def main():
	""" Combine functions to make predictions """
	#load data
	train_X, train_Y, test_X, test_Y, idf= loadDat()
	#make default models
	clf_model = train_model(train_X, train_Y)
	#look at predictions
	predictor(clf_model,test_X,test_Y,idf)
	#save prediction
	#test[['id','loss']].to_csv("pred.csv",index=False)




# run everything when calling script from CLI
if __name__ == "__main__":
	main()



