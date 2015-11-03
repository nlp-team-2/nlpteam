"""
	Filename:       RandomForrestReg.py
	Author:         Cody
	Author:         Anil Kumar Behera
	Author:         Pitor
	Author:         Prateek

	Description:    Scikit Learn Random Forrest Regressor/ No Wrapper exist in nltk as classifier
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from numpy import array
import numpy as np
import nltk
from collections import defaultdict
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
#classifier
class RandomForestReg:

	__slots__ = ('forest','keylist')
	def train( self, observations, keylist , k=5 ):
			
		self.keylist = keylist		
		data = np.array([ [fe[key]  for key in keylist]+[g]  for fe,g in observations])
		self.forest = []
		splitdata = np.array_split(data, k)
		combos = list(reversed(list(itertools.combinations(splitdata, k-1))))
		MSE = 0
		for i in range(k):
			train = np.array(list(itertools.chain(*combos[i])))
			test = np.array(splitdata[i])
			train_X=train[:,:-1]
			train_y=train[:,-1]
			test_X=test[:,:-1]
			test_y=test[:,-1]
			#c = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
			c = RandomForestRegressor()
			c.fit(train_X,train_y)
			MSE += mean_squared_error(test_y,c.predict(test_X))
			self.forest.append(c)

		print('MSE Train data(Using K fold)= ', MSE/k )
		#self.forest = SklearnClassifier(SVC())
                #self.forest = nltk.NaiveBayesClassifier.train(observations)
                #self.forest.show_most_informative_features()
                #self.forest.train(observations)

	def predict( self, observations, groundTruth = False, include_id=False):
		if groundTruth:
			if include_id:
				data = np.array([[id]+[fe[key]  for key in self.keylist ]+[g]  for fe,g,id in observations])
				X = data[:,1:-1]
				y = data[:,-1]
				ID = data[:,0]
			else:
				data = np.array([[fe[key]  for key in self.keylist ]+[g]  for fe,g in observations])
				X = data[:,:-1]
				y = data[:,-1]
		else:
			X= np.array([[fe[key]  for key in self.keylist ]  for fe in observations])
		
		xx = np.zeros(len(observations))
		for c in self.forest:
			p = c.predict(X)
			xx =xx+p
		predict = np.round(xx/len(self.forest))

		if include_id:
			predict = np.vstack((ID, predict))
		if groundTruth:
			predict = np.vstack((predict,y)).T

		return predict
	

	def accuracy( self, observations ):
		predict = self.predict(observations, groundTruth = True)
		return accuracy_score(predict[:,1],predict[:,0])

	def MSE( self, observations ):
		predict = self.predict(observations, groundTruth = True)
		return mean_squared_error(predict[:,1],predict[:,0])

	def r2( self, observations):
		predict = self.predict(observations, groundTruth = True)
		return r2_score(predict[:,1],predict[:,0])
