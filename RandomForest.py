"""
	Filename:       RandomForrest.py
	Author:         Cody
	Author:         Anil Kumar Behera
	Author:         Pitor
	Author:         Prateek

	Description:    Scikit Learn Random Forest wrapper
"""

from sklearn.ensemble import RandomForestClassifier
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from numpy import array
import numpy as np
import nltk
from collections import defaultdict
import itertools
from sklearn.metrics import accuracy_score
#import pdb

#classifier
class RandomForest:

	__slots__ = ('forest')

	def train( self, observations ,  k=5 ):
		'''
		An ensamble K-Fold Classifier 
		'''
		self.forest = []
		splitdata = np.array_split(observations, k)
		combos = list(reversed(list(itertools.combinations(splitdata, k-1))))
		accuracy_sum = 0
		for i in range(k):
			train = list(itertools.chain(*combos[i]))
			test = splitdata[i]
			if k==1:
				train = observations
				test = observations
			c = SklearnClassifier(RandomForestClassifier())
			#c = SklearnClassifier(cls)	
			c.train(train)
			accuracy_sum += nltk.classify.accuracy(c,test)
			self.forest.append(c)

		print('Accuracy on Train data(Using K fold)= ', accuracy_sum/k )

	def predict( self, observations, groundTruth = False, include_id = False):
		if groundTruth:
			if include_id:
				predict = np.array([[id,self.classify(fe),g] for fe,g,id in observations])
			else:
				predict = np.array([[self.classify(fe),g] for fe,g in observations])
		else:
			if include_id:
				predict = np.array([[id,self.classify(fe)] for fe,id  in observations])
			else:
				predict = np.array([[self.classify(fe)] for fe in observations])

		return predict
	
	def classify( self, fe):
		#print([c.classify(fe)  for c in self.forest])
		return nltk.FreqDist([c.classify(fe)  for c in self.forest]).max()

	def accuracy( self, observations ):
		output= np.array(self.predict(observations, groundTruth=True))
		return accuracy_score(output[:,1],output[:,0])

	def cm( self, data ):
		output= np.array(self.predict(data, groundTruth=True))
		print(nltk.ConfusionMatrix(list(output[:,1]),list(output[:,0])))


