"""
	Filename:       RandomForrest.py
	Author:         Cody
	Author:         Anil Kumar Behera
	Author:         Pitor
	Author:         Prateek

	Description:    Scikit Learn Random Forrest wrapper
"""

from sklearn.ensemble import RandomForestClassifier
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from numpy import array
import numpy as np
import nltk
from collections import defaultdict
import itertools

#classifier
class RandomForest:

	__slots__ = ('forest')

	def train( self, observations , cls = RandomForestClassifier() , k=5 ):
		self.forest = []
		splitdata = np.array_split(observations, k)
		combos = list(reversed(list(itertools.combinations(splitdata, k-1))))
		accuracy_sum = 0
		for i in range(k):
			train = list(itertools.chain(*combos[i]))
			test = splitdata[i]
			c = SklearnClassifier(cls)
			c.train(train)
			accuracy_sum += nltk.classify.accuracy(c,test)
			self.forest.append(c)

		print('Accuracy on Train data(Using K fold)= ', accuracy_sum/k )
		#self.forest = SklearnClassifier(SVC())
                #self.forest = nltk.NaiveBayesClassifier.train(observations)
                #self.forest.show_most_informative_features()
                #self.forest.train(observations)

	def predict( self, observations, groundTruth = False):
		if groundTruth:
			predict = np.array([[self.classify(fe),g] for fe,g in observations])
		else:
			predict = np.array([[self.classify(fe)] for fe  in observations])

		return predict
	
	def classify( self, fe):
		return nltk.FreqDist([c.classify(fe)  for c in self.forest]).max()

	def accuracy( self, observations ):
		return sum([nltk.classify.accuracy(c,observations) for c in self.forest])/len(self.forest)

	def cm( self, data ):
		output= np.array(self.predict(data,True))
		print(nltk.ConfusionMatrix(list(output[:,1]),list(output[:,0])))


