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

#classifier
class RandomForest:

	__slots__ = ('forest')

	def train( self, observations ):
                self.forest = SklearnClassifier(SVC())
                #self.forest = nltk.NaiveBayesClassifier.train(observations)
                #self.forest.show_most_informative_features()
                self.forest.train(observations)

	def predict( self, observations, groundTruth = False):
		if groundTruth:
			predict = np.array([[self.forest.classify(fe),g] for (fe,g) in obesravations])
		else:
			predict = np.array([[self.forest.classify(fe)] for fe  in obesravations])

		return predict

	def accuracy( self, observations ):
		return nltk.classify.accuracy(self.forest,observations)

