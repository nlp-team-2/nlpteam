from prob3 import *
from RandomForest import *
from RandomForestReg import *
import numpy as np
try:
	import cPickle as pickle
except:
	import pickle


'''
MYSTERY
def train(classifier, features):
	classifier.train(*features)
	return classifier
'''

def train_pickle():
	data = read_data('devdata')

	a = RandomForest()
	a.train(age_features(data))

	b = RandomForestReg()
	x, y = birthyear_features(data)
	b.train(x, y)
	
	c = RandomForest()
	c.train(gender_features(data))

	d = RandomForest()
	d.train(education_features(data))

	'''
	a = train(RandomForest(), (age_features(data),))
	b = train(RandomForestReg(), birthyear_features(data))
	c = train(RandomForest(), (gender_features(data),))
	d = train(RandomForest(), (education_features(data),))
	'''

	with open('classifiers.pickl', 'wb') as f:
		pickle.dump((a, b, c, d), f)
	'''
	with open('classifiers.pickl', 'rb') as f:
		a1, b1, c1, d1 = pickle.load(f)
	a1.predict(age_features(data, include_id = True), groundTruth = True, include_id = True)
	'''


if __name__ == '__main__':
	train_pickle()
