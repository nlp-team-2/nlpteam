from prob3 import *
from RandomForest import *
from RandomForestReg import *
import numpy as np
try:
	import cPickle as pickle
except:
	import pickle


def train(classifier, features):
	classifier.train(*features)
	return classifier


def train_pickle():
	data = read_data('devdata')

	a = train(RandomForest(), (age_features(data),))
	b = train(RandomForestReg(), birthyear_features(data))
	c = train(RandomForest(), (gender_features(data),))
	d = train(RandomForest(), (education_features(data),))

	with open('classifiers.pickl', 'wb') as f:
		pickle.dump((a, b, c, d), f)


if __name__ == '__main__':
	train_pickle()
