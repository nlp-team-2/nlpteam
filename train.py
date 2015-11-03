from prob3 import *
from RandomForest import *
from RandomForestReg import *
import os
import numpy as np
try:
	import cPickle as pickle
except:
	import pickle


pickledir = 'classifiers'


def train(classifier, features, fname):
	classifier.train(*features)
	with open(os.path.join(pickledir, fname), 'wb') as f:
		pickle.dump(classifier, f)


def train_pickle():
	data = read_data('devdata')

	if not os.path.exists(pickledir):
		os.makedirs(pickledir)

	train(RandomForest(), (age_features(data),), 'age.pickl')
	train(RandomForestReg(), birthyear_features(data), 'birthyear.pickl')
	train(RandomForest(), (gender_features(data),), 'gender.pickl')
	train(RandomForest(), (education_features(data),), 'education.pickl')


if __name__ == '__main__':
	train_pickle()
