import pickle
import argparse
from prob3 import *


def test(picklepath, testpath):
	with open(picklepath, 'rb') as f:
		age_classifier, birthyear_classifier, gender_classifier, education_classifier = pickle.load(f)
	data = read_data(testpath)

	age_features = age_features(data)
	birthyear_features = birthyear_features(data)
	gender_features = gender_features(data)
	education_features = education_features(data)

	age_predictions = age_classifier.predict(age_features)
	birthyear_predictions = birthyear_classifier.predict(birthyear_features)
	gender_predictions = gender_classifier.predict(gender_features)
	education_predictions = education_classifier.predict(education_features)

	with open('age.txt', 'w') as f:
		for line, p in enumerate(age_predictions):
			#fmt = '{}\tprediction: {}'.format(data[line]['id']
			pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test project 3 classifiers')
	parser.add_argument('-p', help='the pickle file holding the classifiers')
	parser.add_argument('-t', help='the test folder')
	args = parser.parse_args()
	test(args.p, args.t)
