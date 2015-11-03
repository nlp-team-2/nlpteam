import pickle
import argparse
import os
import prob3
import numpy as np


picklepath = 'classifiers'
textdir = 'textfiles'


def load_classifier(fname):
	with open(os.path.join(picklepath, fname), 'rb') as f:
		return pickle.load(f)


def test(picklepath, testpath):
	age_classifier = load_classifier('age.pickl')
	birthyear_classifier = load_classifier('birthyear.pickl')
	gender_classifier = load_classifier('gender.pickl')
	education_classifier = load_classifier('education.pickl')

	data = prob3.read_data(testpath)

	age_features = prob3.age_features(data, include_id = True)
	birthyear_features, _ = prob3.birthyear_features(data, include_id = True)
	gender_features = prob3.gender_features(data, include_id = True)
	education_features = prob3.education_features(data, include_id = True)

	age_predictions = age_classifier.predict(age_features, True, True)
	birthyear_predictions = birthyear_classifier.predict(birthyear_features, True, True)
	gender_predictions = gender_classifier.predict(gender_features, True, True)
	education_predictions = education_classifier.predict(education_features, True, True)

	if not os.path.exists(textdir):
		os.makedirs(textdir)

	with open(os.path.join(textdir, 'age.txt'), 'w') as f:
		age_predictions = np.sort(age_predictions, axis=0)
		birthyear_predictions = np.sort(birthyear_predictions, axis=0)
		for a, b in zip(age_predictions, birthyear_predictions):
			f.write('{}\t{} {}\n'.format(a[0], a[1], b[1]))

	with open(os.path.join(textdir, 'gender.txt'), 'w') as f:
		for line in gender_predictions:
			f.write('{}\t{}\n'.format(line[0], line[1]))

	with open(os.path.join(textdir, 'education.txt'), 'w') as f:
		for line in education_predictions:
			f.write('{}\t{}\n'.format(line[0], line[1]))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test project 3 classifiers')
	parser.add_argument('-p', help='the directory holding the pickled classifiers')
	parser.add_argument('-t', help='the test folder')
	args = parser.parse_args()
	if args.p is None or args.t is None:
		parser.print_help()
	else:
		test(args.p, args.t)
