import pickle
import argparse
import prob3


def test(picklepath, testpath):
	with open(picklepath, 'rb') as f:
		age_classifier, birthyear_classifier, gender_classifier, education_classifier = pickle.load(f)
	data = prob3.read_data(testpath)

	age_features = prob3.age_features(data, include_id = True)
	#birthyear_features = prob3.birthyear_features(data, include_id = True)
	#gender_features = prob3.gender_features(data, include_id = True)
	#education_features = prob3.education_features(data, include_id = True)

	age_predictions = age_classifier.predict(age_features, True, True)
	#birthyear_predictions = birthyear_classifier.predict(birthyear_features, True, True)
	#gender_predictions = gender_classifier.predict(gender_features, True, True)
	#education_predictions = education_classifier.predict(education_features, True, True)

	with open('age.txt', 'w') as f:
		for line in age_predictions:
			print('{}\tprediction: {}'.format(line[0], line[1]))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test project 3 classifiers')
	parser.add_argument('-p', help='the pickle file holding the classifiers')
	parser.add_argument('-t', help='the test folder')
	args = parser.parse_args()
	test(args.p, args.t)
