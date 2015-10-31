import pickle
import numpy as np
import os
import re
import pdb
import random
import numpy
import itertools
from textblob import TextBlob
from collections import defaultdict
from RandomForest import *


user_file_name = 'user.pickl'
tweet_file_name = 'tweets.pickl'

def read_data(dirname):
	'''
	return a list of dictionaries with possible keys
	user, tweets, transforms, replacements, or ngrams
	'''
	lst = []
	for d in sorted(os.listdir(dirname)):
		# take only numerical folders
		if not re.match(r'\d+', d):
			continue

		loaded = {}
		for pickl in os.listdir(os.path.join(dirname, d)):
			# take only pickle files
			name, ext = os.path.splitext(pickl)
			if ext != '.pickl':
				continue
			with open(os.path.join(dirname, d, pickl), 'rb') as f:
				loaded[name] = pickle.load(f)

		lst.append(loaded)
	return lst


def categorize(transforms):
	'''
	return a dictionary with keys 0 to 9 inclusive corresponding to the
	position of the true boolean in the transform data. each key holds
	a list of transforms.
	'''
	dct = defaultdict(list)
	for k, v in transforms.items():
		for trans in v:
			t, bools = trans[0], trans[1:]
			for i, b in enumerate(bools):
				if b:
					num = i
					break
			dct[num].append(t)
	return dct


def categorize_all(data):
	'''
	dictionary mapping user id to its transform categories
	'''
	dct = {}
	for i, d in enumerate(data):
		if 'transforms' in d.keys():
			dct[i] = categorize(d['transforms'])
		else:
			dct[i] = []
	return dct


def avg_pronouns(tweets):
	thesum = 0
	words = 0
	for tweet in tweets.values():
		tags = TextBlob(tweet['text']).tags
		words += len(tags)
		thesum += sum([1 for word, tag in tags if tag == 'PRP'])
	return thesum / words


def avg_word(word, tweets):
    return sum([1 for tweet in tweets.values() if word in tweet['text']]) / len(tweets)



def gender_features(data):
	features=[]
	counta, countb= 0,0
	for user in data:
		target = user['user']['Gender']
		user_d = user['user']
		if target == None:
			continue
		f = defaultdict(int)
		#f['cnt_Lang'] = len(user_d['Languages'])
		#f['cnt_Regn'] = len(user_d['Regions'])
		if 'tweets' in user.keys():
			tweets_d = user['tweets']
			for tweet in tweets_d.values():
				if 'haha' in tweet['text']:
					if target == 'Male':
						counta += 1
					else:
						countb += 1
			f_punc_tokens =np.array([[tweets_d[key]['punc'],tweets_d[key]['tokens']] for key in tweets_d.keys()])
			f_punc_tokens = list(np.mean(f_punc_tokens,axis=0))
			f['avg_punc'] = f_punc_tokens[0]
			f['tokens'] = f_punc_tokens[1]
			#f['No_Of_tweets'] = len(tweets_d)
			f['No_pronouns'] = avg_pronouns(tweets_d)
			f['haha'] = avg_word('haha', tweets_d)

		features.append((f,target))
		
	print(counta, countb)
	return features


def split_data(data):
        #random.shuffle(data)
        divider = int(len(data)*.8)
        return data[divider:], data[:divider]


def kfold(data, k):
    splitdata = np.array_split(data, k)
    combos = list(reversed(list(itertools.combinations(splitdata, k-1))))
    accuracy_sum = 0
    for i in range(k):
        train= list(itertools.chain(*combos[i]))
        
        test = splitdata[i]
        c = RandomForest()
        c.train(train)
        accuracy_sum += c.accuracy(test)
    return accuracy_sum/k


if __name__ == '__main__':
    data = read_data('devdata')
    categories = categorize_all(data)
    print(kfold(gender_features(data), 5))
    pdb.set_trace()
