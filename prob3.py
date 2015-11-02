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
from RandomForestReg import *

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


def occupation_feature(user):
	occ = user['Occupation']
	if occ is None:
		return 'NA'
	elif 'student' in occ:
		return 'student'
	else:
		return 'worker'


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
			f['cute'] = avg_word('cute', tweets_d)
			f['yay!'] = avg_word('yay!', tweets_d)
			f['love'] = avg_word('love', tweets_d)
			f['<3'] = avg_word('<3', tweets_d)
			f['occupation'] = occupation_feature(user_d)

		features.append((f,target))
		
	print(counta, countb)
	return features


quad_regex = re.compile(r'([a-zA-Z!])\1{3,}')
def avg_quads(user):
	if 'tweets' not in user:
		return 0

	total = 0
	for tweet in user['tweets'].values():
		total += len(quad_regex.findall(tweet['text']))

	num_tokens = sum(tweet['tokens'] for tweet in user['tweets'].values())
	return total / num_tokens


def age_features(data):
	features=[]
	for user in data:
		if 'Year' in user['user'] and user['user']['Year'] is not None:
			age = 2013 - int(user['user']['Year'])
			if age <= 25:
				target = '25under'
			elif age <= 35:
				target = '35under'
			else:
				target = '36over'
		else:
			target = 'NA'
		f = defaultdict(int)
		f['avg_quads'] = avg_quads(user)
		f['cnt_Lang'] = len(user['user']['Languages'])
		f['cnt_Regs'] = len(user['user']['Regions'])
		if 'tweets' in user.keys():
			tweets_d = user['tweets']
			f['haha'] = avg_word('haha', tweets_d)
			f['cute'] = avg_word('cute', tweets_d)
			f['yay!'] = avg_word('yay!', tweets_d)
			f['love'] = avg_word('love', tweets_d)
			f['<3'] = avg_word('<3', tweets_d)
		features.append((f, target))
	return features


def age_features2(data):
	features = []
	for user in data:
		if 'tweets' not in user:
			continue
		if 'Year' in user['user'] and user['user']['Year'] is not None:
			age = 2013 - int(user['user']['Year'])
			if age <= 25:
				target = '25under'
			elif age <= 35:
				target = '35under'
			else:
				target = '36over'
		langs = len(user['user']['Languages'])
		regions = len(user['user']['Regions'])
		for tweet in user['tweets'].values():
			f = defaultdict(int)
			f['tokens'] = tweet['tokens']
			f['avg_tokenlen'] = sum([len(t) for t in tweet['tokenized']])/tweet['tokens']
			f['cnt_Lang'] = langs
			#f['cnt_Regs'] = regions
			f['punc'] = tweet['punc']
			f['haha'] = tweet['text'].count('haha')
			f['cute'] = tweet['text'].count('cute')

			features.append((f, target))

	return features

def birthyear_features(data):
	features = []
	for user in data:
		f = defaultdict(int)
		if user['user']['Year'] != None:
			target = int(user['user']['Year'])
		else:
			target = -1
		f['cnt_Lang'] = len(user['user']['Languages'])
		f['cnt_Regs']  = len(user['user']['Regions'])
		if 'tweets' in user.keys():
			tweets_d = user['tweets']
			f['haha'] = avg_word('haha', tweets_d)
			f['No_pronouns'] = avg_pronouns(tweets_d)
		
		features.append((f,target))
	
	keylist = ['cnt_Lang','cnt_Regs','haha','No_pronouns']
	return features,keylist


def split_data(data):
	divider = int(len(data)*.8)
	return data[divider:], data[:divider]


if __name__ == '__main__':
	data = read_data('devdata')
	#f = RandomForest()
	#features = age_features(data)
	#f.train(features)
	#f.cm(features)
	#print(f.accuracy(features))
	by_features,by_keylist = birthyear_features(data)
	by_r= RandomForestReg()
	by_r.train(by_features,by_keylist)
	print(by_r.accuracy(by_features))
	print(by_r.MSE(by_features))
	output= by_r.predict(by_features,True)
	print(2013-output)
	pdb.set_trace()
