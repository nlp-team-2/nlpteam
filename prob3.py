import pickle
import numpy as np
import os
import re
import pdb
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

def gender_features(data):
    features=[]
    for user in data:
        target = user['user']['Gender']
        user_d = user['user']
        f = defaultdict(0)
        f['cnt_Lang'] = len(user_d['Languages'])
        f['cnt_Regn'] = len(user['Regions'])
        if 'tweets' in user.keys():
            tweets_d = user['tweets']
            f_punc_tokens =np.array([[tweets_d[key]['punc'],tweets_d[key]['tokens']] for key in tweets_d.keys()])
            f_punc_tokens = list(np.mean(f_punc_tokens,axis=0))
            f['avg_punc'] = f_punc_tokens[0]
            f['tokens'] = f_punc_tokens[1]
            f['No_Of_tweets'] = len(tweets_d)
			
        features.append((f,target))
		
    return features

if __name__ == '__main__':
    data = read_data('devdata')
    categories = categorize_all(data)
    c = RandomForest()
    c.train(GenderClassifiyFeatures(data))
    pdb.set_trace()
