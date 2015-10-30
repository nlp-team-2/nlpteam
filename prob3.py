import pickle
import numpy as np
import os
import re
import pdb
from collections import defaultdict

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
    dct = {}
    for i, d in enumerate(data):
        if 'transforms' in d.keys():
            dct[i] = categorize(d['transforms'])
        else:
            dct[i] = []
    return dct


if __name__ == '__main__':
    data = read_data('devdata')
    categories = categorize_all(data)
    pdb.set_trace()
