# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm


from DataHandler import DataHandler
from Kernel import Kernel
from LargeMargin import LargeMargin
from utils import *

fname = '0'
dataset = DataHandler('data/Xtr'+fname+'.csv')


labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

def split_data(dataset, y, k, m):
    
    dataset.populate_kmer_set(k)
    dataset.mismatch_preprocess(k, m)
    idx = range(len(dataset.data))
    pairs = []
    data_tranches = [idx[500*i : 500*i+ 500] for i in range(4)]
    label_tranches = [y[500*i: 500*i + 500] for i in range(4)]
    for i in range(4):
        test, ytest = data_tranches[i], label_tranches[i]
        train = np.concatenate([data_tranches[j] for j in range(4) if j != i])
        ytrain = np.concatenate([label_tranches[j] for j in range(4) if j != i])
        
        pairs.append((train, ytrain, test, ytest))
    return pairs
    
kernel = Kernel(Kernel.sparse_gaussian(0.1))

pairs = split_data(dataset, y, 8, 0)
bigK = kernel.gram(dataset.data)

avg = 0
lmda = 0.01

for train, ytrain, test, ytest in pairs:
    K = np.array([[bigK[i,j] for j in train] for i in train])
    alpha = LargeMargin.SVM(K, ytrain, lmda)
    predict = []
    for j in tqdm(test):
        eval_f = np.sum([alpha[k]*bigK[j, i] for k,i in enumerate(train)])
        predict.append(np.sign(eval_f))
    s = score(predict, ytest)
    print("Score : ", s)
    avg += s

print("Parameter Score : ", avg/4)


