# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm


from DataHandler import DataHandler
from Kernel import Kernel
from LargeMargin import LargeMargin
from utils import *


'''
------------------------------------------------------------
        DATASET 0
------------------------------------------------------------
'''

fname = '0'
dataset = DataHandler('data/Xtr'+fname+'.csv')

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataHandler('data/Xte'+fname+'.csv')

    
dataset.populate_kmer_set(11)
test.kmer_set = dataset.kmer_set

dataset.mismatch_preprocess(11 , 1)
test.mismatch_preprocess(11, 1)

kernel = Kernel(Kernel.mismatch())

lmda = 0.1

alpha = kernel_train(kernel, dataset.data, y, lmda)
pred0 = kernel_predict(kernel, alpha, dataset.data, test.data)


'''
------------------------------------------------------------
        DATASET 1
------------------------------------------------------------
'''

fname = '1'
dataset = DataHandler('data/Xtr'+fname+'.csv')

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataHandler('data/Xte'+fname+'.csv')

    
dataset.populate_kmer_set(9)
test.kmer_set = dataset.kmer_set

dataset.mismatch_preprocess(9 , 1)
test.mismatch_preprocess(9, 1)

kernel = Kernel(Kernel.mismatch())

lmda = 0.1

alpha = kernel_train(kernel, dataset.data, y, lmda)
pred1 = kernel_predict(kernel, alpha, dataset.data, test.data)


'''
------------------------------------------------------------
        DATASET 2
------------------------------------------------------------
'''

fname = '2'
dataset = DataHandler('data/Xtr'+fname+'.csv')

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataHandler('data/Xte'+fname+'.csv')

    
dataset.populate_kmer_set(12)
test.kmer_set = dataset.kmer_set

dataset.mismatch_preprocess(12 , 0)
test.mismatch_preprocess(12, 0)

kernel = Kernel(Kernel.sparse_gaussian(10))


lmda = 0.0000001

alpha = kernel_train(kernel, dataset.data, y, lmda)
pred2 = kernel_predict(kernel, alpha, dataset.data, test.data)


'''
------------------------------------------------------------
        KAGGLEIZER
------------------------------------------------------------
'''

out_fname = "kaggle.csv"
predictions = pred0 + pred1 + pred2
write_predictions(predictions, out_fname)
