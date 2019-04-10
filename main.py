# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm


from DataHandler import DataHandler
from LargeMargin import LargeMargin
from Kernel import Kernel
from utils import kernel_train, kernel_predict, write_predictions


print('''
------------------------------------------------------------
        DATASET 0
------------------------------------------------------------
''')

fname = '0'
dataset = DataHandler('data/Xtr'+fname+'.csv')

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataHandler('data/Xte'+fname+'.csv')

dataset.X = pd.concat([dataset.X, test.X], axis = 0, ignore_index = True)


dataset.populate_kmer_set(k = 9)
dataset.mismatch_preprocess(k=9, m=1)
K9 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.populate_kmer_set(k = 10)
dataset.mismatch_preprocess(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.populate_kmer_set(k = 11)
dataset.mismatch_preprocess(k=11, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)


K = K9 + K10 + K11

training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

lmda = 0.76


alpha = LargeMargin.SVM(K[training][:, training], y, lmda)

pred0 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += alpha[k]*K[i, j]
    pred0.append(np.sign(val))


print('''
------------------------------------------------------------
        DATASET 1
------------------------------------------------------------
''')

fname = '1'
dataset = DataHandler('data/Xtr'+fname+'.csv')

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataHandler('data/Xte'+fname+'.csv')


dataset.X = pd.concat([dataset.X, test.X], axis = 0, ignore_index = True)


dataset.populate_kmer_set(k = 9)
dataset.mismatch_preprocess(k=9, m=1)
K9 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.populate_kmer_set(k = 10)
dataset.mismatch_preprocess(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.populate_kmer_set(k = 11)
dataset.mismatch_preprocess(k=11, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)


K = K9 + K10 + K11

training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

lmda = 0.833


alpha = LargeMargin.SVM(K[training][:, training], y, lmda)

pred1 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += alpha[k]*K[i, j]
    pred1.append(np.sign(val))


print('''
------------------------------------------------------------
        DATASET 2
------------------------------------------------------------
''')

fname = '2'
dataset = DataHandler('data/Xtr'+fname+'.csv')

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

test = DataHandler('data/Xte'+fname+'.csv')

    
dataset.populate_kmer_set(12)
test.kmer_set = dataset.kmer_set

dataset.mismatch_preprocess(12 , 0)
test.mismatch_preprocess(12, 0)

kernel = Kernel(Kernel.sparse_gaussian(7.8))


lmda = 0.00000001

alpha = kernel_train(kernel, dataset.data, y, lmda)
pred2 = kernel_predict(kernel, alpha, dataset.data, test.data)


print('''
------------------------------------------------------------
        KAGGLEIZER
------------------------------------------------------------
''')

out_fname = "Yte.csv"
predictions = pred0 + pred1 + pred2
write_predictions(predictions, out_fname)
