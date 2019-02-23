# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm


from DataHandler import DataHandler
from Kernel import Kernel
from LargeMargin import LargeMargin
from utils import *

fname = '1'
dataset = DataHandler('data/Xtr'+fname+'.csv')
dataset.spectrum_preprocess(k = 8)

labels = pd.read_csv('data/Ytr'+fname+'.csv')
y = 2.0*np.array(labels['Bound']) - 1

ytrain, yval = y[:1500], y[1500:]

training, validation = dataset.data[:1500], dataset.data[1500:]


kernel = Kernel(Kernel.spectrum())

alpha = kernel_train(kernel, training, ytrain, 0.01)
pred = kernel_predict(kernel, alpha, training, validation)

print(score(pred, yval))

