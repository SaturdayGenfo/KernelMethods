# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from LargeMargin import LargeMargin
from tqdm import tqdm



def write_predictions(predictions, out_fname):
    
    data = [[np.abs((pred+1)//2)] for i, pred in enumerate(predictions)]
    data = np.concatenate([[['Bound']], data])
    
                
    data_frame = pd.DataFrame(data=data[1:,:], columns=data[0])
    data_frame.index.name = 'Id'
    data_frame.to_csv(out_fname)
    
    
def kernel_train(kernel, training_data, ytrain, lmda):
    
    K = kernel.gram(training_data)
    alpha = LargeMargin.SVM(K, ytrain, lmda)
    return alpha

def kernel_predict(kernel, alpha, training, test):
    
    predict = []
    for x in tqdm(test):
        predict.append(np.sign(kernel.eval_f(x, alpha, training)))
    
    return predict

def score(predict, yreal):
    
    return sum([int(predict[i]==yreal[i]) for i in range(len(yreal))])/len(yreal)
