#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from LargeMargin import LargeMargin
from Kernel import Kernel
from tqdm import tqdm
import pandas as pd

def project(v):
        
        mu = list(v)
        mu.sort()
        cumul_sum = np.cumsum(mu)
        rho = np.max([j for j in range(0, len(mu)) if mu[j] - 1/(j+1)*(cumul_sum[j] - 1) > 0])
        
        theta = 1/(rho+1)*(cumul_sum[rho] - 1)
        return np.array([max(0, vi - theta) for vi in v])

def MKL(kernels, y, lmda, T):
    
    m = len(kernels)
    d = np.array([1/m for k in range(m)])
    
    for t in range(T):
        
        K = np.zeros_like(kernels[0])
        for i, Km in enumerate(kernels):
            K = K + d[i]*Km
        
        alpha = LargeMargin.SVM(K, y, lmda) #Resoud pour la somme
        
        
        grad = [-0.5*lmda*np.dot(alpha.T, np.dot(Km, alpha))[0][0] for Km in kernels]
        step = 0.01
        d = project(d - step*np.array(grad)) #Projette le gradient sur le simplexe
    
    return d
        
