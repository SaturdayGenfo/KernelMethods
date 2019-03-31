import pandas as pd
import numpy as np
import time
from cvxopt import solvers, matrix, spmatrix, sparse
from sklearn.model_selection import train_test_split

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
from string_kernel import ssk

def SVM(K, y, lmda):
    print("Optimizing")
    solvers.options['show_progress'] = False

    n = len(y)
    q = -matrix(y, (n, 1), tc='d')
    h = matrix(np.concatenate([np.ones(n)/(2*lmda*n), np.zeros(n)]).reshape((2*n, 1)))
    P = matrix(K)
    Gtop = spmatrix(y, range(n), range(n))
    G = sparse([Gtop, -Gtop])

    sol = solvers.qp(P, q, G, h)['x']
    return sol

def substring_kernel(X, k, lam, norm = False):
    n = X.shape[0]
    gram = np.zeros((n, n))
    if norm:
        kxx_val = {}
        for i in range(n):
            kxx_val[i] = ssk(X[i], X[i], k, lam)
    for i in range(n):
        for j in range(i+1):
            gram[i,j] = ssk(X[i], X[j], k, lam)
            if norm:
                gram[i,j] = gram[i,j] / ( kxx_val[i] * kxx_val[j] )**0.5
            gram[j,i] = gram[i,j]
    return gram

def f_ssk(x, X, alpha, k, lam):
    out = sum([ alpha[i]*ssk(x, X[i], k, lam) for i in range(X.shape[0]) ])
    return out

def evaluateSVM(K, lamSSK, lamSVM, X_train, Y_train, X_val, Y_val):
    alpha = SVM(K, Y_train, lamSVM)
    s = 0
    s_plus = 0
    s_moins = 0
    for i in range(Y_val.shape[0]):
        f_val = f_ssk(X_val[i], X_train, alpha, k, lamSSK)
        if (f_val > 0 and Y_val[i] == 1):
            s_plus += 1
            s += 1
        if (f_val < 0 and Y_val[i] == -1):
            s_moins += 1
            s += 1
    return s, s_plus, s_moins

train_string = pd.read_csv("./Xtr2.csv", sep=",", header = None)
X = train_string.values[1:,1]
label = pd.read_csv("./Ytr2.csv")
Y = label["Bound"].values
for i in range(2000):
    if Y[i] == 0:
        Y[i] = -1

test_string = pd.read_csv("./Xte2.csv", sep=",", header = None)
Xtest = test_string.values[1:,1]

X = np.concatenate((X,Xtest))

k = 9 #8 8 9
lamSSK = 0.01 #0.01 0.1 0.01
K = substring_kernel(X, k, lamSSK, norm = False)
np.save("K2.npy", K)

# lam = 0.01
# k = 10
# start = time.time()
# #out = substring_kernel(X[:1500], k, lam)
# out = ssk(X[17], X[11], k, lam)
# end = time.time()
# print(out, end - start)

# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, random_state=42)
# X_val1, X_val2, y_val1, y_val2 = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
# X_train1 = np.concatenate((X_train, X_val2))
# y_train1 = np.concatenate((y_train, y_val2))
# X_train2 = np.concatenate((X_train, X_val1))
# y_train2 = np.concatenate((y_train, y_val1))
# X_train = [X_train1, X_train2]
# y_train = [y_train1, y_train2]
# X_val = [X_val1, X_val2]
# y_val = [y_val1, y_val2]
#
# results = {}
# average_results = {}
# k_params = [7, 8, 9, 10]
# lamSSK_params = [0.01, 0.05, 0.1]
# lamSVM_params = [1e-3, 1e-4]
#
# start = time.time()
# for k in k_params:
#     for lamSSK in lamSSK_params:
#         for i in range(2):
#             K = substring_kernel(X_train[i], k, lamSSK, norm = False)
#             for lamSVM in lamSVM_params:
#                 s, s_plus, s_moins = evaluateSVM(K, lamSSK, lamSVM, X_train[i], y_train[i], X_val[i], y_val[i])
#                 print(i,k,lamSSK,lamSVM,s,s_plus,s_moins)
#                 results[(i,k,lamSSK,lamSVM)] = s
# for k in k_params:
#     for lamSSK in lamSSK_params:
#         for lamSVM in lamSVM_params:
#             average_results[(k,lamSSK,lamSVM)] = (results[(0,k,lamSSK,lamSVM)] + results[(1,k,lamSSK,lamSVM)]) / 2
#
# print(average_results)
#
# np.save("d1.npy", results)
# np.save("d2.npy", average_results)
# end = time.time()
# print(end - start)

# dicta = np.load("./d2.npy").item()
# for k in k_params:
#    for lamSSK in lamSSK_params:
#        for lamSVM in lamSVM_params:
#            print ("k: {} \t| lamSSK: {} \t| lamSVM: {}     \t| score {}".format(k,lamSSK,lamSVM,dicta[(k,lamSSK,lamSVM)]) )
