import pandas as pd
import numpy as np
import scipy as sp
import qpsolvers as qp
import time

x1, x2 = "",""
def substring_kernel(X, k):
    global x1
    global x2
    n = X.shape[0]
    gram = np.zeros((n, n))
    kxx_val = {}
    for i in range(n):
        x1 = X[i]
        x2 = X[i]
        kxx_val[i] = memoK(k, len(x1), len(x2))
    for i in range(n):
        x1 = X[i]
        for j in range(i+1):
            x2 = X[j]
            gram[i,j] = memoK(k, len(x1), len(x2)) / ( kxx_val[i] * kxx_val[j] )**0.5
            gram[j,i] = gram[i,j]
    return gram

def memoK(k, lenX1, lenX2):
    memo = {}
    return K(k, lenX1, lenX2, memo)

def K(k, lenX1, lenX2, memo):
    global x1
    global x2
    if min(lenX1, lenX2) < k:
        return 0
    else:
        tmp = 0
        for j in range(1, lenX2):
            if x2[j] == x1[lenX1-1]:
                tmp += B(k-1, lenX1-1, j, memo)
        out = K(k, lenX1-1, lenX2, memo) + (lam**2)*tmp
        return out

def B(k, lenX1, lenX2, memo):
    if (k, lenX1, lenX2) in memo:
        return memo[(k, lenX1, lenX2)]
    else:
        global x1
        global x2
        if k == 0:
            return 1
        elif min(lenX1, lenX2) < k:
            return 0
        else:
            out = lam*B(k, lenX1-1, lenX2, memo) + lam*B(k, lenX1, lenX2-1, memo) - (lam**2)*B(k, lenX1-1, lenX2-1, memo)
            if x1[lenX1-1] == x2[lenX2-1]:
                out += (lam**2)*B(k-1, lenX1-1, lenX2-1, memo)
            memo[(k, lenX1, lenX2)] = out
            return out

train_string = pd.read_csv("./Xtr0.csv", sep=",", header = None)
train_string_matrix = train_string.values[1:,1]
X_train = train_string_matrix[:1500]
X_val = train_string_matrix[1500:]
label = pd.read_csv("./Ytr0.csv")
Y = label["Bound"].values
Y_train = Y[:1500]
Y_val = Y[1500:]

lam = 0.6
k = 10

start = time.time()
x1 = X_train[10]
x2 = X_train[17]
out = memoK(k, len(x1), len(x2))
end = time.time()
print(out, end - start)
