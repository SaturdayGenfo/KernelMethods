import numpy as np
from tqdm import tqdm

class Kernel():
   
    def gaussian(sigma):
        return lambda x, y : 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-np.linalg.norm(x - y)**2/(2*sigma**2))
    
    def linear():
        return lambda x, y: np.dot(x, y)
    
    def polynomial(c, n):
        return lambda x, y : (np.dot(x, y) + c)**n
    
    def spectrum():
        def f(x, y):
            prod_scal = 0
            for kmer in x:
                if kmer in y:
                    prod_scal += x[kmer]*y[kmer]
            return prod_scal
        return f
    
    def mismatch():
        def f(x, y):
            prod_scal = 0
            for idx in x:
                if idx in y:
                    prod_scal += x[idx]*y[idx]
            return prod_scal
        return f
    
    def __init__(self, func, normalized = False):
        self.kernel = func
        self.normalized = normalized
        self.diag = np.array([])
        
    def gram(self, data):
        n = len(data)
        K = np.zeros((n, n))
        print("Computing Gram Matrix")
        for i in tqdm(range(n)):
            for j in range(i+1):
                prod_scal = self.kernel(data[i], data[j])
                K[i, j] = prod_scal
                K[j, i] = prod_scal
        
        if self.normalized:
            self.diag = np.sqrt(np.diag(K))
            print(self.diag.shape)
            for i in range(n):
                K[i, :] = K[i,:]/self.diag[i]
                K[:, i] = K[:, i]/self.diag[i]
            
        return K
    
    def eval_f(self, x, alpha, data):
        if self.normalized:
            square_norm_x = np.sqrt(self.kernel(x, x))
            result = np.sum([(alpha[i]*self.kernel(x, xi))/(square_norm_x * self.diag[i]) for i, xi in enumerate(data)])
        else:
            result =  np.sum([alpha[i]*self.kernel(x, xi) for i, xi in enumerate(data)])
        return result 
