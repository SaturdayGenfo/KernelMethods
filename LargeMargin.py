from cvxopt import solvers, matrix, spmatrix, sparse

class LargeMargin():
    
    def SVM(K, y, lmda):
    
        print("Optimizing")
    
        solvers.options['show_progress'] = True
    
        n = len(y)
        q = -matrix(y, (n, 1), tc='d')
        h = matrix(np.concatenate([np.ones(n)/(2*lmda*n), np.zeros(n)]).reshape((2*n, 1)))
        P = matrix(K)
        Gtop = spmatrix(y, range(n), range(n))
        G = sparse([Gtop, -Gtop])

    
        sol = solvers.qp(P, q, G, h)['x']
    
        return sol
