# code modified from helq's code on github

import numpy as np
cimport numpy as np
from cpython cimport array
import array
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def ssk(s_, t_, int n, float lbda):
    cdef int lens, lent
    cdef int i, sa, tb
    cdef float tmp
    cdef long[:] s # this reduces the overhead 10x fold!!!
    cdef long[:] t

    s = array.array('l', [ord(c) for c in s_])
    t = array.array('l', [ord(c) for c in t_])
    lens, lent = len(s), len(t)

    cdef np.ndarray[np.float64_t, ndim=3] \
        B = np.zeros( (n+1, lens+1, lent+1), dtype=np.float )
    B[0,:,:] = 1

    for i in range(1,n+1):
        for sa in range(i,lens+1):
            tmp = 0.
            for tb in range(i,lent+1):
                if s[sa-1]==t[tb-1]: # trick taken from shogun implemantion of SSK
                    tmp = lbda * (tmp + lbda*B[i-1,sa-1,tb-1])
                else:
                    tmp *= lbda
                B[i,sa,tb] = tmp + lbda * B[i, sa-1, tb]

    cdef float K = 0.
    for sa in range(n-1,lens):
        for tb in range(n-1,lent):
            if s[sa]==t[tb]:
                K += B[n-1,sa,tb]

    return K*lbda*lbda
