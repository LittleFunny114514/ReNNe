from ..cfg import *
from ..ccfg import *
from ..ccfg cimport *


cdef np.ndarray argmax2dc(np.ndarray[DTYPE_t, ndim=3] arr)
cdef np.ndarray argmax3dc(np.ndarray[DTYPE_t, ndim=4] arr)

#b means batch in function declarations below.

cpdef np.ndarray maxpooling2db(np.ndarray[DTYPE_t, ndim=3] fm, int ksize, int stride=*)
cpdef np.ndarray bwdmaxpooling2db(np.ndarray[DTYPE_t, ndim=3] fm, np.ndarray[DTYPE_t, ndim=3] grad_output, int ksize, int stride=*)
cpdef np.ndarray conv2db(np.ndarray[DTYPE_t,ndim=3] fm,np.ndarray[DTYPE_t,ndim=3] krnl,int padding=*)