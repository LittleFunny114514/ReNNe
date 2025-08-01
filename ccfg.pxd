# cython: language_level=3
cimport numpy as np
cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t
from cython cimport parallel
from cython.parallel cimport prange, parallel, threadid
cimport openmp as omp
from libc.math cimport sqrt, pow, fabs, ceil, floor
from libc.math cimport exp, log

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t
    np.int32_t
    np.int64_t

ctypedef np.int64_t np_size_t