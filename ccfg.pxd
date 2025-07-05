cimport numpy as np
cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.math cimport exp, log
from libc.stddef cimport size_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t
    np.int32_t
    np.int64_t

ctypedef np.int64_t np_size_t