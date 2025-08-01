# cython: language_level=3
import numpy as np
cimport numpy as np
from . import cfg
import cython
from cython.parallel import prange, parallel, threadid