import numpy
import scipy

np = numpy  # We will write a numpy-like interface for a compute framework satisfied with OpenCL
sp = scipy
dtype = np.float32
devide_epsilon = dtype(1e-8)

IGNORE_PYTHON_VERSION_ISNT_SATISFIED = True
INDENT_LEN = 4

NO_GRAD = False
IMMEDIATE_REMOVE_HIDDEN = True

PYX_BOOST = True
BUILD_PYX = True
CREATE_DBGINFO = False


import os, sys

RENNE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MAX_PYCALLSTACK_SIZE = 8192
