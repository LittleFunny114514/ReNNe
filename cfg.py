from typing import Literal
import numpy
import scipy
import scipy.ndimage
import numba
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import asyncio
import os

np = numpy  # We will write a numpy-like interface for a compute framework satisfied with OpenCL
sci = scipy
nb = numba
dtype = np.float32
divide_epsilon = dtype(1e-8)
nb.NUMBA_DISABLE_CACHE = False

IGNORE_PYTHON_VERSION_ISNT_SATISFIED = True
INDENT_LEN = 4

NO_GRAD = False
IMMEDIATE_REMOVE_HIDDEN = False


CREATE_DBGINFO = False

PARALLEL_MODE: Literal["none", "asyncio_multiprocess"] = "none"
PARALLEL_MAX_PROCCESS_N: int = 4

PARALLELMODE_SYNC = PARALLEL_MODE == "none"
PARALLELMODE_ASYNCIO_MULTIPROCCESS = PARALLEL_MODE == "asyncio_multiprocess"

if __name__ == "__main__" and PARALLELMODE_ASYNCIO_MULTIPROCCESS:
    ent_loop = asyncio.get_event_loop()
    procpool = ProcessPoolExecutor()
    ctx = mp.get_context("spawn")

RENNE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MAX_PYCALLSTACK_SIZE = 8192
