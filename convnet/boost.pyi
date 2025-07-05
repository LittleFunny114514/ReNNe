import numpy as np
from .. import ccfg

def argmax2d(arr: np.ndarray) -> np.ndarray:
    """
    Input a 3d array, return the index of the maximum value in each 2d array
    """
    ...

# b means batch in function declarations below
def maxpooling2db(fm: np.ndarray, ksize: int, stride: int = 0) -> np.ndarray:
    ...
def bwdmaxpooling2db(fm: np.ndarray, grad_output: np.ndarray, ksize: int, stride: int = 0) -> np.ndarray:
    ...
def conv2db(fm: np.ndarray, krnl: np.ndarray, padding: int = 0) -> np.ndarray:
    ...
