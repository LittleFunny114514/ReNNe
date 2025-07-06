import numpy as np
from .. import ccfg

def argmax2d(arr: np.ndarray) -> np.ndarray:
    """
    Input a 3d array, return the index of the maximum value in each 2d array
    """
    ...

# b means batch in function declarations below
def maxpooling2db(fm: np.ndarray, ksize: int, stride: int = 0) -> np.ndarray: ...
def bwdmaxpooling2db(
    fm: np.ndarray, grad_output: np.ndarray, ksize: int, stride: int = 0
) -> np.ndarray: ...
def conv2db(
    fm: np.ndarray, krnl: np.ndarray, padx: int = 0, pady: int = -2147483648
) -> np.ndarray: ...
def conv3db(
    fm: np.ndarray,
    krnl: np.ndarray,
    padx: int = 0,
    pady: int = -2147483648,
    padz: int = -2147483648,
) -> np.ndarray: ...
def grilleInterpolation2db(fm: np.ndarray, size: int, mode: bool = True) -> np.ndarray:
    """
    if mode==True, then upsample the image by size times
    if mode==False, then downsample the image by size times
    """
    ...
