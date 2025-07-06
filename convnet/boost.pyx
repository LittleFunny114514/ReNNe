# cython: language_level=3

from ..cfg import *
from ..ccfg import *
from ..ccfg cimport *


cdef np.ndarray argmax2dc(np.ndarray[DTYPE_t, ndim=3] arr):
    cdef size_t i,j,k,maxj=0,maxk=0
    cdef np.ndarray[np_size_t,ndim=2] argmax = np.zeros((arr.shape[0],2),dtype=np.int64)
    for i in range(arr.shape[0]):
        maxj, maxk = 0, 0
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if arr[i,j,k] > arr[i,maxj,maxk]:
                    maxj, maxk = j, k
        argmax[i, 0], argmax[i,1] = maxj, maxk
    return argmax

cdef np.ndarray argmax3dc(np.ndarray[DTYPE_t, ndim=4] arr):
    cdef size_t i,j,k,l,maxj=0,maxk=0,maxl=0
    cdef np.ndarray[np_size_t,ndim=3] argmax = np.zeros((arr.shape[0],3),dtype=np.int64)
    for i in range(arr.shape[0]):
        maxj, maxk, maxl = 0, 0, 0
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                for l in range(arr.shape[3]):
                    if arr[i,j,k,l] > arr[i,maxj,maxk,maxl]:
                        maxj, maxk, maxl = j, k, l
        argmax[i, 0], argmax[i,1], argmax[i,2] = maxj, maxk, maxl
    return argmax

def argmaxnd(arr):
    amax=np.zeros(arr.shape[0],dtype=np.int32)
    for i in range(arr.shape[0]):
        amax[i]=np.argmax(arr[i])
    ret=np.zeros((arr.ndim-1,arr.shape[0]),dtype=np.int32)
    prod=np.prod(arr.shape[2:])
    for i in range(arr.ndim-1):
        ret[i]=amax//prod
        amax%=prod
        prod//=arr.shape[i+1]
    return ret.T

def argmax2d(arr:np.ndarray):
    # return the index of the maximum element in each fm in arr
    # arr: (fm, h, w)
    # return: (fm, 2)
    if arr.dtype == np.float32:
        return argmax2dc(<np.ndarray[np.float32_t,ndim=3]>arr)
    elif arr.dtype == np.float64:
        return argmax2dc(<np.ndarray[np.float64_t,ndim=3]>arr)
    elif arr.dtype == np.int32:
        return argmax2dc(<np.ndarray[np.int32_t,ndim=3]>arr)
    elif arr.dtype == np.int64:
        return argmax2dc(<np.ndarray[np.int64_t,ndim=3]>arr)
    else:return argmaxnd(arr)

import numpy as np
cimport numpy as np
from ..ccfg cimport *

#b means batch in function declaration.

cpdef np.ndarray maxpooling2db(np.ndarray[DTYPE_t, ndim=3] fm, int ksize, int stride=0):
    if stride == 0:
        stride = ksize
    cdef size_t c,i,j,I=0,J=0
    cdef np.ndarray[DTYPE_t,ndim=3] ret = np.zeros((fm.shape[0],fm.shape[1]//ksize,fm.shape[2]//ksize),dtype=fm.dtype)
    for c in range(fm.shape[0]):
        for i in range(0,fm.shape[1],stride):
            for j in range(0,fm.shape[2],stride):
                ret[c,I,J]=np.max(fm[c,i:i+ksize,j:j+ksize])
                J+=1
            I+=1
            J=0
        I=0
    return ret

cpdef np.ndarray bwdmaxpooling2db(np.ndarray[DTYPE_t, ndim=3] fm, np.ndarray[DTYPE_t, ndim=3] grad_output, int ksize, int stride=0):
    if stride == 0:
        stride = ksize
    cdef size_t c,i,j,I=0,J=0
    cdef np.ndarray[DTYPE_t,ndim=3] ret = np.zeros((fm.shape[0],fm.shape[1],fm.shape[2]),dtype=fm.dtype)
    for c in range(fm.shape[0]):
        for i in range(0,fm.shape[1],stride):
            for j in range(0,fm.shape[2],stride):
                ret[c,i:i+ksize,j:j+ksize] = (grad_output[c,I,J] == fm[c,i:i+ksize,j:j+ksize])*grad_output[c,I,J]
                J+=1
            I+=1
        I=0
        J=0
    return ret


cpdef np.ndarray conv2db(np.ndarray[DTYPE_t,ndim=3] fm,np.ndarray[DTYPE_t,ndim=3] krnl,int padx=0,int pady=-2147483648):
    if pady==-2147483648:
        pady=padx
    assert fm.shape[0]==krnl.shape[0]
    cdef size_t output_rows=fm.shape[1]-krnl.shape[1]+2*padx+1
    cdef size_t output_cols=fm.shape[2]-krnl.shape[2]+2*pady+1
    cdef size_t channels = fm.shape[0]

    cdef size_t krnli,krnlj,fmlefttopi,fmlefttopj,c,i,j
    cdef np.ndarray[DTYPE_t,ndim=3] ret = np.zeros((fm.shape[0],output_rows,output_cols),dtype=fm.dtype)
    cdef maxx0=max(padx,0),maxy0=max(pady,0),minx0=min(padx,0),miny0=min(pady,0)
    cdef np.ndarray[DTYPE_t,ndim=3] padfm=np.pad(fm,((0,0),(maxx0,maxx0),(maxy0,maxy0)),'constant')
    padfm=padfm[:,-minx0:minx0+padfm.shape[1],-miny0:miny0+padfm.shape[2]]
    if output_rows*output_cols<krnl.shape[1]*krnl.shape[2]:
        for i in range(output_rows):
            for j in range(output_cols):
                ret[:,i,j]=np.sum(padfm[:,i:i+krnl.shape[1],j:j+krnl.shape[2]]*krnl,axis=(1,2)).reshape(channels,1,1)
    else:
        for krnli in range(krnl.shape[1]):
            for krnlj in range(krnl.shape[2]):
                ret+=padfm[:,krnli:krnli+output_rows,krnlj:krnlj+output_cols]*krnl[:,krnli,krnlj].reshape(channels,1,1)
    return ret
