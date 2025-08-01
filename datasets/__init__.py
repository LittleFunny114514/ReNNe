from .. import cfg
from .. import chklibs as _

_.checkLib("h5py")
np=cfg.np

def toOneHot(labels: np.ndarray):
    onehot = np.zeros((labels.shape[0], 10), dtype=cfg.dtype)
    for i in range(labels.shape[0]):
        onehot[i][labels[i]] = 1
    return onehot
