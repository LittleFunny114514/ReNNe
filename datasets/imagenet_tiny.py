"""
This is a script to read the subset of imagenet.
It will read the subset of imagenet and save it to a numpy array.
The numpy array will be used to train some models for ILSVRC.
"""

import h5py
from .. import cfg

np = cfg.np
