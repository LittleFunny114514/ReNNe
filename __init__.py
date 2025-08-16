from . import chklibs as _

__version__=['alpha',2,2,1,'beta',3]

_.chkmain()
'''
# deprecated, now we're using numba.
from . import pyxbuild

if cfg.PYX_BOOST and cfg.BUILD_PYX:
    pyxbuild.install()
'''

from . import cfg
from .base import *
from .misc import *
from . import conv, moduleblock, learning


import sys
import gc

sys.setrecursionlimit(cfg.MAX_PYCALLSTACK_SIZE)
gc.enable()

if __name__ != "__main__":
    print("Module ReNNe has imported successfully")
else:
    with open("README.txt", "r") as f:
        print(f.read())
