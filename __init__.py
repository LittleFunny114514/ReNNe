from . import chklibs as _
from . import cfg

_.chkmain()
from . import pyxbuild

if cfg.PYX_BOOST and cfg.BUILD_PYX:
    pyxbuild.install()
from .base import *
from .misc import *
from . import convnet, moduleblock, learning


import sys
import gc

sys.setrecursionlimit(cfg.MAX_PYCALLSTACK_SIZE)
gc.enable()

if __name__ != "__main__":
    print("Module ReNNe has imported successfully")
else:
    with open("README.txt", "r") as f:
        print(f.read())
