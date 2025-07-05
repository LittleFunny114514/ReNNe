from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import scipy
import os, sys

RENNE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

ccomplier_args = ["-O3", "-std=c++14", "/std:c++14", "/O2"]
linker_args = []

PYX_registered_modules = [
    {"name": "ReNNe.ccfg", "sources": [RENNE_ROOT_PATH + "/ccfg.pyx"]},
    {
        "name": "ReNNe.convnet.boost",
        "sources": [
            RENNE_ROOT_PATH + "/convnet/boost.pyx",
        ],
    },
]

PYX_registered_includes = [
    numpy.get_include(),
    # scipy.get_include(),
]


def install():
    if __name__ != "__main__":
        import subprocess

        subprocess.run(
            [sys.executable, RENNE_ROOT_PATH + "/pyxbuild.py", "build_ext", "--inplace"]
        )
        return
    extensions = [
        Extension(
            pack["name"],
            pack["sources"],
            include_dirs=PYX_registered_includes,
            language="c++",
            extra_compile_args=ccomplier_args,
            extra_link_args=linker_args,
        )
        for pack in PYX_registered_modules
    ]
    for ext in extensions:
        setup(name=ext.name, ext_modules=cythonize([ext], annotate=True))


if __name__ == "__main__":
    install()
