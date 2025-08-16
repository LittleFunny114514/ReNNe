import importlib
import sys
from sysconfig import get_python_version

IGNORE_PYTHON_VERSION_ISNT_SATISFIED = True


def checkLib(libname: str):
    try:
        importlib.import_module(libname)
        return True
    except ImportError:
        # install the missing library
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", libname])
        return False


def checkPyVer():
    ver = [int(n) for n in get_python_version().split(".")]
    if not IGNORE_PYTHON_VERSION_ISNT_SATISFIED:
        assert (
            ver[0] == 3 and 9 <= ver[1] <= 12
        ), """
Library ReNNe has better running on python 3.9 ~ python 3.12.
You can set IGNORE_PYTHON_VERSION_ISNT_SATISFIED = False to disable this verification
    """


checkPyVer()


def chkmain():
    checkPyVer()
    checkLib("numpy")
    checkLib("matplotlib")
    checkLib("scipy")
    checkLib("numba")
