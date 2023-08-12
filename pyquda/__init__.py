from . import pyquda as quda

try:
    from . import pyqcu as qcu
except ImportError as e:
    print(e)
from .mpi import init


def plaq():
    ret = [0.0, 0.0, 0.0]
    quda.plaqQuda(ret)
    return ret[0]
