from . import pyquda as quda
from . import pyqcu as qcu
from .mpi import init


def plaq():
    ret = [0., 0., 0.]
    quda.plaqQuda(ret)
    return ret[0]
