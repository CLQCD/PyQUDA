from . import pyquda as quda
from .mpi import init


def plaq():
    ret = [0., 0., 0.]
    quda.plaqQuda(ret)
    return ret[0]
