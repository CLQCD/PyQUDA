import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, mpi, quda
from pyquda.enum_quda import QudaDslashType
from pyquda.utils import io, source

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
mpi.init([1, 1, 1, 1])


def covdev(U, x, mu):
    x.toHost()
    U_ = core.lexico(U.data, [1, 2, 3, 4, 5])
    x_ = core.lexico(x.data, [0, 1, 2, 3, 4])
    if 0 <= mu <= 3:
        x_ = np.einsum("tzyxab,tzyxib->tzyxia", U_[mu], np.roll(x_, -1, 3 - mu))
    elif 4 <= mu <= 7:
        mu -= 4
        x_ = np.einsum("tzyxba,tzyxib->tzyxia", U_[mu].conj(), np.roll(x_, 1, 3 - mu))
    x.data = core.cb2(x_, [0, 1, 2, 3])
    x.toDevice()


covdev_mu = 3

dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
dslash.invert_param.dslash_type = QudaDslashType.QUDA_COVDEV_DSLASH
dslash.invert_param.covdev_mu = covdev_mu
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))

dslash.loadGauge(gauge)

x = source.wall(latt_size, 0, 0, 0)
b = core.LatticeFermion(latt_size)
quda.MatQuda(b.data_ptr, x.data_ptr, dslash.invert_param)
covdev(gauge, x, covdev_mu)
print(cp.linalg.norm(x.data - b.data))

dslash.destroy()
