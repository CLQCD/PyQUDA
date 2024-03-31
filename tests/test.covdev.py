import numpy as np
import cupy as cp

from check_pyquda import weak_field

from pyquda import init, core
from pyquda.enum_quda import QudaDslashType
from pyquda.utils import io, source

init([1, 1, 1, 1], [4, 4, 4, 8], 1, 1.0, resource_path=".cache")
latt_info = core.getDefaultLattice()


def covdev(U, x, mu):
    U_ = U.lexico()
    x_ = x.lexico()
    if 0 <= mu <= 3:
        x_ = np.einsum("tzyxab,tzyxib->tzyxia", U_[mu], np.roll(x_, -1, 3 - mu))
    elif 4 <= mu <= 7:
        mu -= 4
        x_ = np.roll(np.einsum("tzyxba,tzyxib->tzyxia", U_[mu].conj(), x_), 1, 3 - mu)
    x.setData(core.cb2(x_, [0, 1, 2, 3]))
    x.toDevice()


gauge = io.readQIOGauge(weak_field)
dirac = core.getDefaultDirac(-3, 0, 0)
dirac.invert_param.dslash_type = QudaDslashType.QUDA_COVDEV_DSLASH
dirac.loadGauge(gauge)

for covdev_mu in range(8):
    dirac.invert_param.covdev_mu = covdev_mu
    x = source.wall(latt_info, 0, 0, 0)
    b = dirac.mat(x)
    covdev(gauge, x, covdev_mu)

    print(covdev_mu, cp.linalg.norm(x.data - b.data))
