import numpy as np
import cupy as cp

from check_pyquda import weak_field

from pyquda import init, core
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
    x.data = core.cb2(x_, [0, 1, 2, 3])
    x.toDevice()


gauge = io.readQIOGauge(weak_field)
gauge.loadCovDev()

for covdev_mu in range(8):
    x = source.wall(latt_info, 0, 0, 0)
    b = gauge.covDev(x, covdev_mu)
    covdev(gauge, x, covdev_mu)

    print(covdev_mu, cp.linalg.norm(x.data - b.data))
