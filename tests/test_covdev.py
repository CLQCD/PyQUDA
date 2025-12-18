import numpy as np
import cupy as cp

from check_pyquda import weak_field

from pyquda.field import LatticeGauge, LatticeFermion, Nd
from pyquda_utils import core, io, source

core.init([1, 1, 1, 1], resource_path=".cache")
latt_info = core.LatticeInfo([4, 4, 4, 8], 1, 1.0)


def covdev(U: LatticeGauge, x: LatticeFermion, mu: int):
    U_ = U.lexico()
    x_ = x.lexico()
    if 0 <= mu <= 3:
        x_ = np.einsum("tzyxab,tzyxib->tzyxia", U_[mu], np.roll(x_, -1, 3 - mu))
    elif 4 <= mu <= 7:
        x_ = np.roll(np.einsum("tzyxba,tzyxib->tzyxia", U_[mu - 4].conj(), x_), 1, 7 - mu)
    x.data = U.latt_info.evenodd(x_, False)
    x.toDevice()


gauge = io.readQIOGauge(weak_field)

x = source.wall(latt_info, 0, 0, 0)
for covdev_mu in range(8):
    b = gauge.covDev(x, covdev_mu)
    covdev(gauge, x, covdev_mu)

    print(covdev_mu, cp.linalg.norm(x.data - b.data))


def shift(U: LatticeGauge, dim: int, mu: int):
    U_ = U.lexico()[dim]
    if 0 <= mu <= 3:
        U_ = np.roll(U_, -1, 3 - mu)
    elif 4 <= mu <= 7:
        U_ = np.roll(U_, 1, 7 - mu)
    U.data[dim] = cp.asarray(U.latt_info.evenodd(U_, False))


unit = LatticeGauge(latt_info)

gauge.toDevice()
gauge2 = gauge.copy()
with unit.use() as dirac:
    x = LatticeFermion(latt_info)
    for dim in range(Nd):
        for covdev_mu in range(8):
            gauge3_dim = gauge2[dim].shift(1, covdev_mu)

            gauge2.pack(dim, x)
            gauge2.unpack(dim, dirac.covDev(x, covdev_mu))
            gauge2.projectSU3(2e-15)

            shift(gauge, dim, covdev_mu)

            print(dim, covdev_mu, cp.linalg.norm(gauge.data[dim] - gauge2.data[dim]))
            print(dim, covdev_mu, cp.linalg.norm(gauge2.data[dim] - gauge3_dim.data))
