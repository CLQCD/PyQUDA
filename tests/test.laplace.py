from time import perf_counter

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import linalg
from opt_einsum import contract

from check_pyquda import weak_field

from pyquda import init, core, enum_quda, quda
from pyquda.field import LatticeGauge, LaplaceLatticeInfo, LatticeStaggeredFermion, MultiLatticeStaggeredFermion, Nc
from pyquda.utils import io
from pyquda.dirac import setPrecision

init(resource_path=".cache")
setPrecision(eigensolver=8)

t = 3

gauge = io.readChromaQIOGauge(weak_field)
gauge.smearSTOUT(10, 0.12, 3)
Lx, Ly, Lz = gauge.latt_info.Lx, gauge.latt_info.Ly, gauge.latt_info.Lz
latt_info = LaplaceLatticeInfo([Lx, Ly, Lz, 1])
gauge_tmp_lexico = cp.array(gauge.lexico()[:, t])
gauge_tmp_lexico_dagger = gauge_tmp_lexico.transpose(0, 1, 2, 3, 5, 4).conj().copy()
gauge_tmp = LatticeGauge(latt_info, core.cb2(gauge.lexico()[:, t : t + 1], [1, 2, 3, 4]))

Lx, Ly, Lz, _ = latt_info.size
n_ev = 20
n_kr = min(max(2 * n_ev, n_ev + 32), Lz * Ly * Lx * Nc - 1)
tol = 1e-9
max_restarts = 10 * Lz * Ly * Lx * Nc // (n_kr - n_ev)


def Laplacian(x):
    x = x.reshape(Lz * Ly * Lx * Nc, -1)
    ret = cp.zeros_like(x, "<c16")
    for i in range(x.shape[1]):
        ret[:, i] = gauge_tmp.laplace(LatticeStaggeredFermion(latt_info, x[:, i]), 3).data.reshape(Lz * Ly * Lx * Nc)
    return ret


A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Laplacian, matmat=Laplacian)
s = perf_counter()
evals, evecs = linalg.eigsh(A, n_ev, which="SA", tol=tol)
print(f"{perf_counter() - s:.3f} secs")
print(evals)


def _Laplacian(x):
    x = x.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        x
        - (1 / 6)
        * (
            contract("zyxab,zyxbc->zyxac", gauge_tmp_lexico[0], cp.roll(x, -1, 2))
            + contract("zyxab,zyxbc->zyxac", gauge_tmp_lexico[1], cp.roll(x, -1, 1))
            + contract("zyxab,zyxbc->zyxac", gauge_tmp_lexico[2], cp.roll(x, -1, 0))
            + cp.roll(contract("zyxab,zyxbc->zyxac", gauge_tmp_lexico_dagger[0], x), 1, 2)
            + cp.roll(contract("zyxab,zyxbc->zyxac", gauge_tmp_lexico_dagger[1], x), 1, 1)
            + cp.roll(contract("zyxab,zyxbc->zyxac", gauge_tmp_lexico_dagger[2], x), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=_Laplacian, matmat=_Laplacian)
s = perf_counter()
evals, evecs = linalg.eigsh(A, n_ev, which="SA", tol=tol)
print(f"{perf_counter() - s:.3f} secs")
print(evals)

gauge_tmp.pure_gauge.loadGauge(gauge_tmp)
eig_param = quda.QudaEigParam()
eig_param.invert_param = gauge_tmp.pure_gauge.invert_param
eig_param.eig_type = enum_quda.QudaEigType.QUDA_EIG_TR_LANCZOS
eig_param.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
eig_param.n_ev = n_ev
eig_param.n_kr = n_kr
eig_param.n_conv = n_ev
eig_param.tol = tol
eig_param.vec_infile = b""
eig_param.vec_outfile = b""
eig_param.max_restarts = max_restarts

evecs = MultiLatticeStaggeredFermion(latt_info, n_ev)
evals = np.zeros((n_ev), "<c16")
s = perf_counter()
quda.eigensolveQuda(evecs.data_ptrs, evals, eig_param)
print(f"{perf_counter() - s:.3f} secs")
print(evals.real)
