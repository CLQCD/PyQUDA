from math import sqrt
import os
import numpy as np
from pyquda import quda, LatticeGauge, LatticeFermion
from pyquda.enum_quda import (
    QudaCloverFieldOrder, QudaDagType, QudaDiracFieldOrder, QudaDslashType, QudaGammaBasis, QudaGaugeFieldOrder,
    QudaGaugeFixed, QudaInverterType, QudaLinkType, QudaMassNormalization, QudaMatPCType, QudaParity, QudaPrecision,
    QudaPreserveSource, QudaReconstructType, QudaSolutionType, QudaSolveType, QudaTboundary, QudaTune,
    QudaUseInitGuess, QudaVerbosity
)

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nc, Ns, Nd = 3, 4, 4
Lx, Ly, Lz, Lt = 4, 4, 4, 8
xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07


def read_gauge(filename: str):
    gauge_raw = np.fromfile(filename, ">c16").astype("<c16").reshape(Lt, Lz, Ly, Lx, Nd, Nc, Nc)
    gauge = np.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")

    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    gauge[:, 0, t, z, y, :, :, :] = gauge_raw[t, z, y, 0::2, :, :, :].transpose(1, 0, 2, 3)
                    gauge[:, 1, t, z, y, :, :, :] = gauge_raw[t, z, y, 1::2, :, :, :].transpose(1, 0, 2, 3)
                else:
                    gauge[:, 0, t, z, y, :, :, :] = gauge_raw[t, z, y, 1::2, :, :, :].transpose(1, 0, 2, 3)
                    gauge[:, 1, t, z, y, :, :, :] = gauge_raw[t, z, y, 0::2, :, :, :].transpose(1, 0, 2, 3)

    return gauge.reshape(-1).view("<f8")


xi = xi_0 / nu
if xi != 1.0:
    kappa = 1 / (2 * (mass + 1 + 3 / xi))
    coeff = xi_0 * coeff_t**2 / coeff_r
    xi_clover = sqrt(xi_0 * coeff_t / coeff_r)
else:
    kappa = kappa
    coeff = coeff
    xi_clover = 1.0

gauge_param = quda.QudaGaugeParam()
quda_inv_param = quda.QudaInvertParam()

gauge_param.X = [Lx, Ly, Lz, Lt]
gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
gauge_param.t_boundary = QudaTboundary.QUDA_ANTI_PERIODIC_T
gauge_param.cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
gauge_param.cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
gauge_param.cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
gauge_param.reconstruct_sloppy = QudaReconstructType.QUDA_RECONSTRUCT_12
gauge_param.reconstruct_precondition = QudaReconstructType.QUDA_RECONSTRUCT_NO
gauge_param.gauge_fix = QudaGaugeFixed.QUDA_GAUGE_FIXED_NO
gauge_param.anisotropy = xi
gauge_param.ga_pad = 2048

quda_inv_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
quda_inv_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
quda_inv_param.mass = mass + 1 + (Nd - 1) / xi
quda_inv_param.kappa = kappa
quda_inv_param.clover_coeff = kappa * coeff
quda_inv_param.tol = 1e-8
quda_inv_param.maxiter = 1000
quda_inv_param.reliable_delta = 0.001
quda_inv_param.pipeline = 0

quda_inv_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
quda_inv_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
quda_inv_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD

quda_inv_param.dagger = QudaDagType.QUDA_DAG_NO
quda_inv_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION

quda_inv_param.cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
quda_inv_param.cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
quda_inv_param.cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
quda_inv_param.cuda_prec_precondition = QudaPrecision.QUDA_HALF_PRECISION
quda_inv_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_YES
quda_inv_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
quda_inv_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
quda_inv_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

quda_inv_param.clover_cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
quda_inv_param.clover_cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
quda_inv_param.clover_cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
quda_inv_param.clover_cuda_prec_precondition = QudaPrecision.QUDA_HALF_PRECISION

quda_inv_param.clover_order = QudaCloverFieldOrder.QUDA_PACKED_CLOVER_ORDER

quda_inv_param.tune = QudaTune.QUDA_TUNE_YES

quda_inv_param.inv_type_precondition = QudaInverterType.QUDA_INVALID_INVERTER
quda_inv_param.tol_precondition = 1.0e-1
quda_inv_param.maxiter_precondition = 1000
quda_inv_param.verbosity_precondition = QudaVerbosity.QUDA_SILENT
quda_inv_param.gcrNkrylov = 1

quda_inv_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE

quda_inv_param.sp_pad = 0
quda_inv_param.cl_pad = 0

Vol = Lx * Ly * Lz * Lt
Sx, Sy, Sz, St = 0, 0, 0, 0
Seo = (Sx + Sy + Sz + St) % 2

quda.initQuda(0)

gauge = LatticeGauge([Lx, Ly, Lz, Lt], read_gauge("weak_field"))
gauge_clover = LatticeGauge([Lx, Ly, Lz, Lt], read_gauge("weak_field"))
if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
    gauge.setAntiPeroidicT()
    gauge_clover.setAntiPeroidicT()

gauge_param.anisotropy = 1.0
quda_inv_param.compute_clover = 1
quda_inv_param.compute_clover_inverse = 1
data = gauge_clover.data.reshape(Nd, -1)
data[:3] /= xi_clover
quda.loadGaugeQuda(gauge_clover.ptr, gauge_param)
quda.loadCloverQuda(quda.Pointer("void"), quda.Pointer("void"), quda_inv_param)

gauge_param.anisotropy = xi
data = gauge.data.reshape(Nd, -1)
data[:3] /= xi
quda.loadGaugeQuda(gauge.ptr, gauge_param)

propagator = np.zeros((Vol, Ns, Ns, Nc, Nc), "<c16")
for spin in range(Ns):
    for color in range(Nc):
        x = LatticeFermion([Lx, Ly, Lz, Lt])
        b = LatticeFermion([Lx, Ly, Lz, Lt])
        tmp = LatticeFermion([Lx, Ly, Lz, Lt])
        tmp2 = LatticeFermion([Lx, Ly, Lz, Lt])

        data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
        data[Seo, St, Sz, Sy, Sx // 2, spin, color] = 1

        quda.cloverQuda(tmp.even_ptr, b.even_ptr, quda_inv_param, QudaParity.QUDA_EVEN_PARITY, 1)
        quda.cloverQuda(tmp.odd_ptr, b.odd_ptr, quda_inv_param, QudaParity.QUDA_ODD_PARITY, 1)
        tmp.data *= 2 * kappa
        quda.dslashQuda(tmp2.odd_ptr, tmp.even_ptr, quda_inv_param, QudaParity.QUDA_ODD_PARITY)
        tmp2.odd *= -kappa
        tmp.odd = tmp.odd - tmp2.odd
        quda.invertQuda(x.odd_ptr, tmp.odd_ptr, quda_inv_param)
        quda.dslashQuda(tmp2.even_ptr, x.odd_ptr, quda_inv_param, QudaParity.QUDA_EVEN_PARITY)
        tmp2.even *= -kappa
        x.even = tmp.even - tmp2.even

        propagator[:, spin, :, color, :] = x.data.reshape(Vol, Ns, Nc).get()

quda.endQuda()

propagator_chroma = np.fromfile("pt_prop_1", ">c16", offset=8).reshape(Vol, Ns, Ns, Nc, Nc)
print(np.linalg.norm(propagator - propagator_chroma))
