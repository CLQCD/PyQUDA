import os
import numpy as np
from pyquda import quda
from pyquda.enum_quda import (
    QudaDagType, QudaDiracFieldOrder, QudaDslashType, QudaGammaBasis, QudaGaugeFieldOrder, QudaGaugeFixed, QudaInverterType, QudaLinkType,
    QudaMassNormalization, QudaMatPCType, QudaParity, QudaPrecision, QudaPreserveSource, QudaReconstructType, QudaSolutionType, QudaSolveType, QudaTboundary,
    QudaTune, QudaUseInitGuess, QudaVerbosity
)

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

gauge_param = quda.QudaGaugeParam()
quda_inv_param = quda.QudaInvertParam()

gauge_param.X = [16, 16, 16, 16]
gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
gauge_param.t_boundary = QudaTboundary.QUDA_ANTI_PERIODIC_T
gauge_param.cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
gauge_param.cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
gauge_param.cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
gauge_param.reconstruct_sloppy = QudaReconstructType.QUDA_RECONSTRUCT_12
gauge_param.gauge_fix = QudaGaugeFixed.QUDA_GAUGE_FIXED_NO
gauge_param.anisotropy = 1.0
gauge_param.ga_pad = 2048

quda_inv_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
quda_inv_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
quda_inv_param.kappa = 0.1
quda_inv_param.clover_coeff = 0.0

quda_inv_param.tol = 1e-7
quda_inv_param.maxiter = 1000
quda_inv_param.reliable_delta = 0.001
quda_inv_param.pipeline = 0

quda_inv_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
quda_inv_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
quda_inv_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD
quda_inv_param.dagger = QudaDagType.QUDA_DAG_NO
quda_inv_param.mass_normalization = QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION

quda_inv_param.cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
quda_inv_param.cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
quda_inv_param.cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
quda_inv_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_YES
quda_inv_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
quda_inv_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
quda_inv_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

quda_inv_param.tune = QudaTune.QUDA_TUNE_YES

quda_inv_param.inv_type_precondition = QudaInverterType.QUDA_INVALID_INVERTER
quda_inv_param.tol_precondition = 1.0e-1
quda_inv_param.maxiter_precondition = 1000
quda_inv_param.verbosity_precondition = QudaVerbosity.QUDA_SILENT
quda_inv_param.gcrNkrylov = 1

quda_inv_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE

quda_inv_param.sp_pad = 0
quda_inv_param.cl_pad = 0

quda.initQuda(0)

gauge = np.identity(3, "<c16").reshape(1, -1).repeat(4 * 16**4, 0).view("<f8").reshape(4, -1)
if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
    gauge = gauge.reshape(4, 2, 16, -1)
    gauge[3, :, -1] *= -1
    gauge = gauge.reshape(4, -1)
quda.loadGaugeQuda(quda.getPointerArray(gauge), gauge_param)

spin = 0
color = 2

x = np.zeros((2, 16**4 // 2 * 4 * 3 * 2))
b = np.zeros((2, 16**4 // 2 * 4 * 3 * 2))
tmp = np.zeros((2, 16**4 // 2 * 4 * 3 * 2))

b = b.reshape(2, 16**4 // 2, 4, 3, 2)
b[0, 0, spin, color, 0] = 1
b = b.reshape(2, -1)

quda.dslashQuda(quda.OddPointer(tmp), quda.EvenPointer(b), quda_inv_param, QudaParity.QUDA_ODD_PARITY)
b[1] += quda_inv_param.kappa * tmp[1]

quda.invertQuda(quda.OddPointer(x), quda.OddPointer(b), quda_inv_param)
x[0] = 2 * quda_inv_param.kappa * b[0]

quda.dslashQuda(quda.EvenPointer(tmp), quda.OddPointer(x), quda_inv_param, QudaParity.QUDA_EVEN_PARITY)
x[0] += quda_inv_param.kappa * tmp[0]

quda.endQuda()

# x2 = np.fromfile("sh_prop_1", ">f8", offset=8).reshape(16**4, 4, 4, 3, 3, 2)[:, spin, :, color, :].reshape(2, -1)
# print(x.reshape(2, 16**4 // 2, 4, 3, 2)[0, :, spin, color])
# print(x2.reshape(2, 16**4 // 2, 4, 3, 2)[0, :, spin, color])
# print(np.linalg.norm(x - x2))
