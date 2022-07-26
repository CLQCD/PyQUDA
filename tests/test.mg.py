import os
from math import sqrt
from typing import List

import numpy as np
import cupy as cp

from pyquda import mpi
from pyquda import quda, core
from pyquda.core import Nd, Nc, Ns
from pyquda.enum_quda import (
    QudaBoolean, QudaCABasis, QudaCloverFieldOrder, QudaComputeNullVector, QudaDagType, QudaDiracFieldOrder,
    QudaDslashType, QudaFieldLocation, QudaGammaBasis, QudaGaugeFieldOrder, QudaGaugeFixed, QudaInverterType,
    QudaLinkType, QudaMassNormalization, QudaMatPCType, QudaMultigridCycleType, QudaParity, QudaPrecision,
    QudaPreserveSource, QudaReconstructType, QudaResidualType, QudaSchwarzType, QudaSetupType, QudaSolutionType,
    QudaSolveType, QudaSolverNormalization, QudaTboundary, QudaTransferType, QudaTune, QudaUseInitGuess, QudaVerbosity
)
from pyquda.utils import source, gauge_utils, prop_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [16, 16, 16, 128]
grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

# mpi.init(grid_size)

xi_0, nu = 4.8965, 0.86679
mass = 0.09253
coeff = 1.17
coeff_r, coeff_t = 2.32582045, 0.8549165664

Lx, Ly, Lz, Lt = latt_size
xi = xi_0 / nu
kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
if xi != 1.0:
    clover_coeff = xi_0 * coeff_t**2 / coeff_r
    clover_xi = sqrt(xi_0 * coeff_t / coeff_r)
else:
    clover_coeff = coeff_t
    clover_xi = 1.0

###############################################################################

cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
gpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
gpu_half_prec = QudaPrecision.QUDA_SINGLE_PRECISION

coarse_solve_type = [QudaSolveType.QUDA_NORMOP_PC_SOLVE] * 5
smoother_solve_type = [QudaSolveType.QUDA_DIRECT_PC_SOLVE] * 5

###############################################################################

mg_invert_param = quda.QudaInvertParam()

mg_invert_param.sp_pad = 0
mg_invert_param.cl_pad = 0
mg_invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
mg_invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
mg_invert_param.tol = 1e-10
mg_invert_param.maxiter = 10000
mg_invert_param.reliable_delta = 1e-10
mg_invert_param.cpu_prec = cpu_prec
mg_invert_param.cuda_prec = gpu_prec
mg_invert_param.cuda_prec_sloppy = gpu_half_prec
mg_invert_param.cuda_prec_precondition = gpu_half_prec
mg_invert_param.clover_cpu_prec = cpu_prec
mg_invert_param.clover_cuda_prec = gpu_prec
mg_invert_param.clover_cuda_prec_sloppy = gpu_half_prec
mg_invert_param.clover_cuda_prec_precondition = gpu_half_prec
mg_invert_param.clover_order = QudaCloverFieldOrder.QUDA_FLOAT2_CLOVER_ORDER
mg_invert_param.tune = QudaTune.QUDA_TUNE_YES
mg_invert_param.gcrNkrylov = 10
mg_invert_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE
mg_invert_param.verbosity_precondition = QudaVerbosity.QUDA_SUMMARIZE
mg_invert_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO
mg_invert_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS
mg_invert_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
mg_invert_param.input_location = QudaFieldLocation.QUDA_CPU_FIELD_LOCATION
mg_invert_param.output_location = QudaFieldLocation.QUDA_CPU_FIELD_LOCATION
mg_invert_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
mg_invert_param.dagger = QudaDagType.QUDA_DAG_NO
mg_invert_param.kappa = kappa
mg_invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
mg_invert_param.clover_csw = clover_xi
mg_invert_param.clover_coeff = kappa * clover_coeff
mg_invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
mg_invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
mg_invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
mg_invert_param.Ls = 1

###############################################################################

mg_param = quda.QudaMultigridParam()

mg_param.invert_param = mg_invert_param
mg_param.n_level = 2
mg_param.run_verify = QudaBoolean.QUDA_BOOLEAN_TRUE
mg_param.setup_location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * 5
mg_param.geo_block_size = [[4, 4, 4, 4, 1, 1], [2, 2, 2, 2, 1, 1]]
mg_param.spin_block_size = [2, 1, 1, 1, 1]
mg_param.n_vec = [24] * 5
mg_param.nu_pre = [0] * 5
mg_param.nu_post = [8] * 5
mg_param.mu_factor = [1.] * 5
mg_param.setup_inv_type = [QudaInverterType.QUDA_CG_INVERTER] * 5
mg_param.setup_tol = [5e-6] * 5
mg_param.setup_maxiter = [5000] * 5
mg_param.setup_maxiter_refresh = [5000] * 5
mg_param.num_setup_iter = [1] * 5
mg_param.precision_null = [gpu_half_prec] * 5
mg_param.coarse_solver = [QudaInverterType.QUDA_GCR_INVERTER] * 5
mg_param.coarse_solver_tol = [1e-1] * 5
mg_param.coarse_solver_maxiter = [12] * 5
mg_param.smoother = [
    QudaInverterType.QUDA_MR_INVERTER, QudaInverterType.QUDA_GCR_INVERTER, QudaInverterType.QUDA_MR_INVERTER,
    QudaInverterType.QUDA_MR_INVERTER, QudaInverterType.QUDA_MR_INVERTER
]
mg_param.smoother_tol = [0.25] * 5
mg_param.smoother_solve_type = [QudaSolveType.QUDA_DIRECT_PC_SOLVE] * 5
mg_param.omega = [1.] * 5
# mg_param.smoother_schwarz_type = [QudaSchwarzType.QUDA_ADDITIVE_SCHWARZ] * 5
mg_param.smoother_schwarz_cycle = [1] * 5
mg_param.global_reduction = [QudaBoolean.QUDA_BOOLEAN_FALSE] * 5
mg_param.location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * 5
mg_param.cycle_type = [QudaMultigridCycleType.QUDA_MG_CYCLE_RECURSIVE] * 5
mg_param.coarse_grid_solution_type = [QudaSolutionType.QUDA_MATPC_SOLUTION] * 5
mg_param.setup_type = QudaSetupType.QUDA_NULL_VECTOR_SETUP
mg_param.pre_orthonormalize = QudaBoolean.QUDA_BOOLEAN_FALSE
mg_param.post_orthonormalize = QudaBoolean.QUDA_BOOLEAN_TRUE
mg_param.compute_null_vector = QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES
mg_param.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_TRUE
mg_param.verbosity = [QudaVerbosity.QUDA_SILENT] * 5
mg_param.use_mma = QudaBoolean.QUDA_BOOLEAN_TRUE

###############################################################################

invert_param = quda.QudaInvertParam()

invert_param.sp_pad = 0
invert_param.cl_pad = 0
invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
invert_param.kappa = kappa
invert_param.clover_csw = clover_xi
invert_param.clover_coeff = kappa * clover_coeff
invert_param.Ls = 1
invert_param.tol = 1e-12
invert_param.maxiter = 5000
invert_param.reliable_delta = 1e-1
invert_param.pipeline = 0
invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
invert_param.dagger = QudaDagType.QUDA_DAG_NO
invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
invert_param.cpu_prec = cpu_prec
invert_param.cuda_prec = gpu_prec
invert_param.cuda_prec_sloppy = gpu_half_prec
invert_param.cuda_prec_precondition = gpu_half_prec
invert_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO
invert_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS
invert_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
invert_param.input_location = QudaFieldLocation.QUDA_CPU_FIELD_LOCATION
invert_param.output_location = QudaFieldLocation.QUDA_CPU_FIELD_LOCATION
invert_param.tune = QudaTune.QUDA_TUNE_YES
invert_param.clover_cpu_prec = cpu_prec
invert_param.clover_cuda_prec = gpu_prec
invert_param.clover_cuda_prec_sloppy = gpu_half_prec
invert_param.clover_cuda_prec_precondition = gpu_half_prec
invert_param.tol_precondition = mg_param.coarse_solver_tol[0]
invert_param.maxiter_precondition = mg_param.coarse_solver_maxiter[0]
invert_param.gcrNkrylov = mg_invert_param.gcrNkrylov
invert_param.residual_type = QudaResidualType.QUDA_L2_RELATIVE_RESIDUAL
invert_param.schwarz_type = QudaSchwarzType.QUDA_ADDITIVE_SCHWARZ
invert_param.precondition_cycle = 1
invert_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE
invert_param.verbosity_precondition = mg_param.verbosity[0]
invert_param.inv_type_precondition = QudaInverterType.QUDA_MG_INVERTER
invert_param.clover_order = QudaCloverFieldOrder.QUDA_FLOAT2_CLOVER_ORDER
invert_param.clover_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION
invert_param.omega = 1.0

###############################################################################

Lmin = min(Lx, Ly, Lz, Lt)
ga_pad = Lx * Ly * Lz * Lt // Lmin

gauge_param = quda.QudaGaugeParam()

gauge_param.X = latt_size
gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
gauge_param.t_boundary = QudaTboundary.QUDA_ANTI_PERIODIC_T
gauge_param.cpu_prec = cpu_prec
gauge_param.cuda_prec = gpu_prec
gauge_param.cuda_prec_sloppy = gpu_half_prec
gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
gauge_param.reconstruct_sloppy = QudaReconstructType.QUDA_RECONSTRUCT_NO
gauge_param.gauge_fix = QudaGaugeFixed.QUDA_GAUGE_FIXED_NO
gauge_param.anisotropy = xi
gauge_param.ga_pad = ga_pad

###############################################################################

gauge = gauge_utils.readIldg(
    "/hpcfs/lqcd/qcd/gongming/productions/confs/light.20200720.b20.16_128/s1.0_cfg_1000.lime", grid_size
)

quda.initQuda(0)

###############################################################################

gauge_data_bak = gauge.data.copy()
if clover_xi != 1.0:
    gauge.setAnisotropy(clover_xi)
gauge_param.anisotropy = 1.0
quda.loadGaugeQuda(gauge.data_ptrs, gauge_param)
quda.loadCloverQuda(quda.Pointer("void"), quda.Pointer("void"), invert_param)
gauge_param.anisotropy = xi
gauge.data = gauge_data_bak.copy()
if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
    gauge.setAntiPeroidicT()
if xi != 1.0:
    gauge.setAnisotropy(xi)
quda.loadGaugeQuda(gauge.data_ptrs, gauge_param)
gauge.data = gauge_data_bak

###############################################################################

mg_instance = quda.newMultigridQuda(mg_param)
invert_param.preconditioner = mg_instance

###############################################################################

b12 = source.source12(latt_size, "point", [0, 0, 0, 0])
x12 = core.LatticePropagator(latt_size)
b = core.LatticeFermion(latt_size)
x = core.LatticeFermion(latt_size)
for spin in range(Ns):
    for color in range(Nc):
        data = b.data.reshape(Vol, Ns, Nc)
        data[:] = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)[:, :, spin, :, color]
        quda.invertQuda(x.data_ptr, b.data_ptr, invert_param)
        data = x12.data.reshape(Vol, Ns, Ns, Nc, Nc)
        data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)

###############################################################################

quda.destroyMultigridQuda(mg_instance)

###############################################################################

quda.endQuda()

propagator_all = prop_utils.collect(x12, grid_size)

if mpi.rank == 0:
    propagator_chroma = cp.array(np.fromfile("pt_prop_2", ">c16", offset=8).astype("<c16"))
    print(cp.linalg.norm(propagator_all.transpose() - propagator_chroma / (2 * kappa)))
