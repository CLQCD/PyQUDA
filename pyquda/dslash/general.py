from typing import List

from ..pointer import Pointer
from ..pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    loadCloverQuda,
    loadGaugeQuda,
    invertQuda,
    dslashQuda,
    cloverQuda,
)
from ..field import LatticeGauge, LatticeFermion
from ..enum_quda import (  # noqa: F401
    QudaMemoryType,
    QudaLinkType,
    QudaGaugeFieldOrder,
    QudaTboundary,
    QudaPrecision,
    QudaReconstructType,
    QudaGaugeFixed,
    QudaDslashType,
    QudaInverterType,
    QudaEigType,
    QudaEigSpectrumType,
    QudaSolutionType,
    QudaSolveType,
    QudaMultigridCycleType,
    QudaSchwarzType,
    QudaResidualType,
    QudaCABasis,
    QudaMatPCType,
    QudaDagType,
    QudaMassNormalization,
    QudaSolverNormalization,
    QudaPreserveSource,
    QudaDiracFieldOrder,
    QudaCloverFieldOrder,
    QudaVerbosity,
    QudaTune,
    QudaPreserveDirac,
    QudaParity,
    QudaDiracType,
    QudaFieldLocation,
    QudaSiteSubset,
    QudaSiteOrder,
    QudaFieldOrder,
    QudaFieldCreate,
    QudaGammaBasis,
    QudaSourceType,
    QudaNoiseType,
    QudaProjectionType,
    QudaPCType,
    QudaTwistFlavorType,
    QudaTwistDslashType,
    QudaTwistCloverDslashType,
    QudaTwistGamma5Type,
    QudaUseInitGuess,
    QudaDeflatedGuess,
    QudaComputeNullVector,
    QudaSetupType,
    QudaTransferType,
    QudaBoolean,
    QudaBLASOperation,
    QudaBLASDataType,
    QudaBLASDataOrder,
    QudaDirection,
    QudaLinkDirection,
    QudaFieldGeometry,
    QudaGhostExchange,
    QudaStaggeredPhase,
    QudaContractType,
    QudaContractGamma,
    QudaExtLibType,
)
from ..enum_quda import QUDA_MAX_DIM, QUDA_MAX_MULTI_SHIFT, QUDA_MAX_MG_LEVEL

nullptr = Pointer("void")

cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
cuda_prec_precondition = QudaPrecision.QUDA_HALF_PRECISION
cuda_prec_eigensolver = QudaPrecision.QUDA_SINGLE_PRECISION
link_recon = QudaReconstructType.QUDA_RECONSTRUCT_12
link_recon_sloppy = QudaReconstructType.QUDA_RECONSTRUCT_12


def newQudaGaugeParam(X: List[int], anisotropy: float, t_boundary: int):
    gauge_param = QudaGaugeParam()

    gauge_param.X = X

    gauge_param.anisotropy = anisotropy
    gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
    gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
    gauge_param.t_boundary = QudaTboundary.QUDA_ANTI_PERIODIC_T if t_boundary == -1 else QudaTboundary.QUDA_PERIODIC_T

    gauge_param.cpu_prec = cpu_prec
    gauge_param.cuda_prec = cuda_prec
    gauge_param.cuda_prec_sloppy = cuda_prec_sloppy
    gauge_param.cuda_prec_refinement_sloppy = cuda_prec_sloppy
    gauge_param.cuda_prec_precondition = cuda_prec_precondition
    gauge_param.cuda_prec_eigensolver = cuda_prec_eigensolver

    gauge_param.reconstruct = link_recon
    gauge_param.reconstruct_sloppy = link_recon_sloppy
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy
    gauge_param.reconstruct_precondition = link_recon_sloppy
    gauge_param.reconstruct_eigensolver = link_recon_sloppy

    Lx, Ly, Lz, Lt = X
    ga_pad = Lx * Ly * Lz * Lt // min(Lx, Ly, Lz, Lt) // 2
    gauge_param.ga_pad = ga_pad
    gauge_param.gauge_fix = QudaGaugeFixed.QUDA_GAUGE_FIXED_NO

    return gauge_param


def newQudaMultigridParam(
    kappa: float,
    geo_block_size: List[List[int]],
    coarse_tol: float,
    coarse_maxiter: int,
    setup_tol: float,
    setup_maxiter: int,
    nu_pre: int,
    nu_post: int,
):
    mg_param = QudaMultigridParam()
    mg_inv_param = QudaInvertParam()

    mg_inv_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
    mg_inv_param.kappa = kappa
    mg_inv_param.Ls = 1
    mg_inv_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    mg_inv_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
    mg_inv_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
    mg_inv_param.dagger = QudaDagType.QUDA_DAG_NO
    mg_inv_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
    mg_inv_param.gcrNkrylov = 12
    mg_inv_param.tol = 1e-10
    mg_inv_param.maxiter = 10000
    mg_inv_param.reliable_delta = 1e-10

    mg_inv_param.cpu_prec = cpu_prec
    mg_inv_param.cuda_prec = cuda_prec
    mg_inv_param.cuda_prec_sloppy = cuda_prec_sloppy
    mg_inv_param.cuda_prec_precondition = cuda_prec_precondition
    mg_inv_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO
    mg_inv_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
    mg_inv_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    mg_inv_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    mg_inv_param.clover_cpu_prec = cpu_prec
    mg_inv_param.clover_cuda_prec = cuda_prec
    mg_inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy
    mg_inv_param.clover_cuda_prec_precondition = cuda_prec_precondition
    mg_inv_param.clover_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION
    mg_inv_param.clover_order = QudaCloverFieldOrder.QUDA_FLOAT2_CLOVER_ORDER
    mg_inv_param.clover_coeff = 1.0

    mg_inv_param.input_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION
    mg_inv_param.output_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION

    mg_inv_param.tune = QudaBoolean.QUDA_BOOLEAN_TRUE
    mg_inv_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE
    mg_inv_param.verbosity_precondition = QudaVerbosity.QUDA_SILENT

    mg_param.invert_param = mg_inv_param

    n_level = len(geo_block_size)
    for i in range(n_level):
        geo_block_size[i] = geo_block_size[i] + [1] * (QUDA_MAX_DIM - len(geo_block_size[i]))
    mg_param.n_level = n_level
    mg_param.geo_block_size = geo_block_size
    mg_param.setup_inv_type = [QudaInverterType.QUDA_CG_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.num_setup_iter = [1] * QUDA_MAX_MG_LEVEL
    mg_param.setup_tol = [setup_tol] * QUDA_MAX_MG_LEVEL
    mg_param.setup_maxiter = [setup_maxiter] * QUDA_MAX_MG_LEVEL
    mg_param.setup_maxiter_refresh = [setup_maxiter // 5] * QUDA_MAX_MG_LEVEL

    mg_param.spin_block_size = [2] + [1] * (QUDA_MAX_MG_LEVEL - 1)
    mg_param.n_vec = [24] * QUDA_MAX_MG_LEVEL
    mg_param.n_block_ortho = [1] * QUDA_MAX_MG_LEVEL
    mg_param.precision_null = [cuda_prec_precondition] * QUDA_MAX_MG_LEVEL
    mg_param.nu_pre = [nu_pre] * QUDA_MAX_MG_LEVEL
    mg_param.nu_post = [nu_post] * QUDA_MAX_MG_LEVEL
    mg_param.mu_factor = [1.0] * QUDA_MAX_MG_LEVEL

    mg_param.cycle_type = [QudaMultigridCycleType.QUDA_MG_CYCLE_RECURSIVE] * QUDA_MAX_MG_LEVEL
    mg_param.transfer_type = [QudaTransferType.QUDA_TRANSFER_AGGREGATE] * QUDA_MAX_MG_LEVEL

    mg_param.coarse_solver = [QudaInverterType.QUDA_GCR_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.coarse_solver_tol = [coarse_tol] * QUDA_MAX_MG_LEVEL
    mg_param.coarse_solver_maxiter = [coarse_maxiter] * QUDA_MAX_MG_LEVEL
    mg_param.coarse_grid_solution_type = [QudaSolutionType.QUDA_MATPC_SOLUTION] * QUDA_MAX_MG_LEVEL

    mg_param.smoother = [QudaInverterType.QUDA_MR_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_tol = [0.25] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_solve_type = [QudaSolveType.QUDA_DIRECT_PC_SOLVE] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_schwarz_type = [QudaSchwarzType.QUDA_INVALID_SCHWARZ] * QUDA_MAX_MG_LEVEL
    mg_param.global_reduction = [QudaBoolean.QUDA_BOOLEAN_TRUE] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_schwarz_cycle = [1] * QUDA_MAX_MG_LEVEL

    mg_param.omega = [1.0] * QUDA_MAX_MG_LEVEL

    mg_param.location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * QUDA_MAX_MG_LEVEL
    mg_param.setup_location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * QUDA_MAX_MG_LEVEL

    mg_param.verbosity = [QudaVerbosity.QUDA_SILENT] * QUDA_MAX_MG_LEVEL

    mg_param.setup_minimize_memory = QudaBoolean.QUDA_BOOLEAN_FALSE

    mg_param.setup_type = QudaSetupType.QUDA_NULL_VECTOR_SETUP
    mg_param.pre_orthonormalize = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.post_orthonormalize = QudaBoolean.QUDA_BOOLEAN_TRUE

    mg_param.compute_null_vector = QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES
    mg_param.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_TRUE

    mg_param.run_verify = QudaBoolean.QUDA_BOOLEAN_TRUE
    mg_param.run_low_mode_check = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.run_oblique_proj_check = QudaBoolean.QUDA_BOOLEAN_FALSE

    mg_param.use_mma = QudaBoolean.QUDA_BOOLEAN_TRUE

    return mg_param, mg_inv_param


def newQudaInvertParam(
    kappa: float,
    tol: float,
    maxiter: int,
    clover_coeff: float,
    clover_anisotropy: float,
    mg_param: QudaMultigridParam = None,
):
    invert_param = QudaInvertParam()

    # invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
    # invert_param.mass = 0.5 / kappa - (1.0 + 3.0 / anisotropy)
    invert_param.kappa = kappa

    invert_param.laplace3D = 3

    invert_param.Ls = 1

    # invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
    invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    # invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
    invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
    invert_param.dagger = QudaDagType.QUDA_DAG_NO
    invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
    invert_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
    invert_param.pipeline = 0
    invert_param.Nsteps = 2
    invert_param.gcrNkrylov = 10
    invert_param.tol = tol
    invert_param.tol_restart = 5e3 * tol
    invert_param.tol_hq = tol
    invert_param.residual_type = QudaResidualType.QUDA_L2_RELATIVE_RESIDUAL
    invert_param.maxiter = maxiter
    invert_param.reliable_delta = 0.1
    # invert_param.use_alternative_reliable = 0
    # invert_param.use_sloppy_partial_accumulator = 0
    # invert_param.solution_accumulator_pipeline = 0
    # invert_param.max_res_increase = 1

    invert_param.cpu_prec = cpu_prec
    invert_param.cuda_prec = cuda_prec
    invert_param.cuda_prec_sloppy = cuda_prec_sloppy
    invert_param.cuda_prec_refinement_sloppy = cuda_prec_sloppy
    invert_param.cuda_prec_precondition = cuda_prec_precondition
    invert_param.cuda_prec_eigensolver = cuda_prec_eigensolver
    invert_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO
    invert_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
    invert_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    invert_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    if clover_coeff != 0.0:
        invert_param.clover_cpu_prec = cpu_prec
        invert_param.clover_cuda_prec = cuda_prec
        invert_param.clover_cuda_prec_sloppy = cuda_prec_sloppy
        invert_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy
        invert_param.clover_cuda_prec_precondition = cuda_prec_precondition
        invert_param.clover_cuda_prec_eigensolver = cuda_prec_eigensolver
        invert_param.clover_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION
        invert_param.clover_order = QudaCloverFieldOrder.QUDA_FLOAT2_CLOVER_ORDER
        invert_param.clover_csw = clover_anisotropy  # to save clover_anisotropy, not real csw
        invert_param.clover_coeff = clover_coeff
        invert_param.compute_clover = 1
        invert_param.compute_clover_inverse = 1

    if False:
        invert_param.num_offset = 1
        invert_param.tol_offset = [invert_param.tol] * QUDA_MAX_MULTI_SHIFT
        invert_param.tol_hq_offset = [invert_param.tol_hq] * QUDA_MAX_MULTI_SHIFT

    if mg_param is not None:
        invert_param.inv_type_precondition = QudaInverterType.QUDA_MG_INVERTER
        invert_param.schwarz_type = QudaSchwarzType.QUDA_ADDITIVE_SCHWARZ
        invert_param.precondition_cycle = 1
        invert_param.tol_precondition = mg_param.coarse_solver_tol[0]
        invert_param.maxiter_precondition = mg_param.coarse_solver_maxiter[0]
        invert_param.verbosity_precondition = mg_param.verbosity[0]
        invert_param.omega = 1.0

    invert_param.input_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION
    invert_param.output_location = QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION

    invert_param.tune = QudaTune.QUDA_TUNE_YES
    invert_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE

    return invert_param


def loadClover(gauge: LatticeGauge, gauge_param: QudaGaugeParam, invert_param: QudaInvertParam):
    clover_anisotropy = invert_param.clover_csw
    anisotropy = gauge_param.anisotropy
    reconstruct = gauge_param.reconstruct
    use_resident_gauge = gauge_param.use_resident_gauge

    gauge_data_bak = gauge.backup()
    if clover_anisotropy != 1.0:
        gauge.setAnisotropy(clover_anisotropy)
    gauge_param.anisotropy = 1.0
    gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
    gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge.data_ptrs, gauge_param)
    gauge_param.use_resident_gauge = use_resident_gauge
    loadCloverQuda(nullptr, nullptr, invert_param)
    gauge_param.anisotropy = anisotropy
    gauge_param.reconstruct = reconstruct
    gauge.data = gauge_data_bak


def loadGauge(gauge: LatticeGauge, gauge_param: QudaGaugeParam):
    anisotropy = gauge_param.anisotropy
    use_resident_gauge = gauge_param.use_resident_gauge

    gauge_data_bak = gauge.backup()
    if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
        gauge.setAntiPeroidicT()
    if anisotropy != 1.0:
        gauge.setAnisotropy(anisotropy)
    gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge.data_ptrs, gauge_param)
    gauge_param.use_resident_gauge = use_resident_gauge
    gauge.data = gauge_data_bak


def invert(b: LatticeFermion, invert_param: QudaInvertParam):
    kappa = invert_param.kappa

    x = LatticeFermion(b.latt_size)

    invertQuda(x.data_ptr, b.data_ptr, invert_param)
    x.data *= 2 * kappa

    return x


def invertPC(b: LatticeFermion, invert_param: QudaInvertParam):
    invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
    invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD

    kappa = invert_param.kappa

    x = LatticeFermion(b.latt_size)
    tmp = LatticeFermion(b.latt_size)

    # tmp.data = 2 * kappa * b.data
    # dslashQuda(x.odd_ptr, tmp.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
    # tmp.odd = tmp.odd + kappa * x.odd
    # invertQuda(x.odd_ptr, tmp.odd_ptr, invert_param)
    # dslashQuda(x.even_ptr, x.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
    # x.even = tmp.even + kappa * x.even

    cloverQuda(tmp.even_ptr, b.even_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY, 1)
    cloverQuda(tmp.odd_ptr, b.odd_ptr, invert_param, QudaParity.QUDA_ODD_PARITY, 1)
    dslashQuda(x.odd_ptr, tmp.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
    tmp.odd = tmp.odd + kappa * x.odd
    invertQuda(x.odd_ptr, tmp.odd_ptr, invert_param)
    dslashQuda(x.even_ptr, x.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
    x.even = tmp.even + kappa * x.even
    x.data *= 2 * kappa

    return x
