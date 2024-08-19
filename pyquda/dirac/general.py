from typing import List

import numpy
from numpy.typing import NDArray

from ..pointer import Pointer, Pointers
from ..pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    loadCloverQuda,
    loadGaugeQuda,
    invertQuda,
    MatQuda,
    MatDagMatQuda,
    dslashQuda,
    cloverQuda,
    computeKSLinkQuda,
)
from ..field import (
    LatticeInfo,
    LatticeGauge,
    LatticeClover,
    LatticeFermion,
    LatticeStaggeredFermion,
)
from ..enum_quda import (  # noqa: F401
    QUDA_MAX_DIM,
    QUDA_MAX_MULTI_SHIFT,
    QUDA_MAX_MG_LEVEL,
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

nullptr = Pointer("void")
nullptrs = Pointers("void", 0)

from . import Precision, Reconstruct


def _fieldLocation():
    from .. import getCUDABackend

    if getCUDABackend() == "numpy":
        return QudaFieldLocation.QUDA_CPU_FIELD_LOCATION
    else:
        return QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION


def _useMMA():
    from .. import isHIP, getCUDAComputeCapability

    return QudaBoolean(not isHIP() and getCUDAComputeCapability().major >= 7)


def newQudaGaugeParam(
    lattice: LatticeInfo,
    tadpole_coeff: float,
    naik_epsilon: float,
    precision: Precision,
    reconstruct: Reconstruct,
):
    gauge_param = QudaGaugeParam()

    gauge_param.X = lattice.size

    gauge_param.anisotropy = lattice.anisotropy
    gauge_param.tadpole_coeff = tadpole_coeff
    gauge_param.scale = -(1 + naik_epsilon) / (24 * tadpole_coeff * tadpole_coeff)
    gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
    gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
    gauge_param.t_boundary = lattice.t_boundary

    gauge_param.cpu_prec = precision.cpu
    gauge_param.cuda_prec = precision.cuda
    gauge_param.cuda_prec_sloppy = precision.sloppy
    gauge_param.cuda_prec_refinement_sloppy = precision.sloppy
    gauge_param.cuda_prec_precondition = precision.precondition
    gauge_param.cuda_prec_eigensolver = precision.eigensolver

    gauge_param.reconstruct = reconstruct.cuda
    gauge_param.reconstruct_sloppy = reconstruct.sloppy
    gauge_param.reconstruct_refinement_sloppy = reconstruct.sloppy
    gauge_param.reconstruct_precondition = reconstruct.precondition
    gauge_param.reconstruct_eigensolver = reconstruct.eigensolver

    gauge_param.gauge_fix = QudaGaugeFixed.QUDA_GAUGE_FIXED_NO
    gauge_param.ga_pad = lattice.ga_pad

    gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_MILC
    gauge_param.staggered_phase_applied = 0

    gauge_param.overwrite_gauge = 0
    gauge_param.overwrite_mom = 0
    gauge_param.use_resident_gauge = 1
    gauge_param.use_resident_mom = 1
    gauge_param.make_resident_gauge = 1
    gauge_param.make_resident_mom = 1
    gauge_param.return_result_gauge = 0
    gauge_param.return_result_mom = 0

    return gauge_param


def newQudaMultigridParam(
    mass: float,
    kappa: float,
    geo_block_size: List[List[int]],
    coarse_tol: float,
    coarse_maxiter: int,
    setup_tol: float,
    setup_maxiter: int,
    nu_pre: int,
    nu_post: int,
    precision: Precision,
):
    mg_param = QudaMultigridParam()
    mg_inv_param = QudaInvertParam()

    # mg_inv_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
    mg_inv_param.mass = mass
    mg_inv_param.kappa = kappa
    mg_inv_param.m5 = 0.0
    mg_inv_param.Ls = 1

    mg_inv_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
    mg_inv_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    mg_inv_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
    mg_inv_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD
    mg_inv_param.dagger = QudaDagType.QUDA_DAG_NO
    mg_inv_param.mass_normalization = QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION
    mg_inv_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
    mg_inv_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO

    mg_inv_param.tol = 0
    mg_inv_param.maxiter = 0
    mg_inv_param.reliable_delta = 1e-5
    mg_inv_param.gcrNkrylov = 8
    mg_inv_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO

    location: QudaFieldLocation = _fieldLocation()
    mg_inv_param.cpu_prec = precision.cpu
    mg_inv_param.cuda_prec = precision.cuda
    mg_inv_param.cuda_prec_sloppy = precision.sloppy
    mg_inv_param.cuda_prec_precondition = precision.precondition
    mg_inv_param.input_location = location
    mg_inv_param.output_location = location
    mg_inv_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    mg_inv_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    mg_inv_param.clover_cpu_prec = precision.cpu
    mg_inv_param.clover_cuda_prec = precision.cuda
    mg_inv_param.clover_cuda_prec_sloppy = precision.sloppy
    mg_inv_param.clover_cuda_prec_precondition = precision.precondition
    mg_inv_param.clover_location = location
    mg_inv_param.clover_order = QudaCloverFieldOrder.QUDA_PACKED_CLOVER_ORDER
    mg_inv_param.clover_coeff = 1.0

    mg_inv_param.tune = QudaTune.QUDA_TUNE_YES
    mg_inv_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE
    mg_inv_param.verbosity_precondition = QudaVerbosity.QUDA_SILENT

    mg_param.invert_param = mg_inv_param

    n_level = len(geo_block_size)
    for i in range(n_level):
        geo_block_size[i] = geo_block_size[i] + [1] * (QUDA_MAX_DIM - len(geo_block_size[i]))
    mg_param.n_level = n_level
    mg_param.geo_block_size = geo_block_size
    mg_param.spin_block_size = [2] + [1] * (QUDA_MAX_MG_LEVEL - 1)
    mg_param.n_vec = [24] * QUDA_MAX_MG_LEVEL
    mg_param.precision_null = [precision.precondition] * QUDA_MAX_MG_LEVEL
    mg_param.n_block_ortho = [1] * QUDA_MAX_MG_LEVEL

    mg_param.setup_inv_type = [QudaInverterType.QUDA_CGNR_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.num_setup_iter = [1] * QUDA_MAX_MG_LEVEL
    mg_param.setup_tol = [setup_tol] * QUDA_MAX_MG_LEVEL
    mg_param.setup_maxiter = [setup_maxiter] * QUDA_MAX_MG_LEVEL
    mg_param.setup_maxiter_refresh = [setup_maxiter // 10] * QUDA_MAX_MG_LEVEL
    mg_param.setup_type = QudaSetupType.QUDA_NULL_VECTOR_SETUP
    mg_param.pre_orthonormalize = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.post_orthonormalize = QudaBoolean.QUDA_BOOLEAN_TRUE

    mg_param.coarse_solver = [QudaInverterType.QUDA_GCR_INVERTER] * (n_level - 1) + [
        QudaInverterType.QUDA_CA_GCR_INVERTER
    ] * (QUDA_MAX_MG_LEVEL - n_level + 1)
    mg_param.coarse_solver_tol = [coarse_tol] * QUDA_MAX_MG_LEVEL
    mg_param.coarse_solver_maxiter = [coarse_maxiter] * QUDA_MAX_MG_LEVEL
    mg_param.transfer_type = [QudaTransferType.QUDA_TRANSFER_AGGREGATE] * QUDA_MAX_MG_LEVEL

    mg_param.smoother = [QudaInverterType.QUDA_CA_GCR_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_tol = [0.25] * QUDA_MAX_MG_LEVEL
    mg_param.nu_pre = [nu_pre] * QUDA_MAX_MG_LEVEL
    mg_param.nu_post = [nu_post] * QUDA_MAX_MG_LEVEL
    mg_param.omega = [1.0] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_schwarz_type = [QudaSchwarzType.QUDA_INVALID_SCHWARZ] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_schwarz_cycle = [1] * QUDA_MAX_MG_LEVEL

    mg_param.coarse_grid_solution_type = [QudaSolutionType.QUDA_MATPC_SOLUTION] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_solve_type = [QudaSolveType.QUDA_DIRECT_PC_SOLVE] * QUDA_MAX_MG_LEVEL
    mg_param.cycle_type = [QudaMultigridCycleType.QUDA_MG_CYCLE_RECURSIVE] * QUDA_MAX_MG_LEVEL
    mg_param.global_reduction = [QudaBoolean.QUDA_BOOLEAN_TRUE] * QUDA_MAX_MG_LEVEL

    mg_param.mu_factor = [1.0] * QUDA_MAX_MG_LEVEL

    mg_param.location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * QUDA_MAX_MG_LEVEL
    mg_param.setup_location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * QUDA_MAX_MG_LEVEL
    mg_param.setup_minimize_memory = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.compute_null_vector = QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES
    mg_param.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_TRUE
    mg_param.run_verify = QudaBoolean.QUDA_BOOLEAN_TRUE
    mg_param.run_low_mode_check = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.run_oblique_proj_check = QudaBoolean.QUDA_BOOLEAN_FALSE

    use_mma: QudaBoolean = _useMMA()
    mg_param.setup_use_mma = [use_mma] * QUDA_MAX_MG_LEVEL
    mg_param.dslash_use_mma = [use_mma] * QUDA_MAX_MG_LEVEL

    mg_param.verbosity = [QudaVerbosity.QUDA_SILENT] * QUDA_MAX_MG_LEVEL

    return mg_param, mg_inv_param


def newQudaInvertParam(
    mass: float,
    kappa: float,
    tol: float,
    maxiter: int,
    clover_coeff: float,
    clover_anisotropy: float,
    mg_param: QudaMultigridParam,
    precision: Precision,
):
    invert_param = QudaInvertParam()

    # invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
    invert_param.mass = mass
    invert_param.kappa = kappa
    invert_param.m5 = 0.0
    invert_param.Ls = 1
    invert_param.laplace3D = 3

    invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
    invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
    invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD
    invert_param.dagger = QudaDagType.QUDA_DAG_NO
    invert_param.mass_normalization = QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION
    invert_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
    invert_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO

    invert_param.tol = tol
    invert_param.tol_restart = 5e3 * tol
    invert_param.tol_hq = tol
    invert_param.residual_type = QudaResidualType.QUDA_L2_RELATIVE_RESIDUAL
    invert_param.maxiter = maxiter
    invert_param.reliable_delta = 1e-1 if mg_param is None else 1e-5
    # invert_param.use_alternative_reliable = 0
    # invert_param.use_sloppy_partial_accumulator = 0
    # invert_param.solution_accumulator_pipeline = 0
    # invert_param.max_res_increase = 1
    invert_param.pipeline = 0
    invert_param.Nsteps = 2
    invert_param.gcrNkrylov = 8
    invert_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO

    location: QudaFieldLocation = _fieldLocation()
    invert_param.cpu_prec = precision.cpu
    invert_param.cuda_prec = precision.cuda
    invert_param.cuda_prec_sloppy = precision.sloppy
    invert_param.cuda_prec_refinement_sloppy = precision.sloppy
    invert_param.cuda_prec_precondition = precision.precondition
    invert_param.cuda_prec_eigensolver = precision.eigensolver
    invert_param.input_location = location
    invert_param.output_location = location
    invert_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    invert_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    if clover_coeff != 0.0:
        invert_param.clover_cpu_prec = precision.cpu
        invert_param.clover_cuda_prec = precision.cuda
        invert_param.clover_cuda_prec_sloppy = precision.sloppy
        invert_param.clover_cuda_prec_refinement_sloppy = precision.sloppy
        invert_param.clover_cuda_prec_precondition = precision.precondition
        invert_param.clover_cuda_prec_eigensolver = precision.eigensolver
        invert_param.clover_location = location
        invert_param.clover_order = QudaCloverFieldOrder.QUDA_PACKED_CLOVER_ORDER
        invert_param.clover_csw = clover_anisotropy  # to save clover_anisotropy, not real csw
        invert_param.clover_coeff = clover_coeff
        invert_param.compute_clover = 1
        invert_param.compute_clover_inverse = 1
        invert_param.return_clover = 0
        invert_param.return_clover_inverse = 0

    # invert_param.num_offset = 1
    invert_param.tol_offset = [invert_param.tol] * QUDA_MAX_MULTI_SHIFT
    invert_param.tol_hq_offset = [invert_param.tol_hq] * QUDA_MAX_MULTI_SHIFT

    if mg_param is not None:
        invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
        invert_param.inv_type_precondition = QudaInverterType.QUDA_MG_INVERTER
        invert_param.tol_precondition = mg_param.coarse_solver_tol[0]
        invert_param.maxiter_precondition = mg_param.coarse_solver_maxiter[0]
        invert_param.verbosity_precondition = mg_param.verbosity[0]
        invert_param.schwarz_type = QudaSchwarzType.QUDA_ADDITIVE_SCHWARZ
        invert_param.precondition_cycle = 1
        invert_param.omega = 1.0

    invert_param.tune = QudaTune.QUDA_TUNE_YES
    invert_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE

    return invert_param


def loadClover(
    clover: LatticeClover,
    clover_inv: LatticeClover,
    gauge: LatticeGauge,
    gauge_param: QudaGaugeParam,
    invert_param: QudaInvertParam,
):
    if clover is None or clover_inv is None:
        clover_anisotropy = invert_param.clover_csw
        t_boundary = gauge_param.t_boundary
        anisotropy = gauge_param.anisotropy
        reconstruct = gauge_param.reconstruct

        gauge_in = gauge.copy()
        if clover_anisotropy != 1.0:
            gauge_in.setAnisotropy(clover_anisotropy)
            gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
        gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        gauge_param.anisotropy = 1.0
        gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, gauge_param)
        if clover_anisotropy != 1.0:
            gauge_param.reconstruct = reconstruct
        gauge_param.t_boundary = t_boundary
        gauge_param.anisotropy = anisotropy
        gauge_param.use_resident_gauge = 1
        loadCloverQuda(nullptr, nullptr, invert_param)
    else:
        invert_param.compute_clover = 0
        invert_param.compute_clover_inverse = 0
        loadCloverQuda(clover.data_ptr, clover_inv.data_ptr, invert_param)
        invert_param.compute_clover = 1
        invert_param.compute_clover_inverse = 1


def saveClover(
    clover: LatticeClover,
    clover_inv: LatticeClover,
    gauge: LatticeGauge,
    gauge_param: QudaGaugeParam,
    invert_param: QudaInvertParam,
):
    clover_anisotropy = invert_param.clover_csw
    t_boundary = gauge_param.t_boundary
    anisotropy = gauge_param.anisotropy
    reconstruct = gauge_param.reconstruct

    gauge_in = gauge.copy()
    if clover_anisotropy != 1.0:
        gauge_in.setAnisotropy(clover_anisotropy)
        gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
    gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
    gauge_param.anisotropy = 1.0
    gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge_in.data_ptrs, gauge_param)
    if clover_anisotropy != 1.0:
        gauge_param.reconstruct = reconstruct
    gauge_param.t_boundary = t_boundary
    gauge_param.anisotropy = anisotropy
    gauge_param.use_resident_gauge = 1
    invert_param.return_clover = 1
    invert_param.return_clover_inverse = 1
    loadCloverQuda(clover.data_ptr, clover_inv.data_ptr, invert_param)
    invert_param.return_clover = 0
    invert_param.return_clover_inverse = 0


def loadGauge(gauge: LatticeGauge, gauge_param: QudaGaugeParam):
    gauge_in = gauge.copy()
    if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
        gauge_in.setAntiPeriodicT()
    if gauge_param.anisotropy != 1.0:
        gauge_in.setAnisotropy(gauge_param.anisotropy)
    gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge_in.data_ptrs, gauge_param)
    gauge_param.use_resident_gauge = 1


def loadFatLongGauge(
    gauge: LatticeGauge,
    fat7_coeff: NDArray[numpy.float64],
    level2_coeff: NDArray[numpy.float64],
    gauge_param: QudaGaugeParam,
):
    staggered_phase_type = gauge_param.staggered_phase_type

    inlink = gauge.copy()
    ulink = LatticeGauge(gauge.latt_info)
    fatlink = LatticeGauge(gauge.latt_info)
    longlink = LatticeGauge(gauge.latt_info)

    inlink.staggeredPhase()
    computeKSLinkQuda(nullptrs, nullptrs, ulink.data_ptrs, inlink.data_ptrs, fat7_coeff, gauge_param)
    computeKSLinkQuda(fatlink.data_ptrs, longlink.data_ptrs, nullptrs, ulink.data_ptrs, level2_coeff, gauge_param)

    gauge_param.use_resident_gauge = 0
    gauge_param.type = QudaLinkType.QUDA_ASQTAD_FAT_LINKS
    loadGaugeQuda(fatlink.data_ptrs, gauge_param)
    gauge_param.type = QudaLinkType.QUDA_ASQTAD_LONG_LINKS
    gauge_param.ga_pad = gauge_param.ga_pad * 3
    gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_NO
    loadGaugeQuda(longlink.data_ptrs, gauge_param)
    gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
    gauge_param.ga_pad = gauge_param.ga_pad // 3
    gauge_param.staggered_phase_type = staggered_phase_type
    gauge_param.use_resident_gauge = 1


def performance(invert_param: QudaInvertParam):
    from .. import getLogger

    gflops, secs = invert_param.gflops, invert_param.secs
    getLogger().info(f"Time = {secs:.3f} secs, Performance = {gflops / secs:.3f} GFLOPS")


def invert(b: LatticeFermion, invert_param: QudaInvertParam):
    x = LatticeFermion(b.latt_info)
    invertQuda(x.data_ptr, b.data_ptr, invert_param)
    performance(invert_param)
    return x


def invertStaggered(b: LatticeStaggeredFermion, invert_param: QudaInvertParam):
    x = LatticeStaggeredFermion(b.latt_info)
    invertQuda(x.data_ptr, b.data_ptr, invert_param)
    performance(invert_param)
    return x


def mat(x: LatticeFermion, invert_param: QudaInvertParam):
    b = LatticeFermion(x.latt_info)
    MatQuda(b.data_ptr, x.data_ptr, invert_param)
    return b


def matStaggered(x: LatticeStaggeredFermion, invert_param: QudaInvertParam):
    b = LatticeStaggeredFermion(x.latt_info)
    MatQuda(b.data_ptr, x.data_ptr, invert_param)
    return b


def matDagMat(x: LatticeFermion, invert_param: QudaInvertParam):
    b = LatticeFermion(x.latt_info)
    MatDagMatQuda(b.data_ptr, x.data_ptr, invert_param)
    return b


def matDagMatStaggered(x: LatticeStaggeredFermion, invert_param: QudaInvertParam):
    b = LatticeStaggeredFermion(x.latt_info)
    MatDagMatQuda(b.data_ptr, x.data_ptr, invert_param)
    return b


def invertPC(b: LatticeFermion, invert_param: QudaInvertParam):
    kappa = invert_param.kappa
    invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION

    latt_info = b.latt_info
    x = LatticeFermion(latt_info)

    dslashQuda(x.odd_ptr, b.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
    x.even = b.odd + kappa * x.odd
    # * QUDA_ASYMMETRIC_MASS_NORMALIZATION makes the even part 1 / (2 * kappa) instead of 1
    invertQuda(x.odd_ptr, x.even_ptr, invert_param)
    performance(invert_param)
    dslashQuda(x.even_ptr, x.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
    x.even = kappa * (2 * b.even + x.even)

    return x


def invertCloverPC(b: LatticeFermion, invert_param: QudaInvertParam):
    tmp = LatticeFermion(b.latt_info)
    cloverQuda(tmp.even_ptr, b.even_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY, 1)
    cloverQuda(tmp.odd_ptr, b.odd_ptr, invert_param, QudaParity.QUDA_ODD_PARITY, 1)
    return invertPC(tmp, invert_param)


def invertStaggeredPC(b: LatticeStaggeredFermion, invert_param: QudaInvertParam):
    mass = invert_param.mass
    invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION

    latt_info = b.latt_info
    x = LatticeStaggeredFermion(latt_info)

    dslashQuda(x.odd_ptr, b.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
    x.even = (2 * mass) * b.odd + x.odd
    invertQuda(x.odd_ptr, x.even_ptr, invert_param)
    performance(invert_param)
    dslashQuda(x.even_ptr, x.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
    x.even = (0.5 / mass) * (b.even + x.even)

    return x
