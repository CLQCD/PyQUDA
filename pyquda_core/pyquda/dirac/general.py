from typing import List, Literal, NamedTuple, Optional

import numpy
from numpy.typing import NDArray

from pyquda_comm import getArrayBackendTarget
from pyquda_comm.pointer import Pointer
from ..field import (
    LatticeInfo,
    LatticeGauge,
    LatticeClover,
)
from ..pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    QudaEigParam,
    loadGaugeQuda,
    freeUniqueGaugeQuda,
    loadCloverQuda,
    freeCloverQuda,
    computeKSLinkQuda,
    staggeredPhaseQuda,
    newMultigridQuda,
    updateMultigridQuda,
    destroyMultigridQuda,
    newDeflationQuda,
    destroyDeflationQuda,
)
from ..enum_quda import (
    QUDA_MAX_DIM,
    QUDA_MAX_MG_LEVEL,
    QudaLinkType,
    QudaGaugeFieldOrder,
    QudaTboundary,
    QudaPrecision,
    QudaReconstructType,
    QudaGaugeFixed,
    QudaDslashType,
    QudaInverterType,
    QudaSolutionType,
    QudaSolveType,
    QudaMultigridCycleType,
    QudaSchwarzType,
    QudaResidualType,
    QudaMatPCType,
    QudaDagType,
    QudaMassNormalization,
    QudaSolverNormalization,
    QudaPreserveSource,
    QudaDiracFieldOrder,
    QudaCloverFieldOrder,
    QudaVerbosity,
    QudaFieldLocation,
    QudaGammaBasis,
    QudaUseInitGuess,
    QudaComputeNullVector,
    QudaSetupType,
    QudaTransferType,
    QudaBoolean,
    QudaStaggeredPhase,
)
from ..quda_define import mmaAvailable

nullptr = numpy.empty((0), "<c16")
nullptrs = numpy.empty((0, 0), "<c16")


class Precision(NamedTuple):
    cpu: QudaPrecision
    cuda: QudaPrecision
    sloppy: QudaPrecision
    precondition: QudaPrecision
    eigensolver: QudaPrecision


class Reconstruct(NamedTuple):
    cuda: QudaReconstructType
    sloppy: QudaReconstructType
    precondition: QudaReconstructType
    eigensolver: QudaReconstructType


_precision = {
    "none": Precision(
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
    ),
    "invert": Precision(
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_HALF_PRECISION,
        QudaPrecision.QUDA_HALF_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
    ),
    "multigrid": Precision(
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
        QudaPrecision.QUDA_SINGLE_PRECISION,
        QudaPrecision.QUDA_HALF_PRECISION,
        QudaPrecision.QUDA_DOUBLE_PRECISION,
    ),
}
_reconstruct = {
    "none": Reconstruct(
        QudaReconstructType.QUDA_RECONSTRUCT_NO,
        QudaReconstructType.QUDA_RECONSTRUCT_NO,
        QudaReconstructType.QUDA_RECONSTRUCT_NO,
        QudaReconstructType.QUDA_RECONSTRUCT_NO,
    ),
    "wilson": Reconstruct(
        QudaReconstructType.QUDA_RECONSTRUCT_12,
        QudaReconstructType.QUDA_RECONSTRUCT_8,
        QudaReconstructType.QUDA_RECONSTRUCT_8,
        QudaReconstructType.QUDA_RECONSTRUCT_12,
    ),
    "staggered": Reconstruct(
        QudaReconstructType.QUDA_RECONSTRUCT_13,
        QudaReconstructType.QUDA_RECONSTRUCT_9,
        QudaReconstructType.QUDA_RECONSTRUCT_9,
        QudaReconstructType.QUDA_RECONSTRUCT_13,
    ),
}


def getGlobalPrecision(key: Literal["none", "invert", "multigrid"]):
    return _precision[key]


def setGlobalPrecision(
    key: Literal["none", "invert", "multigrid"],
    *,
    cuda: Optional[QudaPrecision] = None,
    sloppy: Optional[QudaPrecision] = None,
    precondition: Optional[QudaPrecision] = None,
    eigensolver: Optional[QudaPrecision] = None,
):
    global _precision
    precision = _precision[key]
    _precision[key] = Precision(
        precision.cpu,
        cuda if cuda is not None else precision.cuda,
        sloppy if sloppy is not None else precision.sloppy,
        precondition if precondition is not None else precision.precondition,
        eigensolver if eigensolver is not None else precision.eigensolver,
    )


def getGlobalReconstruct(key: Literal["none", "wilson", "staggered"]):
    return _reconstruct[key]


def setGlobalReconstruct(
    key: Literal["none", "wilson", "staggered"],
    *,
    cuda: Optional[QudaReconstructType] = None,
    sloppy: Optional[QudaReconstructType] = None,
    precondition: Optional[QudaReconstructType] = None,
    eigensolver: Optional[QudaReconstructType] = None,
):
    global _reconstruct
    reconstruct = _reconstruct[key]
    _reconstruct[key] = Reconstruct(
        cuda if cuda is not None else reconstruct.cuda,
        sloppy if sloppy is not None else reconstruct.sloppy,
        precondition if precondition is not None else reconstruct.precondition,
        eigensolver if eigensolver is not None else reconstruct.eigensolver,
    )


def setPrecisionParam(
    precision: Precision,
    gauge_param: Optional[QudaGaugeParam] = None,
    invert_param: Optional[QudaInvertParam] = None,
    mg_param: Optional[QudaMultigridParam] = None,
    mg_inv_param: Optional[QudaInvertParam] = None,
):
    if gauge_param is not None:
        gauge_param.cpu_prec = precision.cpu
        gauge_param.cuda_prec = precision.cuda
        gauge_param.cuda_prec_sloppy = precision.sloppy
        gauge_param.cuda_prec_refinement_sloppy = precision.sloppy
        gauge_param.cuda_prec_precondition = precision.precondition
        gauge_param.cuda_prec_eigensolver = precision.eigensolver

    if invert_param is not None:
        invert_param.cpu_prec = precision.cpu
        invert_param.cuda_prec = precision.cuda
        invert_param.cuda_prec_sloppy = precision.sloppy
        invert_param.cuda_prec_refinement_sloppy = precision.sloppy
        invert_param.cuda_prec_precondition = precision.precondition
        invert_param.cuda_prec_eigensolver = precision.eigensolver

        if invert_param.clover_coeff != 0.0:
            invert_param.clover_cpu_prec = precision.cpu
            invert_param.clover_cuda_prec = precision.cuda
            invert_param.clover_cuda_prec_sloppy = precision.sloppy
            invert_param.clover_cuda_prec_refinement_sloppy = precision.sloppy
            invert_param.clover_cuda_prec_precondition = precision.precondition
            invert_param.clover_cuda_prec_eigensolver = precision.eigensolver

    if mg_inv_param is not None:
        mg_inv_param.cpu_prec = precision.cpu
        mg_inv_param.cuda_prec = precision.cuda
        mg_inv_param.cuda_prec_sloppy = precision.sloppy
        mg_inv_param.cuda_prec_precondition = precision.precondition
        mg_inv_param.cuda_prec_eigensolver = precision.eigensolver

        mg_inv_param.clover_cpu_prec = precision.cpu
        mg_inv_param.clover_cuda_prec = precision.cuda
        mg_inv_param.clover_cuda_prec_sloppy = precision.sloppy
        mg_inv_param.clover_cuda_prec_precondition = precision.precondition
        mg_inv_param.clover_cuda_prec_eigensolver = precision.eigensolver

    if mg_param is not None:
        mg_param.precision_null = [precision.precondition] * QUDA_MAX_MG_LEVEL


def setReconstructParam(reconstruct: Reconstruct, gauge_param: Optional[QudaGaugeParam] = None):
    if gauge_param is not None:
        gauge_param.reconstruct = reconstruct.cuda
        gauge_param.reconstruct_sloppy = reconstruct.sloppy
        gauge_param.reconstruct_refinement_sloppy = reconstruct.sloppy
        gauge_param.reconstruct_precondition = reconstruct.precondition
        gauge_param.reconstruct_eigensolver = reconstruct.eigensolver


def fieldLocation() -> QudaFieldLocation:
    if getArrayBackendTarget() == "cpu":
        return QudaFieldLocation.QUDA_CPU_FIELD_LOCATION
    else:
        return QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION


def newQudaGaugeParam(lattice: LatticeInfo):
    gauge_param = QudaGaugeParam()

    gauge_param.X = lattice.size

    gauge_param.anisotropy = lattice.anisotropy
    gauge_param.tadpole_coeff = 1.0  # tadpole_coeff
    gauge_param.scale = 1.0  # -(1 + naik_epsilon) / (24 * tadpole_coeff * tadpole_coeff)
    gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
    gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
    gauge_param.t_boundary = QudaTboundary(lattice.t_boundary)

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


def newQudaMultigridParam(dslash_type: QudaDslashType, mass: float, kappa: float, geo_block_size: List[List[int]]):
    is_staggered = dslash_type in (QudaDslashType.QUDA_STAGGERED_DSLASH, QudaDslashType.QUDA_ASQTAD_DSLASH)
    mg_param = QudaMultigridParam()
    mg_inv_param = QudaInvertParam()

    mg_inv_param.dslash_type = dslash_type
    mg_inv_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
    mg_inv_param.mass = mass
    mg_inv_param.kappa = kappa
    mg_inv_param.m5 = 0.0
    mg_inv_param.Ls = 1

    mg_inv_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    mg_inv_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
    mg_inv_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
    mg_inv_param.dagger = QudaDagType.QUDA_DAG_NO
    mg_inv_param.mass_normalization = QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION
    mg_inv_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
    mg_inv_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO

    mg_inv_param.tol = 0
    mg_inv_param.maxiter = 0
    mg_inv_param.reliable_delta = 1e-5
    mg_inv_param.gcrNkrylov = 8
    mg_inv_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO

    location = fieldLocation()
    mg_inv_param.input_location = location
    mg_inv_param.output_location = location
    mg_inv_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    mg_inv_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    mg_inv_param.clover_location = location
    mg_inv_param.clover_order = QudaCloverFieldOrder.QUDA_PACKED_CLOVER_ORDER
    mg_inv_param.clover_coeff = 1.0

    mg_inv_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE
    mg_inv_param.verbosity_precondition = QudaVerbosity.QUDA_SILENT

    mg_param.invert_param = mg_inv_param

    geo_block_size = geo_block_size + [[4, 4, 4, 4]]
    if is_staggered:
        geo_block_size = [[1, 1, 1, 1]] + geo_block_size
    n_level = len(geo_block_size)
    for i in range(n_level):
        geo_block_size[i] = geo_block_size[i] + [1] * (QUDA_MAX_DIM - len(geo_block_size[i]))
    mg_param.n_level = n_level
    mg_param.geo_block_size = geo_block_size
    if is_staggered:
        mg_param.spin_block_size = [0] + [0] + [1] * (QUDA_MAX_MG_LEVEL - 2)
        mg_param.n_vec = [3] + [64] * (QUDA_MAX_MG_LEVEL - 1)
        mg_param.n_block_ortho = [1] + [2] * (QUDA_MAX_MG_LEVEL - 1)
    else:
        mg_param.spin_block_size = [2] + [1] * (QUDA_MAX_MG_LEVEL - 1)
        mg_param.n_vec = [24] * QUDA_MAX_MG_LEVEL
        mg_param.n_block_ortho = [1] * QUDA_MAX_MG_LEVEL

    mg_param.verbosity = [QudaVerbosity.QUDA_SILENT] * QUDA_MAX_MG_LEVEL
    use_mma = QudaBoolean(mmaAvailable())
    mg_param.setup_use_mma = [use_mma] * QUDA_MAX_MG_LEVEL
    mg_param.dslash_use_mma = [use_mma] * QUDA_MAX_MG_LEVEL

    mg_param.setup_inv_type = [QudaInverterType.QUDA_CGNR_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.n_vec_batch = [1] * QUDA_MAX_MG_LEVEL
    mg_param.num_setup_iter = [1] * QUDA_MAX_MG_LEVEL
    mg_param.setup_tol = [1e-6] * QUDA_MAX_MG_LEVEL
    mg_param.setup_maxiter = [1000] * QUDA_MAX_MG_LEVEL
    mg_param.setup_maxiter_refresh = [1000 // 10] * QUDA_MAX_MG_LEVEL
    mg_param.setup_type = QudaSetupType.QUDA_NULL_VECTOR_SETUP
    mg_param.pre_orthonormalize = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.post_orthonormalize = QudaBoolean.QUDA_BOOLEAN_TRUE

    mg_param.coarse_solver = [QudaInverterType.QUDA_GCR_INVERTER] * (n_level - 1) + [
        QudaInverterType.QUDA_CA_GCR_INVERTER
    ] * (QUDA_MAX_MG_LEVEL - (n_level - 1))
    mg_param.coarse_solver_tol = [0.25] * QUDA_MAX_MG_LEVEL
    mg_param.coarse_solver_maxiter = [16] * QUDA_MAX_MG_LEVEL
    # mg_param.coarse_solver_ca_basis_size = [16] * QUDA_MAX_MG_LEVEL

    mg_param.smoother = [QudaInverterType.QUDA_CA_GCR_INVERTER] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_tol = [0.25] * QUDA_MAX_MG_LEVEL
    # mg_param.smoother_tol = [1e-10] * QUDA_MAX_MG_LEVEL
    mg_param.nu_pre = [0] * QUDA_MAX_MG_LEVEL
    mg_param.nu_post = [8] * QUDA_MAX_MG_LEVEL
    mg_param.omega = [1.0] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_schwarz_type = [QudaSchwarzType.QUDA_INVALID_SCHWARZ] * QUDA_MAX_MG_LEVEL
    mg_param.smoother_schwarz_cycle = [1] * QUDA_MAX_MG_LEVEL

    coarse_grid_solution_type = [QudaSolutionType.QUDA_MATPC_SOLUTION] * QUDA_MAX_MG_LEVEL
    smoother_solve_type = [QudaSolveType.QUDA_DIRECT_PC_SOLVE] * QUDA_MAX_MG_LEVEL
    if is_staggered:
        coarse_grid_solution_type[1] = QudaSolutionType.QUDA_MAT_SOLUTION
        smoother_solve_type[1] = QudaSolveType.QUDA_DIRECT_SOLVE
    mg_param.coarse_grid_solution_type = coarse_grid_solution_type
    mg_param.smoother_solve_type = smoother_solve_type
    mg_param.cycle_type = [QudaMultigridCycleType.QUDA_MG_CYCLE_RECURSIVE] * QUDA_MAX_MG_LEVEL
    mg_param.global_reduction = [QudaBoolean.QUDA_BOOLEAN_TRUE] * QUDA_MAX_MG_LEVEL

    mg_param.location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * QUDA_MAX_MG_LEVEL
    mg_param.setup_location = [QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION] * QUDA_MAX_MG_LEVEL
    # mg_param.setup_minimize_memory = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.compute_null_vector = QudaComputeNullVector.QUDA_COMPUTE_NULL_VECTOR_YES
    mg_param.generate_all_levels = QudaBoolean.QUDA_BOOLEAN_TRUE
    mg_param.run_verify = QudaBoolean.QUDA_BOOLEAN_TRUE
    mg_param.run_low_mode_check = QudaBoolean.QUDA_BOOLEAN_FALSE
    mg_param.run_oblique_proj_check = QudaBoolean.QUDA_BOOLEAN_FALSE

    mg_param.mu_factor = [1.0] * QUDA_MAX_MG_LEVEL
    transfer_type = [QudaTransferType.QUDA_TRANSFER_AGGREGATE] * QUDA_MAX_MG_LEVEL
    if is_staggered:
        transfer_type[0] = QudaTransferType.QUDA_TRANSFER_OPTIMIZED_KD
    mg_param.transfer_type = transfer_type

    return mg_param, mg_inv_param


def newQudaInvertParam(
    dslash_type: QudaDslashType,
    mass: float,
    kappa: float,
    tol: float,
    maxiter: int,
    clover_coeff: float,
    clover_anisotropy: float,
    mg_param: Optional[QudaMultigridParam],
):
    is_staggered = dslash_type in (QudaDslashType.QUDA_STAGGERED_DSLASH, QudaDslashType.QUDA_ASQTAD_DSLASH)
    invert_param = QudaInvertParam()

    invert_param.dslash_type = dslash_type
    if is_staggered:
        invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
    else:
        invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
    invert_param.mass = mass
    invert_param.kappa = kappa
    invert_param.m5 = 0.0
    invert_param.Ls = 1
    invert_param.laplace3D = 3

    invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
    invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
    invert_param.dagger = QudaDagType.QUDA_DAG_NO
    invert_param.mass_normalization = QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION
    invert_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
    invert_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO

    invert_param.tol = tol
    invert_param.tol_restart = 5e3 * tol
    invert_param.tol_hq = 0.0
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

    location = fieldLocation()
    invert_param.input_location = location
    invert_param.output_location = location
    invert_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    invert_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    if clover_coeff != 0.0:
        invert_param.clover_location = location
        invert_param.clover_order = QudaCloverFieldOrder.QUDA_PACKED_CLOVER_ORDER
        invert_param.clover_csw = clover_anisotropy  # to save clover_anisotropy, not real csw
        invert_param.clover_coeff = clover_coeff
        invert_param.compute_clover = 1
        invert_param.compute_clover_inverse = 1
        invert_param.return_clover = 0
        invert_param.return_clover_inverse = 0

    if mg_param is not None:
        invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
        invert_param.inv_type_precondition = QudaInverterType.QUDA_MG_INVERTER
        invert_param.tol_precondition = mg_param.coarse_solver_tol[0]
        invert_param.maxiter_precondition = mg_param.coarse_solver_maxiter[0]
        invert_param.verbosity_precondition = mg_param.verbosity[0]
        invert_param.schwarz_type = QudaSchwarzType.QUDA_ADDITIVE_SCHWARZ
        invert_param.precondition_cycle = 1
        invert_param.omega = 1.0

    invert_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE

    return invert_param


class Multigrid:
    param: Optional[QudaMultigridParam]
    inv_param: Optional[QudaInvertParam]
    instance: Optional[Pointer]

    def __init__(self, param: Optional[QudaMultigridParam], inv_param: Optional[QudaInvertParam]) -> None:
        self.param = param
        self.inv_param = inv_param
        self.instance = None

    def setParam(
        self,
        *,
        coarse_tol: float = 0.25,
        coarse_maxiter: int = 16,
        setup_tol: float = 1e-6,
        setup_maxiter: int = 1000,
        smoother_tol: float = 0.25,
        smoother_nu_pre: int = 0,
        smoother_nu_post: int = 8,
        smoother_omega: float = 1.0,
    ):
        assert self.param is not None
        self.param.coarse_solver_tol = [coarse_tol] * QUDA_MAX_MG_LEVEL
        self.param.coarse_solver_maxiter = [coarse_maxiter] * QUDA_MAX_MG_LEVEL
        self.param.setup_tol = [setup_tol] * QUDA_MAX_MG_LEVEL
        self.param.setup_maxiter = [setup_maxiter] * QUDA_MAX_MG_LEVEL
        self.param.setup_maxiter_refresh = [setup_maxiter // 10] * QUDA_MAX_MG_LEVEL
        self.param.smoother_tol = [smoother_tol] * QUDA_MAX_MG_LEVEL
        self.param.nu_pre = [smoother_nu_pre] * QUDA_MAX_MG_LEVEL
        self.param.nu_post = [smoother_nu_post] * QUDA_MAX_MG_LEVEL
        self.param.omega = [smoother_omega] * QUDA_MAX_MG_LEVEL

    def new(self):
        assert self.param is not None
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = newMultigridQuda(self.param)

    def update(self, thin_update_only: bool):
        assert self.param is not None
        if self.instance is not None:
            self.param.thin_update_only = QudaBoolean(thin_update_only)
            updateMultigridQuda(self.instance, self.param)
            self.param.thin_update_only = QudaBoolean.QUDA_BOOLEAN_FALSE

    def destroy(self):
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = None


def loadMultigrid(multigrid: Multigrid, invert_param: QudaInvertParam, thin_update_only: bool):
    if multigrid.param is not None:
        if multigrid.instance is None:
            multigrid.new()
            assert multigrid.instance is not None
            invert_param.preconditioner = multigrid.instance
        else:
            multigrid.update(thin_update_only)


def freeMultigrid(multigrid: Multigrid, invert_param: QudaInvertParam):
    if multigrid.param is not None:
        multigrid.destroy()
        invert_param.preconditioner = Pointer("void")


class Deflation:
    param: Optional[QudaEigParam]
    inv_param: Optional[QudaInvertParam]
    instance: Optional[Pointer]

    def __init__(self, param: Optional[QudaEigParam], inv_param: Optional[QudaInvertParam]) -> None:
        self.param = param
        self.inv_param = inv_param
        self.instance = None

    def new(self):
        assert self.param is not None
        if self.instance is not None:
            destroyDeflationQuda(self.instance)
        self.instance = newDeflationQuda(self.param)

    def destroy(self):
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = None


def loadGauge(gauge: LatticeGauge, gauge_param: QudaGaugeParam):
    gauge_in = gauge.copy()
    if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
        gauge_in.setAntiPeriodicT()
    if gauge_param.anisotropy != 1.0:
        gauge_in.setAnisotropy(gauge_param.anisotropy)
    gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge_in.data_ptrs, gauge_param)
    gauge_param.use_resident_gauge = 1


def freeGauge():
    freeUniqueGaugeQuda(QudaLinkType.QUDA_WILSON_LINKS)


def loadClover(
    clover: Optional[LatticeClover],
    clover_inv: Optional[LatticeClover],
    gauge: Optional[LatticeGauge],
    gauge_param: QudaGaugeParam,
    invert_param: QudaInvertParam,
):
    if clover is None or clover_inv is None:
        assert gauge is not None
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


def freeClover():
    freeCloverQuda()


def newAsqtadPathCoeff(tadpole_coeff: float):
    u1 = 1.0 / tadpole_coeff
    u2 = u1 * u1
    u4 = u2 * u2
    u6 = u4 * u2

    path_coeff_1 = [
        ((1.0 / 8.0) + (6.0 / 16.0) + (1.0 / 8.0)),  # one link
        # One link is 1/8 as in fat7 +3/8 for Lepage + 1/8 for Naik
        u2 * (-1.0 / 24.0),  # Naik
        u2 * (-1.0 / 8.0) * 0.5,  # simple staple
        u4 * (1.0 / 8.0) * 0.25 * 0.5,  # displace link in two directions
        u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),  # displace link in three directions
        u4 * (-1.0 / 16.0),  # Correct O(a^2) errors
    ]

    return numpy.array(path_coeff_1, "<f8")


def newHISQPathCoeff(tadpole_coeff: float):
    u1 = 1.0 / tadpole_coeff
    u2 = u1 * u1
    u4 = u2 * u2
    u6 = u4 * u2

    # First path: create V, W links
    path_coeff_1 = [
        (1.0 / 8.0),  # one link
        u2 * (0.0),  # Naik
        u2 * (-1.0 / 8.0) * 0.5,  # simple staple
        u4 * (1.0 / 8.0) * 0.25 * 0.5,  # displace link in two directions
        u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),  # displace link in three directions
        u4 * (0.0),  # Lepage term
    ]

    # Second path: create X, long links
    path_coeff_2 = [
        ((1.0 / 8.0) + (2.0 * 6.0 / 16.0) + (1.0 / 8.0)),  # one link
        # One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik
        (-1.0 / 24.0),  # Naik
        (-1.0 / 8.0) * 0.5,  # simple staple
        (1.0 / 8.0) * 0.25 * 0.5,  # displace link in two directions
        (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),  # displace link in three directions
        (-2.0 / 16.0),  # Lepage term, correct O(a^2) 2x ASQTAD
    ]

    # Paths for epsilon corrections. Not used if n_naiks = 1.
    path_coeff_3 = [
        (1.0 / 8.0),  # one link b/c of Naik
        (-1.0 / 24.0),  # Naik
        0.0,  # simple staple
        0.0,  # displace link in two directions
        0.0,  # displace link in three directions
        0.0,  # Lepage term
    ]

    return numpy.array(path_coeff_1, "<f8"), numpy.array(path_coeff_2, "<f8"), numpy.array(path_coeff_3, "<f8")


def computeULink(gauge: LatticeGauge, gauge_param: QudaGaugeParam):
    u_link = gauge.copy()

    gauge_param.use_resident_gauge = 0
    gauge_param.make_resident_gauge = 0
    gauge_param.return_result_gauge = 1
    gauge_param.staggered_phase_applied = 0
    staggeredPhaseQuda(u_link.data_ptrs, gauge_param)
    gauge_param.use_resident_gauge = 1
    gauge_param.make_resident_gauge = 0
    gauge_param.return_result_gauge = 0
    gauge_param.staggered_phase_applied = 1

    return u_link


def computeWLink(u_link: LatticeGauge, path_coeff: NDArray[numpy.float64], gauge_param: QudaGaugeParam):
    w_link = LatticeGauge(u_link.latt_info)
    computeKSLinkQuda(
        nullptrs,
        nullptrs,
        w_link.data_ptrs,
        u_link.data_ptrs,
        path_coeff,
        gauge_param,
    )

    return w_link


def computeVWLink(u_link: LatticeGauge, path_coeff: NDArray[numpy.float64], gauge_param: QudaGaugeParam):
    v_link = LatticeGauge(u_link.latt_info)
    w_link = LatticeGauge(u_link.latt_info)
    computeKSLinkQuda(
        v_link.data_ptrs,
        nullptrs,
        w_link.data_ptrs,
        u_link.data_ptrs,
        path_coeff,
        gauge_param,
    )

    return v_link, w_link


def computeXLink(
    w_link: LatticeGauge,
    path_coeff: NDArray[numpy.float64],
    gauge_param: QudaGaugeParam,
):
    fatlink = LatticeGauge(w_link.latt_info)
    longlink = LatticeGauge(w_link.latt_info)
    computeKSLinkQuda(
        fatlink.data_ptrs,
        longlink.data_ptrs,
        nullptrs,
        w_link.data_ptrs,
        path_coeff,
        gauge_param,
    )

    return fatlink, longlink


def computeXLinkEpsilon(
    fatlink: LatticeGauge,
    longlink: LatticeGauge,
    w_link: LatticeGauge,
    path_coeff: NDArray[numpy.float64],
    naik_epsilon: float,
    gauge_param: QudaGaugeParam,
):
    if naik_epsilon != 0:
        fatlink_epsilon = LatticeGauge(w_link.latt_info)
        longlink_epsilon = LatticeGauge(w_link.latt_info)
        computeKSLinkQuda(
            fatlink_epsilon.data_ptrs,
            longlink_epsilon.data_ptrs,
            nullptrs,
            w_link.data_ptrs,
            path_coeff,
            gauge_param,
        )
        fatlink_epsilon *= naik_epsilon
        longlink_epsilon *= naik_epsilon
        fatlink_epsilon += fatlink
        longlink_epsilon += longlink

        return fatlink_epsilon, longlink_epsilon
    else:
        return fatlink, longlink


def loadStaggeredGauge(gauge: LatticeGauge, gauge_param: QudaGaugeParam):
    u_link = computeULink(gauge, gauge_param)
    gauge_param.use_resident_gauge = 0
    loadGaugeQuda(u_link.data_ptrs, gauge_param)
    gauge_param.use_resident_gauge = 1


def freeStaggeredGauge():
    freeUniqueGaugeQuda(QudaLinkType.QUDA_WILSON_LINKS)


def loadFatLongGauge(
    fatlink: LatticeGauge,
    longlink: LatticeGauge,
    tadpole_coeff: float,
    naik_epsilon: float,
    gauge_param: QudaGaugeParam,
):
    ga_pad = gauge_param.ga_pad
    staggered_phase_type = gauge_param.staggered_phase_type

    gauge_param.use_resident_gauge = 0
    gauge_param.type = QudaLinkType.QUDA_ASQTAD_FAT_LINKS
    loadGaugeQuda(fatlink.data_ptrs, gauge_param)
    gauge_param.type = QudaLinkType.QUDA_ASQTAD_LONG_LINKS
    setReconstructParam(_reconstruct["staggered"], gauge_param)
    gauge_param.ga_pad = ga_pad * 3
    gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_NO
    gauge_param.tadpole_coeff = tadpole_coeff
    gauge_param.scale = -(1 + naik_epsilon) / (24 * tadpole_coeff * tadpole_coeff)
    loadGaugeQuda(longlink.data_ptrs, gauge_param)
    gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
    setReconstructParam(_reconstruct["none"], gauge_param)
    gauge_param.ga_pad = ga_pad
    gauge_param.staggered_phase_type = staggered_phase_type
    gauge_param.tadpole_coeff = 1.0
    gauge_param.scale = 1.0
    gauge_param.use_resident_gauge = 1


def freeFatLongGauge():
    freeUniqueGaugeQuda(QudaLinkType.QUDA_ASQTAD_FAT_LINKS)
    freeUniqueGaugeQuda(QudaLinkType.QUDA_ASQTAD_LONG_LINKS)
