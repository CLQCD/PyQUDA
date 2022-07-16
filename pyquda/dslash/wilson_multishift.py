from typing import List

from ..pyquda import (  # noqa: F401
    Pointer, QudaGaugeParam, QudaInvertParam, loadGaugeQuda, invertQuda, invertMultiShiftQuda
)
from ..enum_quda import (  # noqa: F401
    QudaConstant, qudaError_t, QudaMemoryType, QudaLinkType, QudaGaugeFieldOrder, QudaTboundary, QudaPrecision,
    QudaReconstructType, QudaGaugeFixed, QudaDslashType, QudaInverterType, QudaEigType, QudaEigSpectrumType,
    QudaSolutionType, QudaSolveType, QudaMultigridCycleType, QudaSchwarzType, QudaResidualType, QudaCABasis,
    QudaMatPCType, QudaDagType, QudaMassNormalization, QudaSolverNormalization, QudaPreserveSource,
    QudaDiracFieldOrder, QudaCloverFieldOrder, QudaVerbosity, QudaTune, QudaPreserveDirac, QudaParity, QudaDiracType,
    QudaFieldLocation, QudaSiteSubset, QudaSiteOrder, QudaFieldOrder, QudaFieldCreate, QudaGammaBasis, QudaSourceType,
    QudaNoiseType, QudaProjectionType, QudaPCType, QudaTwistFlavorType, QudaTwistDslashType, QudaTwistCloverDslashType,
    QudaTwistGamma5Type, QudaUseInitGuess, QudaDeflatedGuess, QudaComputeNullVector, QudaSetupType, QudaTransferType,
    QudaBoolean, QUDA_BOOLEAN_NO, QUDA_BOOLEAN_YES, QudaBLASOperation, QudaBLASDataType, QudaBLASDataOrder,
    QudaDirection, QudaLinkDirection, QudaFieldGeometry, QudaGhostExchange, QudaStaggeredPhase, QudaContractType,
    QudaContractGamma, QudaWFlowType, QudaExtLibType
)

from ..core import LatticeGauge, LatticeFermion


def newQudaGaugeParam(X: List[int], anisotropy: float):
    Lx, Ly, Lz, Lt = X
    Lmin = min(Lx, Ly, Lz, Lt)
    ga_pad = Lx * Ly * Lz * Lt // Lmin

    gauge_param = QudaGaugeParam()

    gauge_param.X = X
    gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
    gauge_param.gauge_order = QudaGaugeFieldOrder.QUDA_QDP_GAUGE_ORDER
    gauge_param.t_boundary = QudaTboundary.QUDA_ANTI_PERIODIC_T
    gauge_param.cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
    gauge_param.cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
    gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
    gauge_param.cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
    gauge_param.reconstruct_sloppy = QudaReconstructType.QUDA_RECONSTRUCT_12
    gauge_param.gauge_fix = QudaGaugeFixed.QUDA_GAUGE_FIXED_NO
    gauge_param.anisotropy = anisotropy
    gauge_param.ga_pad = ga_pad

    return gauge_param


def newQudaInvertParam(kappa: float, tol: float, maxiter: float):
    invert_param = QudaInvertParam()

    invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
    invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
    invert_param.kappa = kappa
    invert_param.tol = tol
    invert_param.maxiter = maxiter
    invert_param.reliable_delta = 0.001
    invert_param.pipeline = 0

    invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE

    invert_param.dagger = QudaDagType.QUDA_DAG_NO
    invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION

    invert_param.cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
    invert_param.cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
    invert_param.cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
    invert_param.cuda_prec_precondition = QudaPrecision.QUDA_HALF_PRECISION
    invert_param.preserve_source = QudaPreserveSource.QUDA_PRESERVE_SOURCE_NO
    invert_param.use_init_guess = QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO
    invert_param.dirac_order = QudaDiracFieldOrder.QUDA_DIRAC_ORDER
    invert_param.gamma_basis = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

    invert_param.tune = QudaTune.QUDA_TUNE_YES

    invert_param.inv_type_precondition = QudaInverterType.QUDA_INVALID_INVERTER
    invert_param.tol_precondition = 1.0e-1
    invert_param.maxiter_precondition = 1000
    invert_param.verbosity_precondition = QudaVerbosity.QUDA_SILENT
    invert_param.gcrNkrylov = 1

    invert_param.verbosity = QudaVerbosity.QUDA_SUMMARIZE

    invert_param.sp_pad = 0
    invert_param.cl_pad = 0

    return invert_param


def loadGauge(gauge: LatticeGauge, gauge_param: QudaGaugeParam, invert_param: QudaInvertParam):
    anisotropy = gauge_param.anisotropy

    gauge_data_bak = gauge.data.copy()
    if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
        gauge.setAntiPeroidicT()
    if anisotropy != 1.0:
        gauge.setAnisotropy(anisotropy)
    loadGaugeQuda(gauge.data_ptrs, gauge_param)
    gauge.data = gauge_data_bak


def invert(b: LatticeFermion, invert_param: QudaInvertParam):
    kappa = invert_param.kappa

    x = LatticeFermion(b.latt_size)

    # invertMultiShiftQuda(x.data_ptr, b.data_ptr, invert_param)
    invertQuda(x.data_ptr, b.data_ptr, invert_param)
    x.data *= 2 * kappa

    return x
