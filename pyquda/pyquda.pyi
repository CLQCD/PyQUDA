from typing import List

import numpy

size_t = int
double = float
double_complex = complex

from enum_quda import (  # noqa: F401
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
    QudaContractGamma, QudaWFlowType, QudaExtLibType,
)


class Pointer:
    def __init__(self, dtype: str):
        ...


class Pointers(Pointer):
    def __init__(self, dtype: str):
        ...


def getDataPointers(ndarray: numpy.ndarray, n: int) -> Pointers:
    ...


def getDataPointer(ndarray: numpy.ndarray) -> Pointer:
    ...


def getEvenPointer(ndarray: numpy.ndarray) -> Pointer:
    ...


def getOddPointer(ndarray: numpy.ndarray) -> Pointer:
    ...


class QudaGaugeParam:
    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    struct_size: size_t
    location: QudaFieldLocation
    X: List[int, 4]
    anisotropy: double
    tadpole_coeff: double
    scale: double
    type: QudaLinkType
    gauge_order: QudaGaugeFieldOrder
    t_boundary: QudaTboundary
    cpu_prec: QudaPrecision
    cuda_prec: QudaPrecision
    reconstruct: QudaReconstructType
    cuda_prec_sloppy: QudaPrecision
    reconstruct_sloppy: QudaReconstructType
    cuda_prec_refinement_sloppy: QudaPrecision
    reconstruct_refinement_sloppy: QudaReconstructType
    cuda_prec_precondition: QudaPrecision
    reconstruct_precondition: QudaReconstructType
    cuda_prec_eigensolver: QudaPrecision
    reconstruct_eigensolver: QudaReconstructType
    gauge_fix: QudaGaugeFixed
    ga_pad: int
    site_ga_pad: int
    staple_pad: int
    llfat_ga_pad: int
    mom_ga_pad: int
    staggered_phase_type: QudaStaggeredPhase
    staggered_phase_applied: int
    i_mu: double
    overlap: int
    overwrite_mom: int
    use_resident_gauge: int
    use_resident_mom: int
    make_resident_gauge: int
    make_resident_mom: int
    return_result_gauge: int
    return_result_mom: int
    gauge_offset: size_t
    mom_offset: size_t
    site_size: size_t


class QudaInvertParam:
    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    struct_size: size_t
    input_location: QudaFieldLocation
    output_location: QudaFieldLocation
    dslash_type: QudaDslashType
    inv_type: QudaInverterType
    mass: double
    kappa: double
    m5: double
    Ls: int
    b_5: List[double_complex, QudaConstant.QUDA_MAX_DWF_LS]
    c_5: List[double_complex, QudaConstant.QUDA_MAX_DWF_LS]
    eofa_shift: double
    eofa_pm: int
    mq1: double
    mq2: double
    mq3: double
    mu: double
    epsilon: double
    twist_flavor: QudaTwistFlavorType
    laplace3D: int
    tol: double
    tol_restart: double
    tol_hq: double
    compute_true_res: int
    true_res: double
    true_res_hq: double
    maxiter: int
    reliable_delta: double
    reliable_delta_refinement: double
    use_alternative_reliable: int
    use_sloppy_partial_accumulator: int
    solution_accumulator_pipeline: int
    max_res_increase: int
    max_res_increase_total: int
    max_hq_res_increase: int
    max_hq_res_restart_total: int
    heavy_quark_check: int
    pipeline: int
    num_offset: int
    num_src: int
    num_src_per_sub_partition: int
    split_grid: List[int, QudaConstant.QUDA_MAX_DIM]
    overlap: int
    offset: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    tol_offset: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    tol_hq_offset: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    true_res_offset: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    iter_res_offset: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    true_res_hq_offset: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    residue: List[double, QudaConstant.QUDA_MAX_MULTI_SHIFT]
    compute_action: int
    action: List[double, 2]
    solution_type: QudaSolutionType
    solve_type: QudaSolveType
    matpc_type: QudaMatPCType
    dagger: QudaDagType
    mass_normalization: QudaMassNormalization
    solver_normalization: QudaSolverNormalization
    preserve_source: QudaPreserveSource
    cpu_prec: QudaPrecision
    cuda_prec: QudaPrecision
    cuda_prec_sloppy: QudaPrecision
    cuda_prec_refinement_sloppy: QudaPrecision
    cuda_prec_precondition: QudaPrecision
    cuda_prec_eigensolver: QudaPrecision
    dirac_order: QudaDiracFieldOrder
    gamma_basis: QudaGammaBasis
    clover_location: QudaFieldLocation
    clover_cpu_prec: QudaPrecision
    clover_cuda_prec: QudaPrecision
    clover_cuda_prec_sloppy: QudaPrecision
    clover_cuda_prec_refinement_sloppy: QudaPrecision
    clover_cuda_prec_precondition: QudaPrecision
    clover_cuda_prec_eigensolver: QudaPrecision
    clover_order: QudaCloverFieldOrder
    use_init_guess: QudaUseInitGuess
    clover_csw: double
    clover_coeff: double
    clover_rho: double
    compute_clover_trlog: int
    trlogA: List[double, 2]
    compute_clover: int
    compute_clover_inverse: int
    return_clover: int
    return_clover_inverse: int
    verbosity: QudaVerbosity
    sp_pad: int
    cl_pad: int
    iter: int
    gflops: double
    secs: double
    tune: QudaTune
    Nsteps: int
    gcrNkrylov: int
    inv_type_precondition: QudaInverterType
    preconditioner: Pointer
    deflation_op: Pointer
    eig_param: Pointer
    deflate: QudaBoolean
    dslash_type_precondition: QudaDslashType
    verbosity_precondition: QudaVerbosity
    tol_precondition: double
    maxiter_precondition: int
    omega: double
    ca_basis: QudaCABasis
    ca_lambda_min: double
    ca_lambda_max: double
    precondition_cycle: int
    schwarz_type: QudaSchwarzType
    residual_type: QudaResidualType
    cuda_prec_ritz: QudaPrecision
    n_ev: int
    max_search_dim: int
    rhs_idx: int
    deflation_grid: int
    eigenval_tol: double
    eigcg_max_restarts: int
    max_restart_num: int
    inc_tol: double
    make_resident_solution: int
    use_resident_solution: int
    chrono_make_resident: int
    chrono_replace_last: int
    chrono_use_resident: int
    chrono_max_dim: int
    chrono_index: int
    chrono_precision: QudaPrecision
    extlib_type: QudaExtLibType
    native_blas_lapack: QudaBoolean


class QudaEigParam:
    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    struct_size: size_t
    invert_param: QudaInvertParam
    eig_type: QudaEigType
    use_poly_acc: QudaBoolean
    poly_deg: int
    a_min: double
    a_max: double
    preserve_deflation: QudaBoolean
    preserve_deflation_space: Pointer
    preserve_evals: QudaBoolean
    use_dagger: QudaBoolean
    use_norm_op: QudaBoolean
    use_eigen_qr: QudaBoolean
    compute_svd: QudaBoolean
    compute_gamma5: QudaBoolean
    require_convergence: QudaBoolean
    spectrum: QudaEigSpectrumType
    n_ev: int
    n_kr: int
    nLockedMax: int
    n_conv: int
    n_ev_deflate: int
    tol: double
    qr_tol: double
    check_interval: int
    max_restarts: int
    batched_rotate: int
    block_size: int
    arpack_check: QudaBoolean
    arpack_logfile: bytes[512]
    QUDA_logfile: bytes[512]
    nk: int
    np: int
    import_vectors: QudaBoolean
    cuda_prec_ritz: QudaPrecision
    mem_type_ritz: QudaMemoryType
    location: QudaFieldLocation
    run_verify: QudaBoolean
    vec_infile: bytes[256]
    vec_outfile: bytes[256]
    save_prec: QudaPrecision
    io_parity_inflate: QudaBoolean
    gflops: double
    secs: double
    extlib_type: QudaExtLibType


class QudaMultigridParam:
    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    struct_size: size_t
    invert_param: QudaInvertParam
    eig_param: List[QudaEigParam, QudaConstant.QUDA_MAX_MG_LEVEL]
    n_level: int
    geo_block_size: List[List[int, QudaConstant.QUDA_MAX_DIM], QudaConstant.QUDA_MAX_MG_LEVEL]
    spin_block_size: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    n_vec: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    precision_null: List[QudaPrecision, QudaConstant.QUDA_MAX_MG_LEVEL]
    n_block_ortho: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    verbosity: List[QudaVerbosity, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_inv_type: List[QudaInverterType, QudaConstant.QUDA_MAX_MG_LEVEL]
    num_setup_iter: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_tol: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_maxiter: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_maxiter_refresh: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_ca_basis: List[QudaCABasis, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_ca_basis_size: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_ca_lambda_min: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_ca_lambda_max: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_type: QudaSetupType
    pre_orthonormalize: QudaBoolean
    post_orthonormalize: QudaBoolean
    coarse_solver: List[QudaInverterType, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_solver_tol: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_solver_maxiter: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_basis: List[QudaCABasis, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_basis_size: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_lambda_min: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_lambda_max: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    smoother: List[QudaInverterType, QudaConstant.QUDA_MAX_MG_LEVEL]
    smoother_tol: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    nu_pre: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    nu_post: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    omega: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    smoother_halo_precision: List[QudaPrecision, QudaConstant.QUDA_MAX_MG_LEVEL]
    smoother_schwarz_type: List[QudaSchwarzType, QudaConstant.QUDA_MAX_MG_LEVEL]
    smoother_schwarz_cycle: List[int, QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_grid_solution_type: List[QudaSolutionType, QudaConstant.QUDA_MAX_MG_LEVEL]
    smoother_solve_type: List[QudaSolveType, QudaConstant.QUDA_MAX_MG_LEVEL]
    cycle_type: List[QudaMultigridCycleType, QudaConstant.QUDA_MAX_MG_LEVEL]
    global_reduction: List[QudaBoolean, QudaConstant.QUDA_MAX_MG_LEVEL]
    location: List[QudaFieldLocation, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_location: List[QudaFieldLocation, QudaConstant.QUDA_MAX_MG_LEVEL]
    use_eig_solver: List[QudaBoolean, QudaConstant.QUDA_MAX_MG_LEVEL]
    setup_minimize_memory: QudaBoolean
    compute_null_vector: QudaComputeNullVector
    generate_all_levels: QudaBoolean
    run_verify: QudaBoolean
    run_low_mode_check: QudaBoolean
    run_oblique_proj_check: QudaBoolean
    vec_load: List[QudaBoolean, QudaConstant.QUDA_MAX_MG_LEVEL]
    vec_infile: List[bytes[256], QudaConstant.QUDA_MAX_MG_LEVEL]
    vec_store: List[QudaBoolean, QudaConstant.QUDA_MAX_MG_LEVEL]
    vec_outfile: List[bytes[256], QudaConstant.QUDA_MAX_MG_LEVEL]
    coarse_guess: QudaBoolean
    preserve_deflation: QudaBoolean
    gflops: double
    secs: double
    mu_factor: List[double, QudaConstant.QUDA_MAX_MG_LEVEL]
    transfer_type: List[QudaTransferType, QudaConstant.QUDA_MAX_MG_LEVEL]
    use_mma: QudaBoolean
    thin_update_only: QudaBoolean


class QudaGaugeObservableParam:
    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    struct_size: size_t
    su_project: QudaBoolean
    compute_plaquette: QudaBoolean
    plaquette: List[double, 3]
    compute_qcharge: QudaBoolean
    qcharge: double
    energy: List[double, 3]
    compute_qcharge_density: QudaBoolean
    qcharge_density: Pointer


class QudaBLASParam:
    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    struct_size: size_t
    trans_a: QudaBLASOperation
    trans_b: QudaBLASOperation
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    a_offset: int
    b_offset: int
    c_offset: int
    a_stride: int
    b_stride: int
    c_stride: int
    alpha: double_complex
    beta: double_complex
    batch_count: int
    data_type: QudaBLASDataType
    data_order: QudaBLASDataOrder


def setVerbosityQuda(verbosity: QudaVerbosity, prefix: str, outfile: Pointer) -> None:
    ...


def initQudaDevice(device: int) -> None:
    ...


def initQudaMemory() -> None:
    ...


def initQuda(device: int) -> None:
    ...


def endQuda() -> None:
    ...


def loadGaugeQuda(h_gauge: Pointer, param: QudaGaugeParam) -> None:
    ...


def freeGaugeQuda() -> None:
    ...


def saveGaugeQuda(h_gauge: Pointer, param: QudaGaugeParam) -> None:
    ...


def loadCloverQuda(h_clover: Pointer, h_clovinv: Pointer, inv_param: QudaInvertParam) -> None:
    ...


def freeCloverQuda() -> None:
    ...


def invertQuda(h_x: Pointer, h_b: Pointer, param: QudaInvertParam) -> None:
    ...


def dslashQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam, parity: QudaParity) -> None:
    ...


def cloverQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam, parity: QudaParity, inverse: int) -> None:
    ...


def MatQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam) -> None:
    ...


def MatDagMatQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam) -> None:
    ...


def createCloverQuda(param: QudaInvertParam) -> None:
    ...


def plaqQuda(plaq: List[double, 3]) -> None:
    ...


def performAPEnStep(n_steps: int, alpha: double, meas_interval: int):
    ...


def performSTOUTnStep(n_steps: int, rho: double, meas_interval: int):
    ...


def performOvrImpSTOUTnStep(n_steps: int, rho: double, epsilon: double, meas_interval: int):
    ...


def performWFlownStep(n_steps: int, step_size: double, meas_interval: int, wflow_type: QudaWFlowType):
    ...
