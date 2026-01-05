from typing import List

size_t = int
double = float
double_complex = complex

from numpy import int32, float64, complex128
from numpy.typing import NDArray

_field = complex128
_fields = complex128

from pyquda_comm.pointer import Pointer, Pointers
from .enum_quda import (  # noqa: F401
    QUDA_INVALID_ENUM,
    QUDA_VERSION_MAJOR,
    QUDA_VERSION_MINOR,
    QUDA_VERSION_SUBMINOR,
    QUDA_VERSION,
    QUDA_MAX_DIM,
    QUDA_MAX_GEOMETRY,
    QUDA_MAX_MULTI_SHIFT,
    QUDA_MAX_MULTI_SRC,
    QUDA_MAX_DWF_LS,
    QUDA_MAX_MG_LEVEL,
    qudaError_t,
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
    QudaAcceleratorType,
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
    QudaDilutionType,
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
    QudaBLASType,
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
    QudaGaugeSmearType,
    QudaExtLibType,
)

class QudaGaugeParam:
    """
    Parameters having to do with the gauge field or the
    interpretation of the gauge field by various Dirac operators
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size"""
    location: QudaFieldLocation
    """The location of the gauge field"""

    X: List[int, 4]
    """The local space-time dimensions (without checkboarding)"""

    anisotropy: double
    """Used for Wilson and Wilson-clover"""
    tadpole_coeff: double
    """Used for staggered only"""
    scale: double
    """Used by staggered long links"""

    type: QudaLinkType
    """The link type of the gauge field (e.g., Wilson, fat, long, etc.)"""
    gauge_order: QudaGaugeFieldOrder
    """The ordering on the input gauge field"""

    t_boundary: QudaTboundary
    """The temporal boundary condition that will be used for fermion fields"""

    cpu_prec: QudaPrecision
    """The precision used by the caller"""

    cuda_prec: QudaPrecision
    """The precision of the cuda gauge field"""
    reconstruct: QudaReconstructType
    """The reconstruction type of the cuda gauge field"""

    cuda_prec_sloppy: QudaPrecision
    """The precision of the sloppy gauge field"""
    reconstruct_sloppy: QudaReconstructType
    """The recontruction type of the sloppy gauge field"""

    cuda_prec_refinement_sloppy: QudaPrecision
    """The precision of the sloppy gauge field for the refinement step in multishift"""
    reconstruct_refinement_sloppy: QudaReconstructType
    """The recontruction type of the sloppy gauge field for the refinement step in multishift"""

    cuda_prec_precondition: QudaPrecision
    """The precision of the preconditioner gauge field"""
    reconstruct_precondition: QudaReconstructType
    """The recontruction type of the preconditioner gauge field"""

    cuda_prec_eigensolver: QudaPrecision
    """The precision of the eigensolver gauge field"""
    reconstruct_eigensolver: QudaReconstructType
    """The recontruction type of the eigensolver gauge field"""

    gauge_fix: QudaGaugeFixed
    """Whether the input gauge field is in the axial gauge or not"""

    ga_pad: int
    """The pad size that native GaugeFields will use (default=0)"""

    site_ga_pad: int
    """Used by link fattening and the gauge and fermion forces"""

    staple_pad: int
    """Used by link fattening"""
    llfat_ga_pad: int
    """Used by link fattening"""
    mom_ga_pad: int
    """Used by the gauge and fermion forces"""

    staggered_phase_type: QudaStaggeredPhase
    """Set the staggered phase type of the links"""
    staggered_phase_applied: int
    """Whether the staggered phase has already been applied to the links"""

    i_mu: double
    """Imaginary chemical potential"""

    overlap: int
    """Width of overlapping domains"""

    overwrite_gauge: int
    """When computing gauge, should we overwrite it or accumulate to it"""
    overwrite_mom: int
    """When computing momentum, should we overwrite it or accumulate to it"""

    use_resident_gauge: int
    """Use the resident gauge field as input"""
    use_resident_mom: int
    """Use the resident momentum field as input"""
    make_resident_gauge: int
    """Make the result gauge field resident"""
    make_resident_mom: int
    """Make the result momentum field resident"""
    return_result_gauge: int
    """Return the result gauge field"""
    return_result_mom: int
    """Return the result momentum field"""

    gauge_offset: size_t
    """Offset into MILC site struct to the gauge field (only if gauge_order=MILC_SITE_GAUGE_ORDER)"""
    mom_offset: size_t
    """Offset into MILC site struct to the momentum field (only if gauge_order=MILC_SITE_GAUGE_ORDER)"""
    site_size: size_t
    """Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER)"""

class QudaInvertParam:
    """
    Parameters relating to the solver and the choice of Dirac operator.
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size"""

    input_location: QudaFieldLocation
    """The location of the input field"""
    output_location: QudaFieldLocation
    """The location of the output field"""

    dslash_type: QudaDslashType
    """The Dirac Dslash type that is being used"""
    inv_type: QudaInverterType
    """Which linear solver to use"""

    mass: double
    """Used for staggered only"""
    kappa: double
    """Used for Wilson and Wilson-clover"""

    m5: double
    """Domain wall height"""
    Ls: int
    """Extent of the 5th dimension (for domain wall)"""

    b_5: List[double_complex, QUDA_MAX_DWF_LS]
    """Mobius coefficients - only real part used if regular Mobius"""
    c_5: List[double_complex, QUDA_MAX_DWF_LS]
    """Mobius coefficients - only real part used if regular Mobius"""

    # The following specifies the EOFA parameters. Notation follows arXiv:1706.05843
    eofa_shift: double
    R"""the "\beta" in the paper"""
    eofa_pm: int
    """plus or minus for the EOFA operator"""
    mq1: double
    """
    three masses corresponds to Hasenbusch mass spliting.
    As far as I know mq1 is always the same as "mass" but it's here just for consistence.
    """
    mq2: double
    """three masses corresponds to Hasenbusch mass spliting."""
    mq3: double
    """three masses corresponds to Hasenbusch mass spliting."""

    mu: double
    """Twisted mass parameter"""
    tm_rho: double
    """Hasenbusch mass shift applied like twisted mass to diagonal (but not inverse)"""
    epsilon: double
    """Twisted mass parameter"""
    evmax: double
    """maximum of the eigenvalues of the ndeg twisted mass operator needed for fermionic forces"""

    twist_flavor: QudaTwistFlavorType
    """Twisted mass flavor"""

    laplace3D: int
    """omit this direction from laplace operator: x,y,z,t -> 0,1,2,3 (-1 is full 4D)"""
    covdev_mu: int
    """Apply forward/backward covariant derivative in direction mu(mu<=3)/mu-4(mu>3)"""

    tol: double
    """Solver tolerance in the L2 residual norm"""
    tol_restart: double
    """Solver tolerance in the L2 residual norm (used to restart InitCG)"""
    tol_hq: double
    """Solver tolerance in the heavy quark residual norm"""

    compute_true_res: int
    """Whether to compute the true residual post solve"""
    true_res: List[double, QUDA_MAX_MULTI_SRC]
    """Actual L2 residual norm achieved in the solver"""
    true_res_hq: List[double, QUDA_MAX_MULTI_SRC]
    """Actual heavy quark residual norm achieved in the solver"""
    maxiter: int
    """Maximum number of iterations in the linear solver"""
    reliable_delta: double
    """Reliable update tolerance"""
    reliable_delta_refinement: double
    """Reliable update tolerance used in post multi-shift solver refinement"""
    use_alternative_reliable: int
    """Whether to use alternative reliable updates"""
    use_sloppy_partial_accumulator: int
    """Whether to keep the partial solution accumuator in sloppy precision"""

    solution_accumulator_pipeline: int
    """
    This parameter determines how often we accumulate into the
    solution vector from the direction vectors in the solver.
    E.g., running with solution_accumulator_pipeline = 4, means we
    will update the solution vector every four iterations using the
    direction vectors from the prior four iterations.  This
    increases performance of mixed-precision solvers since it means
    less high-precision vector round-trip memory travel, but
    requires more low-precision memory allocation.
    """

    max_res_increase: int
    """
    This parameter determines how many consecutive reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge
    """

    max_res_increase_total: int
    """
    This parameter determines how many total reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge
    """

    max_hq_res_increase: int
    """
    This parameter determines how many consecutive heavy-quark
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge
    """

    max_hq_res_restart_total: int
    """
    This parameter determines how many total heavy-quark residual
    restarts we tolerate before terminating the solver, i.e., how long
    do we want to keep trying to converge
    """

    heavy_quark_check: int
    """After how many iterations shall the heavy quark residual be updated"""

    pipeline: int
    """Whether to use a pipelined solver with less global sums"""

    num_offset: int
    """Number of offsets in the multi-shift solver"""

    num_src: int
    """Number of sources in the multiple source solver"""

    num_src_per_sub_partition: int
    """Number of sources in the multiple source solver, but per sub-partition"""

    split_grid: List[int, QUDA_MAX_DIM]
    """
    The grid of sub-partition according to which the processor grid will be partitioned.
    Should have:
        split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3] * num_src_per_sub_partition == num_src.
    """

    overlap: int
    """Width of domain overlaps"""

    offset: List[double, QUDA_MAX_MULTI_SHIFT]
    """Offsets for multi-shift solver"""

    tol_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    """Solver tolerance for each offset"""

    tol_hq_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    """Solver tolerance for each shift when refinement is applied using the heavy-quark residual"""

    true_res_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    """Actual L2 residual norm achieved in solver for each offset"""

    iter_res_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    """Iterated L2 residual norm achieved in multi shift solver for each offset"""

    true_res_hq_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    """Actual heavy quark residual norm achieved in solver for each offset"""

    residue: List[double, QUDA_MAX_MULTI_SHIFT]
    """Residuals in the partial faction expansion"""

    compute_action: int
    """Whether we should evaluate the action after the linear solver"""

    action: List[double, 2]
    R"""
    Computed value of the bilinear action (complex-valued)
    invert: \phi^\dagger A^{-1} \phi
    multishift: \phi^\dagger r(x) \phi = \phi^\dagger (sum_k residue[k] * (A + offset[k])^{-1} ) \phi
    """

    solution_type: QudaSolutionType
    """Type of system to solve"""
    solve_type: QudaSolveType
    """How to solve it"""
    matpc_type: QudaMatPCType
    """The preconditioned matrix type"""
    dagger: QudaDagType
    """Whether we are using the Hermitian conjugate system or not"""
    mass_normalization: QudaMassNormalization
    """The mass normalization is being used by the caller"""
    solver_normalization: QudaSolverNormalization
    """The normalization desired in the solver"""

    preserve_source: QudaPreserveSource
    """Preserve the source or not in the linear solver (deprecated)"""

    cpu_prec: QudaPrecision
    """The precision used by the input fermion fields"""
    cuda_prec: QudaPrecision
    """The precision used by the QUDA solver"""
    cuda_prec_sloppy: QudaPrecision
    """The precision used by the QUDA sloppy operator"""
    cuda_prec_refinement_sloppy: QudaPrecision
    """The precision of the sloppy gauge field for the refinement step in multishift"""
    cuda_prec_precondition: QudaPrecision
    """The precision used by the QUDA preconditioner"""
    cuda_prec_eigensolver: QudaPrecision
    """The precision used by the QUDA eigensolver"""

    dirac_order: QudaDiracFieldOrder
    """The order of the input and output fermion fields"""

    gamma_basis: QudaGammaBasis
    """Gamma basis of the input and output host fields"""

    clover_location: QudaFieldLocation
    """The location of the clover field"""
    clover_cpu_prec: QudaPrecision
    """The precision used for the input clover field"""
    clover_cuda_prec: QudaPrecision
    """The precision used for the clover field in the QUDA solver"""
    clover_cuda_prec_sloppy: QudaPrecision
    """The precision used for the clover field in the QUDA sloppy operator"""
    clover_cuda_prec_refinement_sloppy: QudaPrecision
    """The precision of the sloppy clover field for the refinement step in multishift"""
    clover_cuda_prec_precondition: QudaPrecision
    """The precision used for the clover field in the QUDA preconditioner"""
    clover_cuda_prec_eigensolver: QudaPrecision
    """The precision used for the clover field in the QUDA eigensolver"""

    clover_order: QudaCloverFieldOrder
    """The order of the input clover field"""
    use_init_guess: QudaUseInitGuess
    """Whether to use an initial guess in the solver or not"""

    clover_csw: double
    """Csw coefficient of the clover term"""
    clover_coeff: double
    """Coefficient of the clover term"""
    clover_rho: double
    """Real number added to the clover diagonal (not to inverse)"""

    compute_clover_trlog: int
    """Whether to compute the trace log of the clover term"""
    trlogA: List[double, 2]
    """The trace log of the clover term (even/odd computed separately)"""

    compute_clover: int
    """Whether to compute the clover field"""
    compute_clover_inverse: int
    """Whether to compute the clover inverse field"""
    return_clover: int
    """Whether to copy back the clover matrix field"""
    return_clover_inverse: int
    """Whether to copy back the inverted clover matrix field"""

    verbosity: QudaVerbosity
    """The verbosity setting to use in the solver"""

    iter: int
    """The number of iterations performed by the solver"""
    gflops: double
    """The Gflops rate of the solver"""
    secs: double
    """The time taken by the solver"""
    energy: double
    """The energy consumed by the solver"""
    power: double
    """The mean power of the solver"""
    temp: double
    """The mean temperature of the device for the duration of the solve"""
    clock: double
    """The mean clock frequency of the device for the duration of the solve"""

    Nsteps: int
    """Number of steps in s-step algorithms"""

    gcrNkrylov: int
    """Maximum size of Krylov space used by solver"""

    # The following parameters are related to the solver
    # preconditioner, if enabled

    inv_type_precondition: QudaInverterType
    """
    The inner Krylov solver used in the preconditioner.  Set to
    QUDA_INVALID_INVERTER to disable the preconditioner entirely.
    """

    preconditioner: Pointer
    """Preconditioner instance, e.g., multigrid"""

    deflation_op: Pointer
    """Deflation instance"""

    eig_param: Pointer
    """defines deflation"""

    deflate: QudaBoolean
    """If true, deflate the initial guess"""

    dslash_type_precondition: QudaDslashType
    """Dirac Dslash used in preconditioner"""
    verbosity_precondition: QudaVerbosity
    """Verbosity of the inner Krylov solver"""

    tol_precondition: double
    """Tolerance in the inner solver"""

    maxiter_precondition: int
    """Maximum number of iterations allowed in the inner solver"""

    omega: double
    """Relaxation parameter used in GCR-DD (default = 1.0)"""

    ca_basis: QudaCABasis
    """Basis for CA algorithms"""

    ca_lambda_min: double
    """Minimum eigenvalue for Chebyshev CA basis"""

    ca_lambda_max: double
    """Maximum eigenvalue for Chebyshev CA basis"""

    ca_basis_precondition: QudaCABasis
    """Basis for CA algorithms in a preconditioned solver"""

    ca_lambda_min_precondition: double
    """Minimum eigenvalue for Chebyshev CA basis in a preconditioner solver"""

    ca_lambda_max_precondition: double
    """Maximum eigenvalue for Chebyshev CA basis in a preconditioner solver"""

    precondition_cycle: int
    """Number of preconditioner cycles to perform per iteration"""

    schwarz_type: QudaSchwarzType
    """Whether to use additive or multiplicative Schwarz preconditioning"""

    accelerator_type_precondition: QudaAcceleratorType
    """The type of accelerator type to use for preconditioner"""

    # The following parameters are the ones used to perform the adaptive MADWF in MSPCG
    # See section 3.3 of [arXiv:2104.05615]

    madwf_diagonal_suppressor: double
    """The diagonal constant to suppress the low modes when performing 5D transfer"""

    madwf_ls: int
    """The target MADWF Ls to be used in the accelerator"""

    madwf_null_miniter: int
    """The minimum number of iterations after which to generate the null vectors for MADWF"""

    madwf_null_tol: double
    """The maximum tolerance after which to generate the null vectors for MADWF"""

    madwf_train_maxiter: int
    """The maximum number of iterations for the training iterations"""

    madwf_param_load: QudaBoolean
    """Whether to load the MADWF parameters from the file system"""

    madwf_param_save: QudaBoolean
    """Whether to save the MADWF parameters to the file system"""

    madwf_param_infile: bytes[256]
    """Path to load from the file system"""

    madwf_param_outfile: bytes[256]
    """Path to save to the file system"""

    residual_type: QudaResidualType
    """
    Whether to use the L2 relative residual, Fermilab heavy-quark
    residual, or both to determine convergence.  To require that both
    stopping conditions are satisfied, use a bitwise OR as follows:

    p.residual_type = (QudaResidualType) (QUDA_L2_RELATIVE_RESIDUAL
                                        | QUDA_HEAVY_QUARK_RESIDUAL);
    """

    # Parameters for deflated solvers
    cuda_prec_ritz: QudaPrecision
    """The precision of the Ritz vectors"""
    n_ev: int
    """
    How many vectors to compute after one solve
    for eigCG recommended values 8 or 16
    """
    max_search_dim: int
    """
    EeigCG  : Search space dimension
    gmresdr : Krylov subspace dimension
    """
    rhs_idx: int
    """For systems with many RHS: current RHS index"""
    deflation_grid: int
    """Specifies deflation space volume: total number of eigenvectors is n_ev*deflation_grid"""
    eigenval_tol: double
    """eigCG: selection criterion for the reduced eigenvector set"""
    eigcg_max_restarts: int
    """mixed precision eigCG tuning parameter:  minimum search vector space restarts"""
    max_restart_num: int
    """initCG tuning parameter:  maximum restarts"""
    inc_tol: double
    """initCG tuning parameter:  tolerance for cg refinement corrections in the deflation stage"""

    make_resident_solution: int
    """Whether to make the solution vector(s) after the solve"""

    use_resident_solution: int
    """Whether to use the resident solution vector(s)"""

    chrono_make_resident: int
    """Whether to use the solution vector to augment the chronological basis"""

    chrono_replace_last: int
    """Whether the solution should replace the last entry in the chronology"""

    chrono_use_resident: int
    """Whether to use the resident chronological basis"""

    chrono_max_dim: int
    """The maximum length of the chronological history to store"""

    chrono_index: int
    """The index to indicate which chrono history we are augmenting"""

    chrono_precision: QudaPrecision
    """Precision to store the chronological basis in"""

    extlib_type: QudaExtLibType
    """Which external library to use in the linear solvers (Eigen)"""

    native_blas_lapack: QudaBoolean
    """Whether to use the platform native or generic BLAS / LAPACK"""

    use_mobius_fused_kernel: QudaBoolean
    """Whether to use fused kernels for mobius"""

    # Parameters for distance preconditioning algorithm proposed in arXiv:1006.4028,
    # which is useful to solve a precise heavy quark propagator.
    # alpha0 and t0 follow Eq.(9) in the article.

    distance_pc_alpha0: double
    """The alpha0 parameter for distance preconditioning, related to the pseudoscalar meson mass"""
    distance_pc_t0: int
    """The t0 parameter for distance preconditioning, the timeslice where the source is located"""

class QudaEigParam:
    """
    Parameter set for solving eigenvalue problems.
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size"""

    # EIGENSOLVER PARAMS

    invert_param: QudaInvertParam
    """Used to store information pertinent to the operator"""

    eig_type: QudaEigType
    """Type of eigensolver algorithm to employ"""

    use_poly_acc: QudaBoolean
    """Use Polynomial Acceleration"""

    poly_deg: int
    """Degree of the Chebysev polynomial"""

    a_min: double
    """Range used in polynomial acceleration"""
    a_max: double
    """Range used in polynomial acceleration"""

    preserve_deflation: QudaBoolean
    """
    Whether to preserve the deflation space between solves.  If
    true, the space will be stored in an instance of the
    deflation_space struct, pointed to by preserve_deflation_space
    """

    preserve_deflation_space: Pointer
    """
    This is where we store the deflation space.  This will point
    to an instance of deflation_space. When a deflated solver is enabled, the deflation space will be obtained from this.
    """

    preserve_evals: QudaBoolean
    """
    If we restore the deflation space, this boolean indicates
    whether we are also preserving the evalues or recomputing
    them.  For example if a different mass shift is being used
    than the one used to generate the space, then this should be
    false, but preserve_deflation would be true
    """

    # What type of Dirac operator we are using
    use_dagger: QudaBoolean
    """
    If !(use_norm_op) && !(use_dagger) use M.\n
    If use_dagger, use Mdag\n
    If use_norm_op && use_dagger use MMdag.\n
    """
    use_norm_op: QudaBoolean
    """
    If !(use_norm_op) && !(use_dagger) use M.\n
    If use_norm_op, use MdagM\n
    If use_norm_op && use_dagger use MMdag.\n
    """
    use_pc: QudaBoolean
    """
    If use_pc for any, then use the even-odd pc version
    """

    use_eigen_qr: QudaBoolean
    """Use Eigen routines to eigensolve the upper Hessenberg via QR"""

    compute_svd: QudaBoolean
    """Performs an MdagM solve, then constructs the left and right SVD."""

    compute_gamma5: QudaBoolean
    R"""
    Performs the \gamma_5 OP solve by Post multipling the eignvectors with
    \gamma_5 before computing the eigenvalues
    """

    require_convergence: QudaBoolean
    """If true, the solver will error out if the convergence criteria are not met"""

    spectrum: QudaEigSpectrumType
    """Which part of the spectrum to solve"""

    n_ev: int
    """Size of the eigenvector search space"""
    n_kr: int
    """Total size of Krylov space"""
    nLockedMax: int
    """Max number of locked eigenpairs (deduced at runtime)"""
    n_conv: int
    """Number of requested converged eigenvectors"""
    n_ev_deflate: int
    """Number of requested converged eigenvectors to use in deflation"""
    tol: double
    """Tolerance on the least well known eigenvalue's residual"""
    qr_tol: double
    """Tolerance on the QR iteration"""
    check_interval: int
    """For IRLM/IRAM, check every nth restart"""
    max_restarts: int
    """For IRLM/IRAM, quit after n restarts"""
    batched_rotate: int
    """For the Ritz rotation, the maximal number of extra vectors the solver may allocate"""
    block_size: int
    """For block method solvers, the block size"""
    compute_evals_batch_size: int
    """The batch size used when computing eigenvalues"""
    max_ortho_attempts: int
    """For block method solvers, quit after n attempts at block orthonormalisation"""
    ortho_block_size: int
    """For hybrid modifeld Gram-Schmidt orthonormalisations"""

    arpack_check: QudaBoolean
    """In the test function, cross check the device result against ARPACK"""
    arpack_logfile: bytes[512]
    """For Arpack cross check, name of the Arpack logfile"""

    QUDA_logfile: bytes[512]
    """Name of the QUDA logfile (residua, upper Hessenberg/tridiag matrix updates)"""

    # EIG-CG PARAMS

    nk: int
    np: int

    import_vectors: QudaBoolean
    """Whether to load eigenvectors"""

    cuda_prec_ritz: QudaPrecision
    """The precision of the Ritz vectors"""

    mem_type_ritz: QudaMemoryType
    """The memory type used to keep the Ritz vectors"""

    location: QudaFieldLocation
    """Location where deflation should be done"""

    run_verify: QudaBoolean
    """Whether to run the verification checks once set up is complete"""

    vec_infile: bytes[256]
    """Filename prefix where to load the null-space vectors"""

    vec_outfile: bytes[256]
    """Filename prefix for where to save the null-space vectors"""

    save_prec: QudaPrecision
    """The precision with which to save the vectors"""

    io_parity_inflate: QudaBoolean
    """
    Whether to inflate single-parity eigen-vector I/O to a full
    field (e.g., enabling this is required for compatability with
    MILC I/O)
    """

    partfile: QudaBoolean
    """Whether to save eigenvectors in QIO singlefile or partfile format"""

    extlib_type: QudaExtLibType
    """Which external library to use in the deflation operations (Eigen)"""

class QudaMultigridParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size"""

    invert_param: QudaInvertParam

    eig_param: List[QudaEigParam, QUDA_MAX_MG_LEVEL]

    n_level: int
    """Number of multigrid levels"""

    geo_block_size: List[List[int, QUDA_MAX_DIM], QUDA_MAX_MG_LEVEL]
    """Geometric block sizes to use on each level"""

    spin_block_size: List[int, QUDA_MAX_MG_LEVEL]
    """Spin block sizes to use on each level"""

    n_vec: List[int, QUDA_MAX_MG_LEVEL]
    """Number of null-space vectors to use on each level"""

    precision_null: List[QudaPrecision, QUDA_MAX_MG_LEVEL]
    """Precision to store the null-space vectors in (post block orthogonalization)"""

    n_block_ortho: List[int, QUDA_MAX_MG_LEVEL]
    """Number of times to repeat Gram-Schmidt in block orthogonalization"""

    block_ortho_two_pass: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Whether to do passes at block orthogonalize in fixed point for improved accuracy"""

    verbosity: List[QudaVerbosity, QUDA_MAX_MG_LEVEL]
    """Verbosity on each level of the multigrid"""

    setup_use_mma: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Setup MMA usage on each level of the multigrid"""

    dslash_use_mma: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Dslash MMA usage on each level of the multigrid"""

    setup_inv_type: List[QudaInverterType, QUDA_MAX_MG_LEVEL]
    """Inverter to use in the setup phase"""

    n_vec_batch: List[int, QUDA_MAX_MG_LEVEL]
    """Solver batch size to use in the setup phase"""

    num_setup_iter: List[int, QUDA_MAX_MG_LEVEL]
    """Number of setup iterations"""

    setup_tol: List[double, QUDA_MAX_MG_LEVEL]
    """Tolerance to use in the setup phase"""

    setup_maxiter: List[int, QUDA_MAX_MG_LEVEL]
    """Maximum number of iterations for each setup solver"""

    setup_maxiter_refresh: List[int, QUDA_MAX_MG_LEVEL]
    """Maximum number of iterations for refreshing the null-space vectors"""

    setup_ca_basis: List[QudaCABasis, QUDA_MAX_MG_LEVEL]
    """Basis to use for CA solver setup"""

    setup_ca_basis_size: List[int, QUDA_MAX_MG_LEVEL]
    """Basis size for CA solver setup"""

    setup_ca_lambda_min: List[double, QUDA_MAX_MG_LEVEL]
    """Minimum eigenvalue for Chebyshev CA basis"""

    setup_ca_lambda_max: List[double, QUDA_MAX_MG_LEVEL]
    """Maximum eigenvalue for Chebyshev CA basis"""

    setup_type: QudaSetupType
    """Null-space type to use in the setup phase"""

    pre_orthonormalize: QudaBoolean
    """Pre orthonormalize vectors in the setup phase"""

    post_orthonormalize: QudaBoolean
    """Post orthonormalize vectors in the setup phase"""

    coarse_solver: List[QudaInverterType, QUDA_MAX_MG_LEVEL]
    """The solver that wraps around the coarse grid correction and smoother"""

    coarse_solver_tol: List[double, QUDA_MAX_MG_LEVEL]
    """Tolerance for the solver that wraps around the coarse grid correction and smoother"""

    coarse_solver_maxiter: List[int, QUDA_MAX_MG_LEVEL]
    """Maximum number of iterations for the solver that wraps around the coarse grid correction and smoother"""

    coarse_solver_ca_basis: List[QudaCABasis, QUDA_MAX_MG_LEVEL]
    """Basis to use for CA coarse solvers"""

    coarse_solver_ca_basis_size: List[int, QUDA_MAX_MG_LEVEL]
    """Basis size for CA coarse solvers"""

    coarse_solver_ca_lambda_min: List[double, QUDA_MAX_MG_LEVEL]
    """Minimum eigenvalue for Chebyshev CA basis"""

    coarse_solver_ca_lambda_max: List[double, QUDA_MAX_MG_LEVEL]
    """Maximum eigenvalue for Chebyshev CA basis"""

    smoother: List[QudaInverterType, QUDA_MAX_MG_LEVEL]
    """Smoother to use on each level"""

    smoother_tol: List[double, QUDA_MAX_MG_LEVEL]
    """Tolerance to use for the smoother / solver on each level"""

    nu_pre: List[int, QUDA_MAX_MG_LEVEL]
    """Number of pre-smoother applications on each level"""

    nu_post: List[int, QUDA_MAX_MG_LEVEL]
    """Number of post-smoother applications on each level"""

    smoother_solver_ca_basis: List[QudaCABasis, QUDA_MAX_MG_LEVEL]
    """Basis to use for CA smoother solvers"""

    smoother_solver_ca_lambda_min: List[double, QUDA_MAX_MG_LEVEL]
    """Minimum eigenvalue for Chebyshev CA smoother basis"""

    smoother_solver_ca_lambda_max: List[double, QUDA_MAX_MG_LEVEL]
    """Maximum eigenvalue for Chebyshev CA smoother basis"""

    omega: List[double, QUDA_MAX_MG_LEVEL]
    """Over/under relaxation factor for the smoother at each level"""

    smoother_halo_precision: List[QudaPrecision, QUDA_MAX_MG_LEVEL]
    """Precision to use for halo communication in the smoother"""

    smoother_schwarz_type: List[QudaSchwarzType, QUDA_MAX_MG_LEVEL]
    """Whether to use additive or multiplicative Schwarz preconditioning in the smoother"""

    smoother_schwarz_cycle: List[int, QUDA_MAX_MG_LEVEL]
    """Number of Schwarz cycles to apply"""

    coarse_grid_solution_type: List[QudaSolutionType, QUDA_MAX_MG_LEVEL]
    """
    The type of residual to send to the next coarse grid, and thus the
    type of solution to receive back from this coarse grid
    """

    smoother_solve_type: List[QudaSolveType, QUDA_MAX_MG_LEVEL]
    """The type of smoother solve to do on each grid (e/o preconditioning or not)"""

    cycle_type: List[QudaMultigridCycleType, QUDA_MAX_MG_LEVEL]
    """The type of multigrid cycle to perform at each level"""

    global_reduction: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Whether to use global reductions or not for the smoother / solver at each level"""

    location: List[QudaFieldLocation, QUDA_MAX_MG_LEVEL]
    """Location where each level should be done"""

    setup_location: List[QudaFieldLocation, QUDA_MAX_MG_LEVEL]
    """Location where the coarse-operator construction will be computedn"""

    use_eig_solver: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Whether to use eigenvectors for the nullspace or, if the coarsest instance deflate"""

    setup_minimize_memory: QudaBoolean
    """
    Minimize device memory allocations during the adaptive setup,
    placing temporary fields in mapped memory instad of device
    memory
    """

    compute_null_vector: QudaComputeNullVector
    """Whether to compute the null vectors or reload them"""

    generate_all_levels: QudaBoolean
    """Whether to generate on all levels or just on level 0"""

    run_verify: QudaBoolean
    """Whether to run the verification checks once set up is complete"""

    run_low_mode_check: QudaBoolean
    """Whether to run null Vs eigen vector overlap checks once set up is complete"""

    run_oblique_proj_check: QudaBoolean
    """Whether to run null vector oblique checks once set up is complete"""

    vec_load: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Whether to load the null-space vectors to disk (requires QIO)"""

    vec_infile: List[bytes[256], QUDA_MAX_MG_LEVEL]
    """Filename prefix where to load the null-space vectors"""

    vec_store: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Whether to store the null-space vectors to disk (requires QIO)"""

    vec_outfile: List[bytes[256], QUDA_MAX_MG_LEVEL]
    """Filename prefix for where to save the null-space vectors"""

    mg_vec_partfile: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    """Whether to store the null-space vectors in singlefile or partfile format"""

    coarse_guess: QudaBoolean
    """Whether to use and initial guess during coarse grid deflation"""

    preserve_deflation: QudaBoolean
    """Whether to preserve the deflation space during MG update"""

    mu_factor: List[double, QUDA_MAX_MG_LEVEL]
    """Multiplicative factor for the mu parameter"""

    transfer_type: List[QudaTransferType, QUDA_MAX_MG_LEVEL]
    """Boolean for aggregation type, implies staggered or not"""

    allow_truncation: QudaBoolean
    """Whether or not to let MG coarsening drop improvements, for ex dropping long links in small aggregation dimensions"""

    staggered_kd_dagger_approximation: QudaBoolean
    """Whether or not to use the dagger approximation for the KD preconditioned operator"""

    thin_update_only: QudaBoolean
    """Whether to do a full (false) or thin (true) update in the context of updateMultigridQuda"""

class QudaGaugeObservableParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct"""
    su_project: QudaBoolean
    """Whether to project onto the manifold prior to measurement"""
    compute_plaquette: QudaBoolean
    """Whether to compute the plaquette"""
    plaquette: List[double, 3]
    """Total, spatial and temporal field energies, respectively"""
    compute_polyakov_loop: QudaBoolean
    """Whether to compute the temporal Polyakov loop"""
    ploop: List[double, 2]
    """Real and imaginary part of temporal Polyakov loop"""
    compute_gauge_loop_trace: QudaBoolean
    """Whether to compute gauge loop traces"""
    traces: Pointer[complex128]
    """Individual complex traces of each loop"""
    input_path_buff: Pointers[int32]
    """Array of paths"""
    path_length: Pointer[int32]
    """Length of each path"""
    loop_coeff: Pointer[float64]
    """Multiplicative factor for each loop"""
    num_paths: int
    """Total number of paths"""
    max_length: int
    """Maximum length of any path"""
    factor: double
    """Global multiplicative factor to apply to each loop trace"""
    compute_qcharge: QudaBoolean
    """Whether to compute the topological charge and field energy"""
    qcharge: double
    """Computed topological charge"""
    energy: List[double, 3]
    """Total, spatial and temporal field energies, respectively"""
    compute_qcharge_density: QudaBoolean
    """Whether to compute the topological charge density"""
    qcharge_density: Pointer
    """Pointer to host array of length volume where the q-charge density will be copied"""

    remove_staggered_phase: QudaBoolean
    """
    Whether or not the resident gauge field has staggered phases applied and if they should
    be removed; this was needed for the Polyakov loop calculation when called through MILC,
    with the underlying issue documented https://github.com/lattice/quda/issues/1315
    """

class QudaGaugeSmearParam:
    def __init__(self) -> None: ...
    # def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct"""
    n_steps: int
    """The total number of smearing steps to perform."""
    epsilon: double
    """
    Serves as one of the coefficients in Over Improved Stout smearing, or as the step size in
    Wilson/Symanzik flow
    """
    alpha: double
    """The single coefficient used in APE smearing"""
    rho: double
    """Serves as one of the coefficients used in Over Improved Stout smearing, or as the single coefficient used in Stout"""
    alpha1: double
    """The coefficient used in HYP smearing step 3 (will not be used in 3D smearing)"""
    alpha2: double
    """The coefficient used in HYP smearing step 2"""
    alpha3: double
    """The coefficient used in HYP smearing step 1"""
    meas_interval: int
    """Perform the requested measurements on the gauge field at this interval"""
    smear_type: QudaGaugeSmearType
    """The smearing type to perform"""
    restart: QudaBoolean
    """Used to restart the smearing from existing gaugeSmeared"""
    t0: double
    """Starting flow time for Wilson flow"""
    dir_ignore: int
    """
    The direction to be ignored by the smearing algorithm
    A negative value means 3D for APE/STOUT and 4D for OVRIMP_STOUT/HYP
    """

class QudaBLASParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    """Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct"""

    blas_type: QudaBLASType
    """Type of BLAS computation to perfrom"""

    # GEMM params
    trans_a: QudaBLASOperation
    """operation op(A) that is non- or (conj.) transpose."""
    trans_b: QudaBLASOperation
    """operation op(B) that is non- or (conj.) transpose."""
    m: int
    """number of rows of matrix op(A) and C."""
    n: int
    """number of columns of matrix op(B) and C."""
    k: int
    """number of columns of op(A) and rows of op(B)."""
    lda: int
    """leading dimension of two-dimensional array used to store the matrix A."""
    ldb: int
    """leading dimension of two-dimensional array used to store matrix B."""
    ldc: int
    """leading dimension of two-dimensional array used to store matrix C."""
    a_offset: int
    """position of the A array from which begin read/write."""
    b_offset: int
    """position of the B array from which begin read/write."""
    c_offset: int
    """position of the C array from which begin read/write."""
    a_stride: int
    """stride of the A array in strided(batched) mode"""
    b_stride: int
    """stride of the B array in strided(batched) mode"""
    c_stride: int
    """stride of the C array in strided(batched) mode"""
    alpha: double_complex
    """scalar used for multiplication."""
    beta: double_complex
    """scalar used for multiplication. If beta==0, C does not have to be a valid input."""

    # LU inversion params
    inv_mat_size: int
    """The rank of the square matrix in the LU inversion"""

    # Common params
    batch_count: int
    """number of pointers contained in arrayA, arrayB and arrayC."""
    data_type: QudaBLASDataType
    """Specifies if using S(C) or D(Z) BLAS type"""
    data_order: QudaBLASDataOrder
    """Specifies if using Row or Column major"""

def setVerbosityQuda(verbosity: QudaVerbosity, prefix: bytes) -> None:
    """
    Set parameters related to status reporting.

    In typical usage, this function will be called once (or not at
    all) just before the call to initQuda(), but it's valid to call
    it any number of times at any point during execution.  Prior to
    the first time it's called, the parameters take default values
    as indicated below.

    @param verbosity:
        Default verbosity, ranging from QUDA_SILENT to
        QUDA_DEBUG_VERBOSE.  Within a solver, this
        parameter is overridden by the "verbosity"
        member of QudaInvertParam.  The default value
        is QUDA_SUMMARIZE.

    @param prefix:
        String to prepend to all messages from QUDA.  This
        defaults to the empty string (""), but you may
        wish to specify something like "QUDA: " to
        distinguish QUDA's output from that of your
        application.
    """
    ...

def initCommsGridQuda(nDim: int, dims: List[int, 4], grid_map: bytes) -> None:
    """
    Declare the grid mapping ("logical topology" in QMP parlance)
    used for communications in a multi-GPU grid.  This function
    should be called prior to initQuda().  The only case in which
    it's optional is when QMP is used for communication and the
    logical topology has already been declared by the application.

    @param nDim:
        Number of grid dimensions.  "4" is the only supported
        value currently.

    @param dims:
        Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
        must equal the total number of MPI ranks or QMP nodes.
    """
    ...

def initQudaDevice(device: int) -> None:
    """
    Initialize the library.  This is a low-level interface that is
    called by initQuda.  Calling initQudaDevice requires that the
    user also call initQudaMemory before using QUDA.

    @param device:
        CUDA device number to use.  In a multi-GPU build,
        this parameter may either be set explicitly on a
        per-process basis or set to -1 to enable a default
        allocation of devices to processes.
    """
    ...

def initQudaMemory() -> None:
    """
    Initialize the library persistant memory allocations (both host
    and device).  This is a low-level interface that is called by
    initQuda.  Calling initQudaMemory requires that the user has
    previously called initQudaDevice.
    """
    ...

def initQuda(device: int) -> None:
    """
    Initialize the library.  This function is actually a wrapper
    around calls to initQudaDevice() and initQudaMemory().

    @param device:
        CUDA device number to use.  In a multi-GPU build,
        this parameter may either be set explicitly on a
        per-process basis or set to -1 to enable a default
        allocation of devices to processes.
    """
    ...

def endQuda() -> None:
    """
    Finalize the library.
    """
    ...

def loadGaugeQuda(h_gauge: NDArray[_fields], param: QudaGaugeParam) -> None:
    """
    Load the gauge field from the host.

    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param param:
        Contains all metadata regarding host and device storage
    """
    ...

def freeGaugeQuda() -> None:
    """
    Free QUDA's internal copy of the gauge field.
    """
    ...

def freeUniqueGaugeQuda(link_type: QudaLinkType) -> None:
    """
    Free a unique type (Wilson, HISQ fat, HISQ long, smeared) of internal gauge field.

    @param link_type[in]:
        Type of link type to free up
    """
    ...

def freeGaugeSmearedQuda() -> None:
    """
    Free QUDA's internal smeared gauge field.
    """
    ...

def saveGaugeQuda(h_gauge: NDArray[_fields], param: QudaGaugeParam) -> None:
    """
    Save the gauge field to the host.

    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param param:
        Contains all metadata regarding host and device storage
    """
    ...

def loadCloverQuda(h_clover: NDArray[_field], h_clovinv: NDArray[_field], inv_param: QudaInvertParam) -> None:
    """
    Load the clover term and/or the clover inverse from the host.
    Either h_clover or h_clovinv may be set to NULL.

    @param h_clover:
        Base pointer to host clover field
    @param h_cloverinv:
        Base pointer to host clover inverse field
    @param inv_param:
        Contains all metadata regarding host and device storage
    """
    ...

def freeCloverQuda() -> None:
    """
    Free QUDA's internal copy of the clover term and/or clover inverse.
    """
    ...

def eigensolveQuda(h_evecs: NDArray[_fields], h_evals: NDArray[complex128], param: QudaEigParam):
    """
    Perform the eigensolve. The problem matrix is defined by the invert param, the
    mode of solution is specified by the eig param. It is assumed that the gauge
    field has already been loaded via  loadGaugeQuda().

    @param h_evecs:
        Array of pointers to application eigenvectors
    @param h_evals:
        Host side eigenvalues
    @param param:
        Contains all metadata regarding the type of solve.
    """
    ...

def invertQuda(h_x: NDArray[_field], h_b: NDArray[_field], param: QudaInvertParam) -> None:
    """
    Perform the solve, according to the parameters set in param.  It
    is assumed that the gauge field has already been loaded via
    loadGaugeQuda().

    @param h_x:
        Solution spinor field
    @param h_b:
        Source spinor field
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters
    """

def invertMultiSrcQuda(_hp_x: NDArray[_fields], _hp_b: NDArray[_fields], param: QudaInvertParam) -> None:
    """
    Perform the solve like @invertQuda but for multiple rhs by spliting the comm grid into
    sub-partitions: each sub-partition invert one or more rhs'.
    The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    Unlike @invertQuda, the interface also takes the host side gauge as input. The gauge pointer and
    gauge_param are used if for inv_param split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3]
    is larger than 1, in which case gauge field is not required to be loaded beforehand; otherwise
    this interface would just work as @invertQuda, which requires gauge field to be loaded beforehand,
    and the gauge field pointer and gauge_param are not used.

    @param _hp_x:
        Array of solution spinor fields
    @param _hp_b:
        Array of source spinor fields
    @param param:
        Contains all metadata regarding host and device storage and solver parameters
    """
    ...

def invertMultiShiftQuda(_hp_x: NDArray[_fields], _hp_b: NDArray[_field], param: QudaInvertParam) -> None:
    """
    Perform the solve like @invertQuda but for multiple rhs by spliting the comm grid into sub-partitions:
    each sub-partition invert one or more rhs'.
    The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    Unlike @invertQuda, the interface also takes the host side gauge as input. The gauge pointer and
    gauge_param are used if for inv_param split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3]
    is larger than 1, in which case gauge field is not required to be loaded beforehand; otherwise
    this interface would just work as @invertQuda, which requires gauge field to be loaded beforehand,
    and the gauge field pointer and gauge_param are not used.

    @param _hp_x:
        Array of solution spinor fields
    @param _hp_b:
        Array of source spinor fields
    @param param:
        Contains all metadata regarding host and device storage and solver parameters
    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param gauge_param:
        Contains all metadata regarding host and device storage for gauge field
    """
    ...

def newMultigridQuda(param: QudaMultigridParam) -> Pointer:
    """
    Setup the multigrid solver, according to the parameters set in param.  It
    is assumed that the gauge field has already been loaded via
    loadGaugeQuda().

    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters
    """
    ...

def destroyMultigridQuda(mg_instance: Pointer) -> None:
    """
    Free resources allocated by the multigrid solver

    @param mg_instance:
        Pointer to instance of multigrid_solver
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters
    """
    ...

def updateMultigridQuda(mg_instance: Pointer, param: QudaMultigridParam) -> None:
    """
    Updates the multigrid preconditioner for the new gauge / clover field

    @param mg_instance:
        Pointer to instance of multigrid_solver
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters, of note contains a flag specifying whether
        to do a full update or a thin update.
    """
    ...

def dumpMultigridQuda(mg_instance: Pointer, param: QudaMultigridParam) -> None:
    """
    Dump the null-space vectors to disk

    @param[in] mg_instance:
        Pointer to the instance of multigrid_solver
    @param[in] param:
        Contains all metadata regarding host and device
        storage and solver parameters (QudaMultigridParam::vec_outfile
        sets the output filename prefix).
    """
    ...

def dslashQuda(h_out: NDArray[_field], h_in: NDArray[_field], inv_param: QudaInvertParam, parity: QudaParity) -> None:
    """
    Apply the Dslash operator (D_{eo} or D_{oe}).

    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    @param parity:
        The destination parity of the field
    """
    ...

def dslashMultiSrcQuda(
    _hp_x: NDArray[_fields], _hp_b: NDArray[_fields], param: QudaInvertParam, parity: QudaParity
) -> None:
    """
    Perform the solve like @dslashQuda but for multiple rhs by spliting the comm grid into
    sub-partitions: each sub-partition does one or more rhs'.
    The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    Unlike @invertQuda, the interface also takes the host side gauge as
    input - gauge field is not required to be loaded beforehand.

    @param _hp_x:
        Array of solution spinor fields
    @param _hp_b:
        Array of source spinor fields
    @param param:
        Contains all metadata regarding host and device storage and solver parameters
    @param parity:
        Parity to apply dslash on
    """

def cloverQuda(
    h_out: NDArray[_field], h_in: NDArray[_field], inv_param: QudaInvertParam, parity: QudaParity, inverse: int
) -> None:
    """
    Apply the clover operator or its inverse.

    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    @param parity:
        The source and destination parity of the field
    @param inverse:
        Whether to apply the inverse of the clover term
    """
    ...

def MatQuda(h_out: NDArray[_field], h_in: NDArray[_field], inv_param: QudaInvertParam) -> None:
    """
    Apply the full Dslash matrix, possibly even/odd preconditioned.

    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    """
    ...

def MatDagMatQuda(h_out: NDArray[_field], h_in: NDArray[_field], inv_param: QudaInvertParam) -> None:
    R"""
    Apply M^{\dag}M, possibly even/odd preconditioned.

    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    """
    ...

def computeKSLinkQuda(
    fatlink: NDArray[_fields],
    longlink: NDArray[_fields],
    ulink: NDArray[_fields],
    inlink: NDArray[_fields],
    path_coeff: NDArray[float64],
    param: QudaGaugeParam,
) -> None:
    """ """
    ...

def computeTwoLinkQuda(twolink: NDArray[_fields], inlink: NDArray[_fields], param: QudaGaugeParam) -> None:
    """
    Compute two-link field

    @param[out] twolink:
        computed two-link field
    @param[in] inlink:
        the external field
    @param[in] param:
        Contains all metadata regarding host and device
        storage
    """
    ...

def momResidentQuda(mom: NDArray[_field], param: QudaGaugeParam) -> None:
    """
    Either downloads and sets the resident momentum field, or uploads
    and returns the resident momentum field

    @param[in,out] mom:
        The external momentum field
    @param[in] param:
        The parameters of the external field
    """
    ...

def computeGaugeForceQuda(
    mom: NDArray[_fields],
    sitelink: NDArray[_fields],
    input_path_buf: NDArray[int32],
    path_length: NDArray[int32],
    loop_coeff: NDArray[float64],
    num_paths: int,
    max_length: int,
    dt: double,
    qudaGaugeParam: QudaGaugeParam,
) -> int:
    """
    Compute the gauge force and update the momentum field

    @param[in,out] mom:
        The momentum field to be updated
    @param[in] sitelink:
        The gauge field from which we compute the force
    @param[in] input_path_buf:
        [dim][num_paths][path_length]
    @param[in] path_length:
        One less that the number of links in a loop (e.g., 3 for a staple)
    @param[in] loop_coeff:
        Coefficients of the different loops in the Symanzik action
    @param[in] num_paths:
        How many contributions from path_length different "staples"
    @param[in] max_length:
        The maximum number of non-zero of links in any path in the action
    @param[in] dt:
        The integration step size (for MILC this is dt*beta/3)
    @param[in] param:
        The parameters of the external fields and the computation settings
    """
    ...

def computeGaugePathQuda(
    out: NDArray[_fields],
    sitelink: NDArray[_fields],
    input_path_buf: NDArray[int32],
    path_length: NDArray[int32],
    loop_coeff: NDArray[float64],
    num_paths: int,
    max_length: int,
    dt: double,
    qudaGaugeParam: QudaGaugeParam,
) -> int:
    """
    Compute the product of gauge links along a path and add to/overwrite the output field

    @param[in,out] out:
        The output field to be updated
    @param[in] sitelink:
        The gauge field from which we compute the products of gauge links
    @param[in] input_path_buf:
        [dim][num_paths][path_length]
    @param[in] path_length:
        One less that the number of links in a loop (e.g., 3 for a staple)
    @param[in] loop_coeff:
        Coefficients of the different loops in the Symanzik action
    @param[in] num_paths:
        How many contributions from path_length different "staples"
    @param[in] max_length
        The maximum number of non-zero of links in any path in the action
    @param[in] dt:
        The integration step size (for MILC this is dt*beta/3)
    @param[in] param:
        The parameters of the external fields and the computation settings
    """
    ...

def computeGaugeLoopTraceQuda(
    traces: NDArray[complex128],
    input_path_buf: NDArray[int32],
    path_length: NDArray[int32],
    loop_coeff: NDArray[float64],
    num_paths: int,
    max_length: int,
    factor: double,
) -> None:
    """
    Compute the traces of products of gauge links along paths using the resident field

    @param[in,out] traces:
        The computed traces
    @param[in] input_path_buf:
        [num_paths][path_length]
    @param[in] path_length:
        The number of links in each loop
    @param[in] loop_coeff:
        Multiplicative coefficients for each loop
    @param[in] num_paths:
        Total number of loops
    @param[in] max_length:
        The maximum number of non-zero of links in any path in the action
    @param[in] factor:
        An overall normalization factor
    """
    ...

def updateGaugeFieldQuda(
    gauge: NDArray[_fields], momentum: NDArray[_fields], dt: double, conj_mom: int, exact: int, param: QudaGaugeParam
) -> None:
    """
    Evolve the gauge field by step size dt, using the momentum field
    I.e., Evalulate U(t+dt) = e(dt pi) U(t)

    @param gauge:
        The gauge field to be updated
    @param momentum:
        The momentum field
    @param dt:
        The integration step size step
    @param conj_mom:
        Whether to conjugate the momentum matrix
    @param exact:
        Whether to use an exact exponential or Taylor expand
    @param param:
        The parameters of the external fields and the computation settings
    """
    ...

def staggeredPhaseQuda(gauge_h: NDArray[_fields], param: QudaGaugeParam) -> None:
    """
    Apply the staggered phase factors to the gauge field.  If the
    imaginary chemical potential is non-zero then the phase factor
    exp(imu/T) will be applied to the links in the temporal
    direction.

    @param gauge_h:
        The gauge field
    @param param:
        The parameters of the gauge field
    """
    ...

def projectSU3Quda(gauge_h: NDArray[_fields], tol: double, param: QudaGaugeParam) -> None:
    """
    Project the input field on the SU(3) group.  If the target
    tolerance is not met, this routine will give a runtime error.

    @param gauge_h:
        The gauge field to be updated
    @param tol:
        The tolerance to which we iterate
    @param param:
        The parameters of the gauge field
    """
    ...

def momActionQuda(momentum: NDArray[_field], param: QudaGaugeParam) -> double:
    """
    Evaluate the momentum contribution to the Hybrid Monte Carlo
    action.

    @param momentum:
        The momentum field
    @param param:
        The parameters of the external fields and the computation settings
    @return:
        momentum action
    """
    ...

def createCloverQuda(param: QudaInvertParam) -> None:
    """
    Compute the clover field and its inverse from the resident gauge field.

    @param param:
        The parameters of the clover field to create
    """
    ...

def computeCloverForceQuda(
    mom: NDArray[_fields],
    dt: double,
    x: NDArray[_fields],
    coeff: NDArray[float64],
    kappa2: double,
    ck: double,
    nvector: int,
    multiplicity: double,
    gauge_param: QudaGaugeParam,
    inv_param: QudaInvertParam,
) -> None:
    """
    Compute the clover force contributions from a set of partial
    fractions stemming from a rational approximation suitable for use
    within MILC.

    @param mom:
        Force matrix
    @param dt:
        Integrating step size
    @param x:
        Array of solution vectors
    @param coeff:
        Array of residues for each contribution (multiplied by stepsize)
    @param kappa2:
        -kappa*kappa parameter
    @param ck:
        -clover_coefficient * kappa / 8
    @param nvec:
        Number of vectors
    @param multiplicity:
        Number fermions this bilinear reresents
    @param gauge_param:
        Gauge field meta data
    @param inv_param:
        Dirac and solver meta data
    """
    ...

def computeHISQForceQuda(
    momentum: NDArray[_fields],
    dt: double,
    level2_coeff: NDArray[float64],
    fat7_coeff: NDArray[float64],
    w_link: NDArray[_fields],
    v_link: NDArray[_fields],
    u_link: NDArray[_fields],
    quark: NDArray[_fields],
    num: int,
    num_naik: int,
    coeff: NDArray[float64],
    param: QudaGaugeParam,
):
    """
    Compute the fermion force for the HISQ quark action and integrate the momentum.

    @param momentum:
        The momentum field we are integrating
    @param dt:
        The stepsize used to integrate the momentum
    @param level2_coeff:
        The coefficients for the second level of smearing in the quark action.
    @param fat7_coeff:
        The coefficients for the first level of smearing (fat7) in the quark action.
    @param w_link:
        Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
    @param v_link:
        Fat7 link variables.
    @param u_link:
        SU(3) think link variables.
    @param quark:
        The input fermion field.
    @param num:
        The number of quark fields
    @param num_naik:
        The number of naik contributions
    @param coeff:
        The coefficient multiplying the fermion fields in the outer product
    @param param:
        The field parameters.
    """
    ...

def gaussGaugeQuda(seed: int, sigma: double) -> None:
    """
    Generate Gaussian distributed fields and store in the
    resident gauge field.  We create a Gaussian-distributed su(n)
    field and exponentiate it, e.g., U = exp(sigma * H), where H is
    the distributed su(n) field and sigma is the width of the
    distribution (sigma = 0 results in a free field, and sigma = 1 has
    maximum disorder).

    @param seed:
        The seed used for the RNG
    @param sigma:
        Width of Gaussian distrubution
    """
    ...

def gaussMomQuda(seed: int, sigma: double) -> None:
    """
    Generate Gaussian distributed fields and store in the
    resident momentum field. We create a Gaussian-distributed su(n)
    field, e.g., sigma * H, where H is the distributed su(n) field
    and sigma is the width of the distribution (sigma = 0 results
    in a free field, and sigma = 1 has maximum disorder).

    @param seed:
        The seed used for the RNG
    @param sigma:
        Width of Gaussian distrubution
    """
    ...

def plaqQuda() -> List[double, 3]:
    """
    Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.

    @return plaq:
        Array for storing the averages (total, spatial, temporal)
    """
    ...

def polyakovLoopQuda(dir: int) -> List[double, 2]:
    """
    Computes the trace of the Polyakov loop of the current resident field
    in a given direction.

    @param[in] dir:
        Direction of Polyakov loop
    @return ploop:
        Trace of the Polyakov loop in direction dir
    """
    ...

def performWuppertalnStep(
    h_out: NDArray[_field], h_in: NDArray[_field], param: QudaInvertParam, n_steps: int, alpha: double
) -> None:
    """
    Performs Wuppertal smearing on a given spinor using the gauge field
    gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.

    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage and operator which will be applied to the spinor
    @param n_steps:
        Number of steps to apply.
    @param alpha:
        Alpha coefficient for Wuppertal smearing.
    """
    ...

def performGaugeSmearQuda(smear_param: QudaGaugeSmearParam, obs_param: QudaGaugeObservableParam) -> None:
    """
    Performs APE, Stout, or Over Imroved STOUT smearing on gaugePrecise and stores it in gaugeSmeared

    @param[in] smear_param:
        Parameter struct that defines the computation parameters
    @param[in,out] obs_param:
        Parameter struct that defines which
        observables we are making and the resulting observables.
    """
    ...

def performWFlowQuda(smear_param: QudaGaugeSmearParam, obs_param: QudaGaugeObservableParam) -> None:
    """
    Performs Wilson Flow on gaugePrecise and stores it in gaugeSmeared

    @param[in] smear_param:
        Parameter struct that defines the computation parameters
    @param[in,out] obs_param:
        Parameter struct that defines which
        observables we are making and the resulting observables.
    """
    ...

def gaugeObservablesQuda(param: QudaGaugeObservableParam) -> None:
    """
    Calculates a variety of gauge-field observables.  If a
    smeared gauge field is presently loaded (in gaugeSmeared) the
    observables are computed on this, else the resident gauge field
    will be used.

    @param[in,out] param:
        Parameter struct that defines which
        observables we are making and the resulting observables.
    """
    ...

def contractQuda(
    x: NDArray[_field],
    y: NDArray[_field],
    result: NDArray[_field],
    cType: QudaContractType,
    param: QudaInvertParam,
    X: NDArray[int32],
) -> None:
    """
    Public function to perform color contractions of the host spinors x and y.

    @param[in] x:
        pointer to host data
    @param[in] y:
        pointer to host data
    @param[out] result:
        pointer to the 16 spin projections per lattice site
    @param[in] cType:
        Which type of contraction (open, degrand-rossi, etc)
    @param[in] param:
        meta data for construction of ColorSpinorFields.
    @param[in] X:
        spacetime data for construction of ColorSpinorFields.
    """
    ...

def computeGaugeFixingOVRQuda(
    gauge: NDArray[_fields],
    gauge_dir: int,
    Nsteps: int,
    verbose_interval: int,
    relax_boost: double,
    tolerance: double,
    reunit_interval: int,
    stopWtheta: int,
    param: QudaGaugeParam,
) -> int:
    """
    Gauge fixing with overrelaxation with support for single and multi GPU.

    @param[in,out] gauge:
        gauge field to be fixed
    @param[in] gauge_dir:
        3 for Coulomb gauge fixing, other for Landau gauge fixing
    @param[in] Nsteps:
        maximum number of steps to perform gauge fixing
    @param[in] verbose_interval:
        print gauge fixing info when iteration count is a multiple of this
    @param[in] relax_boost:
        gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
    @param[in] tolerance:
        torelance value to stop the method, if this value is zero then the method stops when
        iteration reachs the maximum number of steps defined by Nsteps
    @param[in] reunit_interval:
        reunitarize gauge field when iteration count is a multiple of this
    @param[in] stopWtheta:
        0 for MILC criterion and 1 to use the theta value
    @param[in] param:
        The parameters of the external fields and the computation settings
    """
    ...

def computeGaugeFixingFFTQuda(
    gauge: NDArray[_fields],
    gauge_dir: int,
    Nsteps: int,
    verbose_interval: int,
    alpha: double,
    autotune: int,
    tolerance: double,
    stopWtheta: int,
    param: QudaGaugeParam,
) -> int:
    """
    Gauge fixing with Steepest descent method with FFTs with support for single GPU only.

    @param[in,out] gauge:
        gauge field to be fixed
    @param[in] gauge_dir:
        3 for Coulomb gauge fixing, other for Landau gauge fixing
    @param[in] Nsteps:
        maximum number of steps to perform gauge fixing
    @param[in] verbose_interval:
        print gauge fixing info when iteration count is a multiple of this
    @param[in] alpha:
        gauge fixing parameter of the method, most common value is 0.08
    @param[in] autotune:
        1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
    @param[in] tolerance:
        torelance value to stop the method, if this value is zero then the method stops when
        iteration reachs the maximum number of steps defined by Nsteps
    @param[in] stopWtheta:
        0 for MILC criterion and 1 to use the theta value
    @param[in] param:
        The parameters of the external fields and the computation settings
    """
    ...

def blasGEMMQuda(
    arrayA: NDArray[_field], arrayB: NDArray[_field], arrayC: NDArray[_field], native: QudaBoolean, param: QudaBLASParam
) -> None:
    """
    Strided Batched GEMM

    @param[in] arrayA:
        The array containing the A matrix data
    @param[in] arrayB:
        The array containing the B matrix data
    @param[in] arrayC:
        The array containing the C matrix data
    @param[in] native:
        Boolean to use either the native or generic version
    @param[in] param:
        The data defining the problem execution.
    """
    ...

def blasLUInvQuda(Ainv: NDArray[_field], A: NDArray[_field], use_native: QudaBoolean, param: QudaBLASParam) -> None:
    """
    Strided Batched in-place matrix inversion via LU

    @param[in] Ainv:
        The array containing the A inverse matrix data
    @param[in] A:
        The array containing the A matrix data
    @param[in] use_native:
        Boolean to use either the native or generic version
    @param[in] param:
        The data defining the problem execution.
    """
    ...

def flushChronoQuda(index: int) -> None:
    """
    Flush the chronological history for the given index

    @param[in] index:
        Index for which we are flushing
    """
    ...

def newDeflationQuda(param: QudaEigParam) -> Pointer:
    """
    Create deflation solver resources.
    """
    ...

def destroyDeflationQuda(df_instance: Pointer) -> None:
    """
    Free resources allocated by the deflated solver
    """
    ...

class QudaQuarkSmearParam:
    """
    Parameter set for quark smearing operations
    """

    def __init__(self) -> None: ...
    # def __repr__(self) -> str: ...

    inv_param: QudaInvertParam
    """Used to store information pertinent to the operator"""

    n_steps: int
    """Number of steps to apply"""
    width: double
    """The width of the Gaussian"""
    compute_2link: int
    """if nonzero then compute two-link, otherwise reuse gaugeSmeared"""
    delete_2link: int
    """if nonzero then delete two-link, otherwise keep two-link for future use"""
    t0: int
    """Set if the input spinor is on a time slice"""
    secs: double
    """Time taken for the smearing operations"""
    gflops: double
    """Flops count for the smearing operations"""
    energy: double
    """The energy consumed by the smearing operations"""
    power: double
    """The mean power of the smearing operations"""
    temp: double
    """The mean temperature of the device for the duration of the smearing operations"""
    clock: double
    """The mean clock frequency of the device for the duration of the smearing operations"""

def performTwoLinkGaussianSmearNStep(h_in: NDArray[_field], smear_param: QudaQuarkSmearParam) -> None:
    """
    Performs two-link Gaussian smearing on a given spinor (for staggered fermions).

    @param[in,out] h_in:
        Input spinor field to smear
    @param[in] smear_param:
        Contains all metadata the operator which will be applied to the spinor
    """
    ...
