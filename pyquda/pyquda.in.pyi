from typing import List

size_t = int
double = float
double_complex = complex

from .enum_quda import (  # noqa: F401
    QUDA_INVALID_ENUM,
    QUDA_VERSION_MAJOR,
    QUDA_VERSION_MINOR,
    QUDA_VERSION_SUBMINOR,
    QUDA_VERSION,
    QUDA_MAX_DIM,
    QUDA_MAX_GEOMETRY,
    QUDA_MAX_MULTI_SHIFT,
    QUDA_MAX_BLOCK_SRC,
    QUDA_MAX_ARRAY_SIZE,
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
    QUDA_BOOLEAN_NO,
    QUDA_BOOLEAN_YES,
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

from .pointer import Pointer, Pointers

class QudaGaugeParam:
    """
    Parameters having to do with the gauge field or the
    interpretation of the gauge field by various Dirac operators
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

##%%!! QudaGaugeParam

class QudaInvertParam:
    """
    Parameters relating to the solver and the choice of Dirac operator.
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

##%%!! QudaInvertParam

class QudaEigParam:
    """
    Parameter set for solving eigenvalue problems.
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

##%%!! QudaEigParam

class QudaMultigridParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

##%%!! QudaMultigridParam

class QudaGaugeObservableParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

##%%!! QudaGaugeObservableParam

class QudaGaugeSmearParam:
    def __init__(self) -> None: ...
    # def __repr__(self) -> str: ...

##%%!! QudaGaugeSmearParam

class QudaBLASParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

##%%!! QudaBLASParam

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

def initCommsGridQuda(nDim: int, dims: List[int, 4]) -> None:
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

def loadGaugeQuda(h_gauge: Pointers, param: QudaGaugeParam) -> None:
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

def saveGaugeQuda(h_gauge: Pointers, param: QudaGaugeParam) -> None:
    """
    Save the gauge field to the host.

    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param param:
        Contains all metadata regarding host and device storage
    """
    ...

def loadCloverQuda(h_clover: Pointer, h_clovinv: Pointer, inv_param: QudaInvertParam) -> None:
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

def invertQuda(h_x: Pointer, h_b: Pointer, param: QudaInvertParam) -> None:
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
    ...

def invertMultiShiftQuda(_hp_x: Pointers, _hp_b: Pointer, param: QudaInvertParam) -> None:
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

def dslashQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam, parity: QudaParity) -> None:
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

def cloverQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam, parity: QudaParity, inverse: int) -> None:
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

def MatQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam) -> None:
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

def MatDagMatQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam) -> None:
    r"""
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
    fatlink: Pointers, longlink: Pointers, ulink: Pointers, inlink: Pointers, path_coeff: Pointer, param: QudaGaugeParam
) -> None:
    """ """
    ...

def computeTwoLinkQuda(twolink: Pointers, inlink: Pointers, param: QudaGaugeParam) -> None:
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

def momResidentQuda(mom: Pointer, param: QudaGaugeParam) -> None:
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
    mom: Pointers,
    sitelink: Pointers,
    input_path_buf: Pointer,
    path_length: Pointer,
    loop_coeff: Pointer,
    num_paths: int,
    max_length: int,
    dt: double,
    qudaGaugeParam: QudaGaugeParam,
) -> None:
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
    out: Pointers,
    sitelink: Pointers,
    input_path_buf: Pointer,
    path_length: Pointer,
    loop_coeff: Pointer,
    num_paths: int,
    max_length: int,
    dt: double,
    qudaGaugeParam: QudaGaugeParam,
) -> None:
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
    traces: Pointer,
    input_path_buf: Pointers,
    path_length: Pointer,
    loop_coeff: Pointer,
    num_paths: int,
    max_length: int,
    factor: double,
) -> None:
    """
    Compute the traces of products of gauge links along paths using the resident field

    @param[in,out] traces:
        The computed traces
    @param[in] sitelink:
        The gauge field from which we compute the products of gauge links
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
    gauge: Pointers, momentum: Pointers, dt: double, conj_mom: int, exact: int, param: QudaGaugeParam
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

def staggeredPhaseQuda(gauge_h: Pointers, param: QudaGaugeParam) -> None:
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

def projectSU3Quda(gauge_h: Pointers, tol: double, param: QudaGaugeParam) -> None:
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

def momActionQuda(momentum: Pointer, param: QudaGaugeParam) -> double:
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
    mom: Pointers,
    dt: double,
    x: Pointers,
    coeff: Pointer,
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

def performWuppertalnStep(h_out: Pointer, h_in: Pointer, param: QudaInvertParam, n_steps: int, alpha: double) -> None:
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
    x: Pointer, y: Pointer, result: Pointer, cType: QudaContractType, param: QudaInvertParam, X: Pointer
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
    gauge: Pointers,
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
    gauge: Pointers,
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

def blasGEMMQuda(arrayA: Pointer, arrayB: Pointer, arrayC: Pointer, native: QudaBoolean, param: QudaBLASParam) -> None:
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

def blasLUInvQuda(Ainv: Pointer, A: Pointer, use_native: QudaBoolean, param: QudaBLASParam) -> None:
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

##%%!! QudaQuarkSmearParam

def performTwoLinkGaussianSmearNStep(h_in: Pointer, smear_param: QudaQuarkSmearParam) -> None:
    """
    Performs two-link Gaussian smearing on a given spinor (for staggered fermions).

    @param[in,out] h_in:
        Input spinor field to smear
    @param[in] smear_param:
        Contains all metadata the operator which will be applied to the spinor
    """
    ...
