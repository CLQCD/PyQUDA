from enum import IntEnum

QUDA_VERSION_MAJOR = 1
QUDA_VERSION_MINOR = 1
QUDA_VERSION_SUBMINOR = 0

QUDA_VERSION = (QUDA_VERSION_MAJOR << 16) | (QUDA_VERSION_MINOR << 8) | QUDA_VERSION_SUBMINOR
"""
This macro is deprecated.  Use QUDA_VERSION_MAJOR, etc., instead.
"""

QUDA_MAX_DIM = 6
"""
Maximum number of dimensions supported by QUDA.  In practice, no
routines make use of more than 5.
"""

QUDA_MAX_GEOMETRY = 8
"""
Maximum geometry supported by a field.  This essentially is
the maximum number of dimensions supported per lattice site.
"""

QUDA_MAX_MULTI_SHIFT = 32
"""
Maximum number of shifts supported by the multi-shift solver.
This number may be changed if need be.
"""

QUDA_MAX_MULTI_SRC = 128
"""
Maximum number of sources that can be supported by the multi-src solver
"""

QUDA_MAX_DWF_LS = 32
"""
Maximum length of the Ls dimension for domain-wall fermions
"""

QUDA_MAX_MG_LEVEL = 5
"""
Maximum number of multi-grid levels.  This number may be
increased if needed.
"""


QUDA_INVALID_ENUM = -0x7FFFFFFF - 1


class qudaError_t(IntEnum):
    QUDA_SUCCESS = 0
    QUDA_ERROR = 1
    QUDA_ERROR_UNINITIALIZED = 2


class QudaMemoryType(IntEnum):
    pass


#
# Types used in QudaGaugeParam
#


class QudaLinkType(IntEnum):
    pass


class QudaGaugeFieldOrder(IntEnum):
    pass


class QudaTboundary(IntEnum):
    pass


class QudaPrecision(IntEnum):
    pass


class QudaReconstructType(IntEnum):
    pass


class QudaGaugeFixed(IntEnum):
    pass


#
# Types used in QudaInvertParam
#


class QudaDslashType(IntEnum):
    """
    Note: make sure QudaDslashType has corresponding entries in
    tests/utils/misc.cpp
    """

    pass


class QudaInverterType(IntEnum):
    pass


class QudaEigType(IntEnum):
    pass


class QudaEigSpectrumType(IntEnum):
    """
    S=smallest L=largest\n
    R=real M=modulus I=imaniary
    """

    pass


class QudaSolutionType(IntEnum):
    pass


class QudaSolveType(IntEnum):
    pass


class QudaMultigridCycleType(IntEnum):
    pass


class QudaSchwarzType(IntEnum):
    pass


class QudaAcceleratorType(IntEnum):
    pass


class QudaResidualType(IntEnum):
    pass


class QudaCABasis(IntEnum):
    """Which basis to use for CA algorithms"""

    pass


class QudaMatPCType(IntEnum):
    """
    Whether the preconditioned matrix is (1-k^2 Deo Doe) or (1-k^2 Doe Deo)

    For the clover-improved Wilson Dirac operator, QUDA_MATPC_EVEN_EVEN
    defaults to the "symmetric" form, (1 - k^2 A_ee^-1 D_eo A_oo^-1 D_oe)
    and likewise for QUDA_MATPC_ODD_ODD.

    For the "asymmetric" form, (A_ee - k^2 D_eo A_oo^-1 D_oe), select
    QUDA_MATPC_EVEN_EVEN_ASYMMETRIC.
    """

    pass


class QudaDagType(IntEnum):
    pass


class QudaMassNormalization(IntEnum):
    pass


class QudaSolverNormalization(IntEnum):
    pass


class QudaPreserveSource(IntEnum):
    pass


class QudaDiracFieldOrder(IntEnum):
    pass


class QudaCloverFieldOrder(IntEnum):
    pass


class QudaVerbosity(IntEnum):
    pass


class QudaTune(IntEnum):
    pass


class QudaPreserveDirac(IntEnum):
    pass


class QudaParity(IntEnum):
    """Type used for "parity" argument to dslashQuda()"""

    pass


#
# Types used only internally
#


class QudaDiracType(IntEnum):
    pass


class QudaFieldLocation(IntEnum):
    """Where the field is stored"""

    pass


class QudaSiteSubset(IntEnum):
    """Which sites are included"""

    pass


class QudaSiteOrder(IntEnum):
    """Site ordering (always t-z-y-x, with rightmost varying fastest)"""

    pass


class QudaFieldOrder(IntEnum):
    """Degree of freedom ordering"""

    pass


class QudaFieldCreate(IntEnum):
    pass


class QudaGammaBasis(IntEnum):
    """
    gamj=((top 2 rows)(bottom 2 rows))  s1,s2,s3 are Pauli spin matrices, 1 is 2x2 identity

    Dirac-Pauli -> DeGrand-Rossi   T = i/sqrt(2)*((s2,-s2)(s2,s2))     field_DR = T * field_DP\n
    UKQCD -> DeGrand-Rossi         T = i/sqrt(2)*((-s2,-s2)(-s2,s2))   field_DR = T * field_UK\n
    Chiral -> DeGrand-Rossi        T = i*((0,-s2)(s2,0))               field_DR = T * field_chiral
    """

    pass


class QudaSourceType(IntEnum):
    pass


class QudaNoiseType(IntEnum):
    pass


class QudaDilutionType(IntEnum):
    pass


class QudaProjectionType(IntEnum):
    """used to select projection method for deflated solvers"""

    pass


class QudaPCType(IntEnum):
    """used to select checkerboard preconditioning method"""

    pass


class QudaTwistFlavorType(IntEnum):
    pass


class QudaTwistDslashType(IntEnum):
    pass


class QudaTwistCloverDslashType(IntEnum):
    pass


class QudaTwistGamma5Type(IntEnum):
    pass


class QudaUseInitGuess(IntEnum):
    pass


class QudaDeflatedGuess(IntEnum):
    pass


class QudaComputeNullVector(IntEnum):
    pass


class QudaSetupType(IntEnum):
    pass


class QudaTransferType(IntEnum):
    pass


class QudaBoolean(IntEnum):
    pass


class QudaBLASType(IntEnum):
    pass


class QudaBLASOperation(IntEnum):
    pass


class QudaBLASDataType(IntEnum):
    pass


class QudaBLASDataOrder(IntEnum):
    pass


class QudaDirection(IntEnum):
    pass


class QudaLinkDirection(IntEnum):
    pass


class QudaFieldGeometry(IntEnum):
    pass


class QudaGhostExchange(IntEnum):
    pass


class QudaStaggeredPhase(IntEnum):
    pass


class QudaSpinTasteGamma(IntEnum):
    pass


class QudaContractType(IntEnum):
    pass


class QudaFFTSymmType(IntEnum):
    pass


class QudaContractGamma(IntEnum):
    pass


class QudaGaugeSmearType(IntEnum):
    pass


class QudaWFlowType(IntEnum):
    pass


class QudaFermionSmearType(IntEnum):
    pass


class QudaExtLibType(IntEnum):
    """Allows to choose an appropriate external library"""

    pass
