from enum import IntEnum

# quda_constants.py

QUDA_INVALID_ENUM = -0x7FFFFFFF - 1


class qudaError_t(IntEnum):
    QUDA_SUCCESS = 0
    QUDA_ERROR = 1
    QUDA_ERROR_UNINITIALIZED = 2


class QudaDslashType(IntEnum):
    """
    Note: make sure QudaDslashType has corresponding entries in
    tests/utils/misc.cpp
    """

    pass


class QudaEigSpectrumType(IntEnum):
    """
    S=smallest L=largest\n
    R=real M=modulus I=imaniary
    """

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


class QudaParity(IntEnum):
    """Type used for "parity" argument to dslashQuda()"""

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


class QudaGammaBasis(IntEnum):
    """
    gamj=((top 2 rows)(bottom 2 rows))  s1,s2,s3 are Pauli spin matrices, 1 is 2x2 identity

    Dirac-Pauli -> DeGrand-Rossi   T = i/sqrt(2)*((s2,-s2)(s2,s2))     field_DR = T * field_DP\n
    UKQCD -> DeGrand-Rossi         T = i/sqrt(2)*((-s2,-s2)(-s2,s2))   field_DR = T * field_UK\n
    Chiral -> DeGrand-Rossi        T = i*((0,-s2)(s2,0))               field_DR = T * field_chiral
    """

    pass


class QudaProjectionType(IntEnum):
    """used to select projection method for deflated solvers"""

    pass


class QudaPCType(IntEnum):
    """used to select checkerboard preconditioning method"""

    pass


class QudaExtLibType(IntEnum):
    """Allows to choose an appropriate external library"""

    pass
