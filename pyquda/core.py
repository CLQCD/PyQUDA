from typing import List, Union
from enum import IntEnum
from math import sqrt
import numpy as np
import cupy as cp

from .pyquda import (
    Pointer, getDataPointers, getDataPointer, getEvenPointer, getOddPointer, QudaGaugeParam, QudaInvertParam,
    loadCloverQuda, loadGaugeQuda, invertQuda, dslashQuda, cloverQuda
)
from .enum_quda import (  # noqa: F401
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


class LatticeConstant(IntEnum):
    Nc = 3
    Nd = 4
    Ns = 4


Nc = LatticeConstant.Nc
Nd = LatticeConstant.Nd
Ns = LatticeConstant.Ns


def newLatticeFieldData(latt_size: List[int], dtype: str) -> cp.ndarray:
    Lx, Ly, Lz, Lt = latt_size
    if dtype.capitalize() == "Gauge":
        return cp.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
    elif dtype.capitalize() == "Fermion":
        return cp.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
    elif dtype.capitalize() == "Propagator":
        return cp.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")


class LatticeField:
    def __init__(self) -> None:
        pass


class LatticeGauge(LatticeField):
    def __init__(self, latt_size: List[int], value=None) -> None:
        self.latt_size = latt_size
        if value is None:
            self.data = newLatticeFieldData(latt_size, "Gauge").reshape(-1)
        else:
            self.data = value.reshape(-1)

    def setAntiPeroidicT(self):
        Lt = self.latt_size[Nd - 1]
        data = self.data.reshape(Nd, 2, Lt, -1)
        data[Nd - 1, :, Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        data = self.data.reshape(Nd, -1)
        data[:Nd - 1] /= anisotropy

    @property
    def data_ptr(self):
        return getDataPointers(self.data.reshape(4, -1), 4)

    @property
    def data_ptrs(self):
        return getDataPointers(self.data.reshape(4, -1), 4)


class LatticeFermion(LatticeField):
    def __init__(self, latt_size: List[int]) -> None:
        self.latt_size = latt_size
        self.data = newLatticeFieldData(latt_size, "Fermion").reshape(-1)

    @property
    def even(self):
        return self.data.reshape(2, -1)[0]

    @even.setter
    def even(self, value):
        data = self.data.reshape(2, -1)
        data[0] = value.reshape(-1)

    @property
    def odd(self):
        return self.data.reshape(2, -1)[1]

    @odd.setter
    def odd(self, value):
        data = self.data.reshape(2, -1)
        data[1] = value.reshape(-1)

    @property
    def data_ptr(self):
        return getDataPointer(self.data)

    @property
    def even_ptr(self):
        return getEvenPointer(self.data.reshape(2, -1))

    @property
    def odd_ptr(self):
        return getOddPointer(self.data.reshape(2, -1))


class LatticePropagator(LatticeField):
    def __init__(self, latt_size: List[int]) -> None:
        self.latt_size = latt_size
        self.data = newLatticeFieldData(latt_size, "Propagator").reshape(-1)


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


def newQudaInvertParam(kappa: float, tol: float, maxiter: float, pc: bool):
    invert_param = QudaInvertParam()

    invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
    invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
    invert_param.kappa = kappa
    invert_param.tol = tol
    invert_param.maxiter = maxiter
    invert_param.reliable_delta = 0.001
    invert_param.pipeline = 0

    if pc:
        invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
    else:
        invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
        invert_param.solve_type = QudaSolveType.QUDA_NORMOP_SOLVE
    invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD

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


def _loadGauge(gauge: LatticeGauge, gauge_param: QudaGaugeParam):
    anisotropy = gauge_param.anisotropy

    gauge_data_bak = gauge.data.copy()
    if gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
        gauge.setAntiPeroidicT()
    if anisotropy != 1.0:
        gauge.setAnisotropy(anisotropy)
    loadGaugeQuda(gauge.data_ptrs, gauge_param)
    gauge.data = gauge_data_bak


def _loadClover(
    gauge: LatticeGauge, gauge_param: QudaGaugeParam, invert_param: QudaInvertParam, clover_coeff: float,
    clover_anisotropy: float
):
    invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH

    invert_param.clover_cpu_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
    invert_param.clover_cuda_prec = QudaPrecision.QUDA_DOUBLE_PRECISION
    invert_param.clover_cuda_prec_sloppy = QudaPrecision.QUDA_HALF_PRECISION
    invert_param.clover_cuda_prec_precondition = QudaPrecision.QUDA_HALF_PRECISION

    invert_param.clover_order = QudaCloverFieldOrder.QUDA_FLOAT2_CLOVER_ORDER
    invert_param.clover_coeff = clover_coeff
    invert_param.compute_clover = 1
    invert_param.compute_clover_inverse = 1

    anisotropy = gauge_param.anisotropy

    gauge_data_bak = gauge.data.copy()
    if clover_anisotropy != 1.0:
        gauge.setAnisotropy(clover_anisotropy)
    gauge_param.anisotropy = 1.0
    loadGaugeQuda(gauge.data_ptrs, gauge_param)
    loadCloverQuda(Pointer("void"), Pointer("void"), invert_param)
    gauge_param.anisotropy = anisotropy
    gauge.data = gauge_data_bak


def invert(b: LatticeFermion, invert_param: QudaInvertParam, pc: bool = None):
    kappa = invert_param.kappa
    x = LatticeFermion(b.latt_size)
    tmp = LatticeFermion(b.latt_size)

    if pc is None:
        solve_type = invert_param.solve_type
        solution_type = invert_param.solution_type
        solve_pc = (
            solve_type == QudaSolveType.QUDA_DIRECT_PC_SOLVE or
            solve_type == QudaSolveType.QUDA_NORMOP_PC_SOLVE or
            solve_type == QudaSolveType.QUDA_NORMERR_PC_SOLVE
        )  # yapf: disable
        solution_pc = (
            solution_type == QudaSolutionType.QUDA_MATPC_SOLUTION or
            solution_type == QudaSolutionType.QUDA_MATPC_DAG_SOLUTION or
            solution_type == QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION or
            solution_type == QudaSolutionType.QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION
        )  # yapf: disable
        assert solve_pc == solution_pc, "solution_type and solve_type must have the same pc setting"
        pc = solve_pc

    if pc:
        tmp2 = LatticeFermion(b.latt_size)

        if invert_param.dslash_type == QudaDslashType.QUDA_WILSON_DSLASH:
            tmp.even = b.even
            tmp.odd = b.odd
        elif invert_param.dslash_type == QudaDslashType.QUDA_CLOVER_WILSON_DSLASH:
            cloverQuda(tmp.even_ptr, b.even_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY, 1)
            cloverQuda(tmp.odd_ptr, b.odd_ptr, invert_param, QudaParity.QUDA_ODD_PARITY, 1)
        else:
            raise NotImplementedError(f"Dslash type {invert_param.dslash_type} is not implemented yet.")
        tmp.data *= 2 * kappa
        dslashQuda(tmp2.odd_ptr, tmp.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
        tmp.odd = tmp.odd + kappa * tmp2.odd
        invertQuda(x.odd_ptr, tmp.odd_ptr, invert_param)
        dslashQuda(tmp2.even_ptr, x.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
        x.even = tmp.even + kappa * tmp2.even
    else:
        invertQuda(x.data_ptr, b.data_ptr, invert_param)
        x.data *= 2 * kappa

    return x


def source(latt_size: List[int], source_type: str, t_srce: Union[int, List[int]], spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if source_type.lower() == "point":
        x, y, z, t = t_srce
        eo = (x + y + z + t) % 2
        data[eo, t, z, y, x // 2, spin, color] = 1
    elif source_type.lower() == "wall":
        t = t_srce
        data[:, t, :, :, :, spin, color] = 1
    else:
        raise NotImplementedError(f"{source_type} source is not implemented yet.")

    return b


class QudaFieldLoader:
    def __init__(
        self,
        latt_size,
        mass,
        tol,
        maxiter,
        xi_0: float = 1.0,
        nu: float = 1.0,
        clover_coeff_t: float = 0.0,
        clover_coeff_r: float = 1.0,
    ) -> None:

        Lx, Ly, Lz, Lt = latt_size
        volume = Lx * Ly * Lz * Lt
        xi = xi_0 / nu
        kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
        if xi != 1.0:
            clover_coeff = xi_0 * clover_coeff_t**2 / clover_coeff_r
            clover_xi = sqrt(xi_0 * clover_coeff_t / clover_coeff_r)
        else:
            clover_coeff = clover_coeff_t
            clover_xi = 1.0
        clover = clover_coeff != 0.0

        self.latt_size = latt_size
        self.volume = volume
        self.xi_0 = xi_0
        self.nu = nu
        self.xi = xi
        self.mass = mass
        self.kappa = kappa
        self.clover_coeff = kappa * clover_coeff
        self.clover_xi = clover_xi
        self.clover = clover
        self.gauge_param = newQudaGaugeParam(latt_size, xi)
        self.invert_param = newQudaInvertParam(kappa, tol, maxiter, True)

    def loadGauge(self, gauge: LatticeGauge):
        if self.clover:
            _loadClover(gauge, self.gauge_param, self.invert_param, self.clover_coeff, self.clover_xi)
        _loadGauge(gauge, self.gauge_param)

    def invert(self, b: LatticeFermion):
        return invert(b, self.invert_param, True)
