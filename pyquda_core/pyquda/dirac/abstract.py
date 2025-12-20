from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Dict, Tuple, Optional

from pyquda_comm import getLogger
from pyquda_comm.field import (
    LatticeInfo,
    LatticeGauge,
    LatticeFermion,
    MultiLatticeFermion,
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
)
from ..pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaEigParam,
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    invertQuda,
    invertMultiSrcQuda,
    MatQuda,
    MatDagMatQuda,
    dslashQuda,
    dslashMultiSrcQuda,
    cloverQuda,
)
from ..enum_quda import (
    QUDA_MAX_MG_LEVEL,
    QudaParity,
    QudaPrecision,
    QudaReconstructType,
    QudaSolutionType,
    QudaMatPCType,
    # QudaSolverNormalization,
    QudaVerbosity,
)

from .general import (
    Precision,
    Reconstruct,
    Multigrid,
    getGlobalPrecision,
    getGlobalReconstruct,
    setPrecisionParam,
    setReconstructParam,
)


class DiracGaugeStack:
    loaded: List[bool]
    stack: List[Tuple["Dirac", LatticeGauge, Dict[str, Any]]]

    def __init__(self):
        self.loaded = []
        self.stack = []

    def load(self, dirac: "Dirac", gauge: LatticeGauge, kwargs: Dict[str, Any]):
        if len(self.stack) > 0 and self.stack[-1][0] is dirac and self.stack[-1][1] is gauge:
            self.loaded.append(False)
        else:
            self.loaded.append(True)
            self.stack.append((dirac, gauge, kwargs))
            dirac.loadGauge(gauge, **kwargs)

    def free(self):
        if self.loaded.pop():
            dirac, gauge, kwargs = self.stack.pop()
            dirac.freeGauge()
            if len(self.loaded) > 0 and self.loaded[-1]:
                dirac, gauge, kwargs = self.stack[-1]
                dirac.loadGauge(gauge, **kwargs)


_DIRAC_GAUGE_STACK = DiracGaugeStack()


class Dirac(ABC):
    latt_info: LatticeInfo
    precision: Precision
    reconstruct: Reconstruct
    gauge_param: QudaGaugeParam
    invert_param: QudaInvertParam
    eig_param: QudaEigParam
    smear_param: QudaGaugeSmearParam
    obs_param: QudaGaugeObservableParam

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.precision = getGlobalPrecision("none")
        self.reconstruct = getGlobalReconstruct("none")

    @abstractmethod
    def loadGauge(self, gauge: LatticeGauge):
        raise NotImplementedError()

    @abstractmethod
    def freeGauge(self):
        raise NotImplementedError()

    @contextmanager
    def useGauge(self, gauge: LatticeGauge):
        try:
            _DIRAC_GAUGE_STACK.load(self, gauge, {})
            yield self
        finally:
            _DIRAC_GAUGE_STACK.free()

    def setPrecision(
        self,
        *,
        cuda: Optional[QudaPrecision] = None,
        sloppy: Optional[QudaPrecision] = None,
        precondition: Optional[QudaPrecision] = None,
        eigensolver: Optional[QudaPrecision] = None,
    ):
        if cuda is not None or sloppy is not None or precondition is not None or eigensolver is not None:
            self.precision = Precision(
                self.precision.cpu,
                cuda if cuda is not None else self.precision.cuda,
                sloppy if sloppy is not None else self.precision.sloppy,
                precondition if precondition is not None else self.precision.precondition,
                eigensolver if eigensolver is not None else self.precision.eigensolver,
            )
        setPrecisionParam(self.precision, self.gauge_param, self.invert_param, None, None)

    def setReconstruct(
        self,
        *,
        cuda: Optional[QudaReconstructType] = None,
        sloppy: Optional[QudaReconstructType] = None,
        precondition: Optional[QudaReconstructType] = None,
        eigensolver: Optional[QudaReconstructType] = None,
    ):
        if cuda is not None or sloppy is not None or precondition is not None or eigensolver is not None:
            self.reconstruct = Reconstruct(
                cuda if cuda is not None else self.reconstruct.cuda,
                sloppy if sloppy is not None else self.reconstruct.sloppy,
                precondition if precondition is not None else self.reconstruct.precondition,
                eigensolver if eigensolver is not None else self.reconstruct.eigensolver,
            )
        setReconstructParam(self.reconstruct, self.gauge_param)

    def setVerbosity(self, verbosity: QudaVerbosity):
        self.invert_param.verbosity = verbosity
        self.invert_param.verbosity_precondition = verbosity


class FermionDirac(Dirac):
    multigrid: Multigrid

    def __init__(self, latt_info: LatticeInfo) -> None:
        super().__init__(latt_info)
        self.reconstruct = getGlobalReconstruct("wilson")

    @abstractmethod
    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def freeGauge(self):
        raise NotImplementedError()

    @contextmanager
    def useGauge(self, gauge: LatticeGauge):
        try:
            _DIRAC_GAUGE_STACK.load(self, gauge, {"thin_update_only": True})
            yield self
        finally:
            _DIRAC_GAUGE_STACK.free()

    def setPrecision(
        self,
        *,
        cuda: Optional[QudaPrecision] = None,
        sloppy: Optional[QudaPrecision] = None,
        precondition: Optional[QudaPrecision] = None,
        eigensolver: Optional[QudaPrecision] = None,
    ):
        if self.multigrid.param is not None and self.multigrid.inv_param is not None:
            self.precision = getGlobalPrecision("multigrid")
        else:
            self.precision = getGlobalPrecision("invert")
        super().setPrecision(cuda=cuda, sloppy=sloppy, precondition=precondition, eigensolver=eigensolver)
        setPrecisionParam(self.precision, None, None, self.multigrid.param, self.multigrid.inv_param)

    def setVerbosity(self, verbosity: QudaVerbosity):
        super().setVerbosity(verbosity)
        if self.multigrid.param is not None and self.multigrid.inv_param is not None:
            self.multigrid.param.verbosity = [verbosity] * QUDA_MAX_MG_LEVEL
            self.multigrid.inv_param.verbosity = verbosity

    def performance(self):
        gflops, secs = self.invert_param.gflops, self.invert_param.secs
        if self.invert_param.verbosity >= QudaVerbosity.QUDA_SUMMARIZE:
            getLogger().info(f"Time = {secs:.3f} secs, Performance = {gflops / secs:.3f} GFLOPS")

    def invert(self, b: LatticeFermion):
        x = LatticeFermion(b.latt_info)
        invertQuda(x.data_ptr, b.data_ptr, self.invert_param)
        self.performance()
        return x

    def invertRestart(self, b: LatticeFermion, restart: int):
        x = self.invert(b)
        # self.invert_param.solver_normalization = QudaSolverNormalization.QUDA_SOURCE_NORMALIZATION
        for _ in range(restart):
            r = b - self.mat(x)
            norm = r.norm2() ** 0.5
            r /= norm
            r = self.invertRestart(r, restart - 1)
            r *= norm
            x += r
        # self.invert_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
        return x

    def mat(self, x: LatticeFermion):
        b = LatticeFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def matDagMat(self, x: LatticeFermion):
        b = LatticeFermion(x.latt_info)
        MatDagMatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def dslash(self, x: LatticeFermion, parity: QudaParity = QudaParity.QUDA_INVALID_PARITY):
        b = LatticeFermion(x.latt_info)
        if parity == QudaParity.QUDA_EVEN_PARITY:
            dslashQuda(b.even_ptr, x.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            dslashQuda(b.odd_ptr, x.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        elif parity == QudaParity.QUDA_INVALID_PARITY:
            dslashQuda(b.even_ptr, x.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
            dslashQuda(b.odd_ptr, x.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        return b

    def invertMultiSrc(self, b: MultiLatticeFermion):
        self.invert_param.num_src = b.L5
        x = MultiLatticeFermion(b.latt_info, b.L5)
        invertMultiSrcQuda(x.data_ptrs, b.data_ptrs, self.invert_param)
        self.performance()
        return x

    def invertMultiSrcRestart(self, b: MultiLatticeFermion, restart: int):
        x = self.invertMultiSrc(b)
        for _ in range(restart):
            r = MultiLatticeFermion(b.latt_info, b.L5)
            norm = []
            for i in range(b.L5):
                r[i] = b[i] - self.mat(x[i])
                norm.append(r[i].norm2() ** 0.5)
                r[i] /= norm[i]
            r = self.invertMultiSrcRestart(r, restart - 1)
            for i in range(b.L5):
                r[i] *= norm[i]
                x[i] += r[i]
        return x

    def dslashMultiSrc(self, x: MultiLatticeFermion, parity: QudaParity = QudaParity.QUDA_INVALID_PARITY):
        self.invert_param.num_src = x.L5
        b = MultiLatticeFermion(x.latt_info, x.L5)
        if parity == QudaParity.QUDA_EVEN_PARITY:
            dslashMultiSrcQuda(b.even_ptrs, x.odd_ptrs, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            dslashMultiSrcQuda(b.odd_ptrs, x.even_ptrs, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        elif parity == QudaParity.QUDA_INVALID_PARITY:
            dslashMultiSrcQuda(b.even_ptrs, x.odd_ptrs, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
            dslashMultiSrcQuda(b.odd_ptrs, x.even_ptrs, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        return b

    def invertPC(self, b: LatticeFermion):
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD

        kappa = self.invert_param.kappa
        x = LatticeFermion(b.latt_info)
        dslashQuda(x.odd_ptr, b.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        x.even = b.odd + kappa * x.odd
        # * QUDA_ASYMMETRIC_MASS_NORMALIZATION makes the even part 1 / (2 * kappa) instead of 1
        invertQuda(x.odd_ptr, x.even_ptr, self.invert_param)
        self.performance()
        dslashQuda(x.even_ptr, x.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
        x.even = kappa * (2 * b.even + x.even)

        self.invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        return x

    def invertCloverPC(self, b: LatticeFermion):
        a = LatticeFermion(b.latt_info)
        cloverQuda(a.even_ptr, b.even_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY, 1)
        cloverQuda(a.odd_ptr, b.odd_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY, 1)
        return self.invertPC(a)


class StaggeredFermionDirac(FermionDirac):
    def __init__(self, latt_info: LatticeInfo) -> None:
        super().__init__(latt_info)
        self.reconstruct = getGlobalReconstruct("none")

    def invert(self, b: LatticeStaggeredFermion):
        x = LatticeStaggeredFermion(b.latt_info)
        invertQuda(x.data_ptr, b.data_ptr, self.invert_param)
        self.performance()
        return x

    def invertRestart(self, b: LatticeStaggeredFermion, restart: int):
        x = self.invert(b)
        for _ in range(restart):
            r = b - self.mat(x)
            norm = r.norm2() ** 0.5
            r /= norm
            r = self.invertRestart(r, restart - 1)
            r *= norm
            x += r
        return x

    def mat(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def matDagMat(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatDagMatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def dslash(self, x: LatticeStaggeredFermion, parity: QudaParity = QudaParity.QUDA_INVALID_PARITY):
        b = LatticeStaggeredFermion(x.latt_info)
        if parity == QudaParity.QUDA_EVEN_PARITY:
            dslashQuda(b.even_ptr, x.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            dslashQuda(b.odd_ptr, x.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        elif parity == QudaParity.QUDA_INVALID_PARITY:
            dslashQuda(b.even_ptr, x.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
            dslashQuda(b.odd_ptr, x.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        return b

    def invertMultiSrc(self, b: MultiLatticeStaggeredFermion):
        self.invert_param.num_src = b.L5
        x = MultiLatticeStaggeredFermion(b.latt_info, b.L5)
        invertMultiSrcQuda(x.data_ptrs, b.data_ptrs, self.invert_param)
        self.performance()
        return x

    def invertMultiSrcRestart(self, b: MultiLatticeStaggeredFermion, restart: int):
        x = self.invertMultiSrc(b)
        for _ in range(restart):
            r = MultiLatticeStaggeredFermion(b.latt_info, b.L5)
            norm = []
            for i in range(b.L5):
                r[i] = b[i] - self.mat(x[i])
                norm.append(r[i].norm2() ** 0.5)
                r[i] /= norm[i]
            r = self.invertMultiSrcRestart(r, restart - 1)
            for i in range(b.L5):
                r[i] *= norm[i]
                x[i] += r[i]
        return x

    def dslashMultiSrc(self, x: MultiLatticeStaggeredFermion, parity: QudaParity = QudaParity.QUDA_INVALID_PARITY):
        self.invert_param.num_src = x.L5
        b = MultiLatticeStaggeredFermion(x.latt_info, x.L5)
        if parity == QudaParity.QUDA_EVEN_PARITY:
            dslashMultiSrcQuda(b.even_ptrs, x.odd_ptrs, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            dslashMultiSrcQuda(b.odd_ptrs, x.even_ptrs, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        elif parity == QudaParity.QUDA_INVALID_PARITY:
            dslashMultiSrcQuda(b.even_ptrs, x.odd_ptrs, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
            dslashMultiSrcQuda(b.odd_ptrs, x.even_ptrs, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        return b

    def invertPC(self, b: LatticeStaggeredFermion):
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD

        mass = self.invert_param.mass
        x = LatticeStaggeredFermion(b.latt_info)
        dslashQuda(x.odd_ptr, b.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        x.even = (2 * mass) * b.odd + x.odd
        invertQuda(x.odd_ptr, x.even_ptr, self.invert_param)
        self.performance()
        dslashQuda(x.even_ptr, x.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
        x.even = (0.5 / mass) * (b.even + x.even)

        self.invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        return x
