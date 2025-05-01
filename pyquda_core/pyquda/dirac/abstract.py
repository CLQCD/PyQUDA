from abc import ABC, abstractmethod

from pyquda_comm import getLogger
from pyquda_comm.pointer import Pointer
from ..field import (
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
    QudaMultigridParam,
    QudaEigParam,
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    invertQuda,
    invertMultiSrcQuda,
    MatQuda,
    MatDagMatQuda,
    dslashQuda,
    dslashMultiSrcQuda,
    newMultigridQuda,
    updateMultigridQuda,
    destroyMultigridQuda,
)
from ..enum_quda import (
    QUDA_MAX_MG_LEVEL,
    QudaBoolean,
    QudaParity,
    QudaPrecision,
    QudaReconstructType,
    # QudaSolverNormalization,
    QudaVerbosity,
)

from .general import (
    Precision,
    Reconstruct,
    getGlobalPrecision,
    getGlobalReconstruct,
    setPrecisionParam,
    setReconstructParam,
)


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
        self.precision = getGlobalPrecision()
        self.reconstruct = getGlobalReconstruct()

    def setPrecision(
        self,
        *,
        cuda: QudaPrecision = None,
        sloppy: QudaPrecision = None,
        precondition: QudaPrecision = None,
        eigensolver: QudaPrecision = None,
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
        cuda: QudaReconstructType = None,
        sloppy: QudaReconstructType = None,
        precondition: QudaReconstructType = None,
        eigensolver: QudaReconstructType = None,
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


class Multigrid:
    param: QudaMultigridParam
    inv_param: QudaInvertParam
    instance: Pointer

    def __init__(self, param: QudaMultigridParam, inv_param: QudaInvertParam) -> None:
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
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = newMultigridQuda(self.param)

    def update(self, thin_update_only: bool):
        if self.instance is not None:
            if thin_update_only:
                self.param.thin_update_only = QudaBoolean.QUDA_BOOLEAN_TRUE
                updateMultigridQuda(self.instance, self.param)
                self.param.thin_update_only = QudaBoolean.QUDA_BOOLEAN_FALSE
            else:
                updateMultigridQuda(self.instance, self.param)

    def destroy(self):
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = None


class FermionDirac(Dirac):
    multigrid: Multigrid

    def __init__(self, latt_info: LatticeInfo) -> None:
        super().__init__(latt_info)

    def setPrecision(
        self,
        *,
        cuda: QudaPrecision = None,
        sloppy: QudaPrecision = None,
        precondition: QudaPrecision = None,
        eigensolver: QudaPrecision = None,
    ):
        super().setPrecision(cuda=cuda, sloppy=sloppy, precondition=precondition, eigensolver=eigensolver)
        setPrecisionParam(self.precision, None, None, self.multigrid.param, self.multigrid.inv_param)

    def setVerbosity(self, verbosity: QudaVerbosity):
        super().setVerbosity(verbosity)
        if self.multigrid.param is not None:
            self.multigrid.inv_param.verbosity = verbosity
            self.multigrid.param.verbosity = [verbosity] * QUDA_MAX_MG_LEVEL

    @abstractmethod
    def loadGauge(self, gauge: LatticeGauge):
        pass

    @abstractmethod
    def destroy(self):
        pass

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

    def dslash(self, x: LatticeFermion, parity: QudaParity):
        b = LatticeFermion(x.latt_info)
        dslashQuda(b.data_ptr, x.data_ptr, self.invert_param, parity)
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

    def dslashMultiSrc(self, x: MultiLatticeFermion, parity: QudaParity):
        self.invert_param.num_src = x.L5
        b = MultiLatticeFermion(x.latt_info, x.L5)
        dslashMultiSrcQuda(b.data_ptrs, x.data_ptrs, self.invert_param, parity)
        return b

    def newMultigrid(self):
        if self.multigrid.param is not None:
            self.multigrid.new()
            self.invert_param.preconditioner = self.multigrid.instance

    def updateMultigrid(self, thin_update_only: bool):
        if self.multigrid.param is not None:
            self.multigrid.update(thin_update_only)
            self.invert_param.preconditioner = self.multigrid.instance

    def destroyMultigrid(self):
        if self.multigrid.param is not None:
            self.multigrid.destroy()


class StaggeredFermionDirac(FermionDirac):
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

    def dslash(self, x: LatticeStaggeredFermion, parity: QudaParity):
        b = LatticeStaggeredFermion(x.latt_info)
        dslashQuda(b.data_ptr, x.data_ptr, self.invert_param, parity)
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

    def dslashMultiSrc(self, x: MultiLatticeStaggeredFermion, parity: QudaParity):
        self.invert_param.num_src = x.L5
        b = MultiLatticeStaggeredFermion(x.latt_info, x.L5)
        dslashMultiSrcQuda(b.data_ptrs, x.data_ptrs, self.invert_param, parity)
        return b
