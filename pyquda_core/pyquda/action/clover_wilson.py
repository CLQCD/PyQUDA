import numpy

from .. import getLogger
from ..pointer import Pointers
from ..pyquda import computeCloverForceQuda, loadCloverQuda, loadGaugeQuda
from ..enum_quda import (
    QudaDagType,
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaSolutionType,
    QudaSolveType,
    QudaVerbosity,
)
from ..field import LatticeInfo, LatticeFermion, MultiLatticeFermion
from ..dirac import CloverWilsonDirac

nullptr = Pointers("void", 0)

from .abstract import RationalParam, FermionAction


class CloverWilsonAction(FermionAction):
    dirac: CloverWilsonDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        rational_param: RationalParam,
        mass: float,
        n_flavor: int,
        tol: float,
        maxiter: int,
        clover_csw: float,
        verbosity: QudaVerbosity = QudaVerbosity.QUDA_SILENT,
    ) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, 1, None))

        kappa = 1 / (2 * (mass + latt_info.Nd))
        self.setForceParam(rational_param, kappa, clover_csw, n_flavor)
        self.quark = MultiLatticeFermion(self.latt_info, self.max_num_offset)
        self.phi = LatticeFermion(latt_info)
        self.eta = LatticeFermion(latt_info)

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE  # This is set to compute action
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        self.invert_param.verbosity = verbosity

    def setForceParam(self, rational_param: RationalParam, kappa: float, clover_csw: float, n_flavor: int):
        self.coeff = numpy.array(rational_param.residue_molecular_dynamics, "<f8")
        self.kappa2 = -(kappa**2)
        self.ck = -kappa * clover_csw / 8
        self.nvector = len(rational_param.offset_molecular_dynamics)
        self.multiplicity = n_flavor
        self.max_num_offset = max(
            len(rational_param.offset_molecular_dynamics),
            len(rational_param.offset_fermion_action),
            len(rational_param.offset_pseudo_fermion),
        )
        self.rational_param = rational_param

    def updateClover(self, new_gauge: bool):
        if new_gauge:
            loadGaugeQuda(nullptr, self.gauge_param)
            loadCloverQuda(nullptr, nullptr, self.invert_param)

    def sample(self, new_gauge: bool):
        self.sampleEta()
        self.updateClover(new_gauge)
        self.invertMultiShift("pseudo_fermion")

    def action(self, new_gauge: bool) -> float:
        self.invert_param.compute_clover_trlog = 1
        self.updateClover(new_gauge)
        self.invert_param.compute_clover_trlog = 0
        self.invert_param.compute_action = 1
        self.invertMultiShift("molecular_dynamics")
        self.invert_param.compute_action = 0
        return (
            self.invert_param.action[0]
            - self.latt_info.volume // 2 * self.latt_info.Ns * self.latt_info.Nc  # volume_cb2 here
            - self.multiplicity * self.invert_param.trlogA[1]
        )

    def force(self, dt, new_gauge: bool):
        self.updateClover(new_gauge)
        self.invertMultiShift("molecular_dynamics")
        # Some conventions force the dagger to be YES here
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        computeCloverForceQuda(
            nullptr,
            dt,
            self.quark.even_ptrs,
            self.coeff,
            self.kappa2,
            self.ck,
            self.nvector,
            self.multiplicity,
            self.gauge_param,
            self.invert_param,
        )
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
