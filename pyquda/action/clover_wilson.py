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
from ..field import Nd, Nc, Ns, LatticeInfo, LatticeFermion
from ..dirac import CloverWilson

nullptr = Pointers("void", 0)

from . import rhmc_param
from .abstract import FermionAction


class CloverWilsonFermion(FermionAction):
    dirac: CloverWilson

    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        num_flavor: int,
        tol: float,
        maxiter: int,
        clover_csw: float,
        verbosity: QudaVerbosity = QudaVerbosity.QUDA_SILENT,
    ) -> None:
        kappa = 1 / (2 * (mass + Nd))
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, CloverWilson(latt_info, mass, kappa, tol, maxiter, clover_csw, 1, None))

        self.kappa2 = -(kappa**2)
        self.ck = -kappa * clover_csw / 8
        self.num_flavor = num_flavor

        self.phi = LatticeFermion(latt_info)
        self.eta = LatticeFermion(latt_info)
        self.rhmc_param = rhmc_param.wilson[num_flavor]

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE  # This is set to compute action
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        self.invert_param.verbosity = verbosity

        self.coeff = self.dirac.forceCoeff(self.rhmc_param.residue_molecular_dynamics)

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
            - self.latt_info.volume_cb2 * Ns * Nc
            - self.num_flavor * self.invert_param.trlogA[1]
        )

    def force(self, dt, new_gauge: bool):
        self.updateClover(new_gauge)
        nvector = len(self.rhmc_param.offset_molecular_dynamics)
        xx = self.invertMultiShift("molecular_dynamics")
        # Some conventions force the dagger to be YES here
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        computeCloverForceQuda(
            nullptr,
            dt,
            xx.even_ptrs,
            self.coeff,
            self.kappa2,
            self.ck,
            nvector,
            self.num_flavor,
            self.gauge_param,
            self.invert_param,
        )
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
