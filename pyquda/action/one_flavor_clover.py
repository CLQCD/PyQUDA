import numpy

from .. import getLogger
from ..pointer import Pointers
from ..pyquda import computeCloverForceQuda, invertMultiShiftQuda, loadCloverQuda, loadGaugeQuda
from ..enum_quda import (
    QUDA_MAX_MULTI_SHIFT,
    QudaDagType,
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaSolutionType,
    QudaSolveType,
    QudaVerbosity,
)
from ..field import Nd, Nc, Ns, LatticeInfo, LatticeFermion, MultiLatticeFermion
from ..dirac.clover_wilson import CloverWilson

nullptr = Pointers("void", 0)

from . import FermionAction


const_fourth_root = 6.10610118771501
residue_fourth_root = [
    -5.90262826538435e-06,
    -2.63363387226834e-05,
    -8.62160355606352e-05,
    -0.000263984258286453,
    -0.000792810319715722,
    -0.00236581977385576,
    -0.00704746125114149,
    -0.0210131715847004,
    -0.0629242233443976,
    -0.190538104129215,
    -0.592816342814611,
    -1.96992441194278,
    -7.70705574740274,
    -46.55440910469,
    -1281.70053339288,
]
offset_fourth_root = [
    0.000109335909283339,
    0.000584211769074023,
    0.00181216713967916,
    0.00478464392272826,
    0.0119020708754186,
    0.0289155646996088,
    0.0695922442548162,
    0.166959610676697,
    0.400720136243831,
    0.965951931276981,
    2.35629923417205,
    5.92110728201649,
    16.0486180482883,
    53.7484938194392,
    402.99686403222,
]
residue_inv_square_root = [
    0.00943108618345698,
    0.0122499930158508,
    0.0187308029056777,
    0.0308130330025528,
    0.0521206555919226,
    0.0890870585774984,
    0.153090120000215,
    0.26493803350899,
    0.466760251501358,
    0.866223656646014,
    1.8819154073627,
    6.96033769739192,
]
offset_inv_square_root = [
    5.23045292201785e-05,
    0.000569214182255549,
    0.00226724207135389,
    0.00732861083302471,
    0.0222608882919378,
    0.0662886891030569,
    0.196319420401789,
    0.582378159903323,
    1.74664271771668,
    5.42569216297222,
    18.850085313508,
    99.6213166072174,
]


class OneFlavorClover(FermionAction):
    def __init__(self, latt_info: LatticeInfo, mass: float, tol: float, maxiter: int, clover_csw: float) -> None:
        super().__init__(latt_info)
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)

        kappa = 1 / (2 * (mass + Nd))
        self.kappa2 = -(kappa**2)
        self.ck = -kappa * clover_csw / 8
        self.num_flavor = 1

        self.dirac = CloverWilson(latt_info, mass, kappa, tol, maxiter, clover_csw, 1, None)
        self.phi = LatticeFermion(latt_info)
        self.gauge_param = self.dirac.gauge_param
        self.invert_param = self.dirac.invert_param

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE  # This is set to compute action
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT

    def updateClover(self, new_gauge: bool):
        if new_gauge:
            loadGaugeQuda(nullptr, self.gauge_param)
            loadCloverQuda(nullptr, nullptr, self.invert_param)

    def action(self, new_gauge: bool) -> float:
        self.invert_param.compute_clover_trlog = 1
        self.updateClover(new_gauge)
        self.invert_param.compute_clover_trlog = 0
        num_offset = len(offset_inv_square_root)
        self.invert_param.num_offset = num_offset
        self.invert_param.offset = offset_inv_square_root + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        self.invert_param.residue = residue_inv_square_root + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        xx = MultiLatticeFermion(self.phi.latt_info, num_offset)
        self.invert_param.compute_action = 1
        invertMultiShiftQuda(xx.even_ptrs, self.phi.odd_ptr, self.invert_param)
        self.dirac.invert_param.compute_action = 0
        return (
            self.invert_param.action[0]
            - self.latt_info.volume_cb2 * Ns * Nc
            - self.num_flavor * self.invert_param.trlogA[1]
        )

    def force(self, dt, new_gauge: bool):
        self.updateClover(new_gauge)
        num_offset = len(offset_inv_square_root)
        self.invert_param.num_offset = num_offset
        self.invert_param.offset = offset_inv_square_root + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        self.invert_param.residue = residue_inv_square_root + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        xx = MultiLatticeFermion(self.phi.latt_info, num_offset)
        invertMultiShiftQuda(xx.even_ptrs, self.phi.odd_ptr, self.invert_param)
        # Some conventions force the dagger to be YES here
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        computeCloverForceQuda(
            nullptr,
            dt,
            xx.even_ptrs,
            numpy.array(residue_inv_square_root, "<f8"),
            self.kappa2,
            self.ck,
            num_offset,
            self.num_flavor,
            self.gauge_param,
            self.invert_param,
        )
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

    def sample(self, noise: LatticeFermion, new_gauge: bool):
        self.updateClover(new_gauge)
        num_offset = len(offset_fourth_root)
        self.invert_param.num_offset = num_offset
        self.invert_param.offset = offset_fourth_root + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        self.invert_param.residue = residue_fourth_root + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        xx = MultiLatticeFermion(noise.latt_info, num_offset)
        invertMultiShiftQuda(xx.even_ptrs, noise.even_ptr, self.invert_param)
        self.phi.even = noise.even
        self.phi.odd = const_fourth_root * noise.even
        for i in range(num_offset):
            self.phi.data[1] += residue_fourth_root[i] * xx.data[i, 0]
