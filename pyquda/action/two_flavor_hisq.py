import numpy

from .. import getLogger
from ..pointer import Pointers
from ..pyquda import (
    computeHISQForceQuda,
    dslashQuda,
    computeKSLinkQuda,
    loadGaugeQuda,
    saveGaugeQuda,
)
from ..enum_quda import (
    QudaInverterType,
    QudaLinkType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaParity,
    QudaSolutionType,
    QudaSolveType,
    QudaStaggeredPhase,
    QudaVerbosity,
)
from ..field import LatticeGauge, LatticeStaggeredFermion, LatticeInfo
from ..dirac.hisq import HISQ

nullptr = Pointers("void", 0)

from . import FermionAction


const_pseudo_fermion = 6.6008968113477318e00
residue_pseudo_fermion = [
    -4.5244529588728673e-04,
    -2.7913286193527722e-03,
    -1.3910919905953502e-02,
    -6.7151227080820358e-02,
    -3.2392713062715339e-01,
    -1.5980690060213258e00,
    -8.5910485755168793e00,
    -6.3332591104281043e01,
    -1.8771881382968977e03,
]
offset_pseudo_fermion = [
    1.3387699397698993e-02,
    3.1462323899226492e-02,
    9.6415385897133263e-02,
    3.2374992445020234e-01,
    1.1208318145761189e00,
    3.9545250804438625e00,
    1.4540181635176147e01,
    6.1963910644237338e01,
    5.3566592269333353e02,
]
const_molecular_dynamics = 2.6567771557480493e-02
residue_molecular_dynamics = [
    5.4415326175599146e-02,
    9.0370308484060427e-02,
    1.8616389644513945e-01,
    4.0655052893921434e-01,
    9.1332229681707799e-01,
    2.2297215526003842e00,
    8.2673806076260998e00,
]
offset_molecular_dynamics = [
    1.1651276172701218e-02,
    3.2128005558694356e-02,
    1.3618256552525199e-01,
    6.4581932161561295e-01,
    3.1710061679824459e00,
    1.6529546460530057e01,
    1.1696719959399913e02,
]
coeff_molecular_dynamics = numpy.array([[2 * res, -1 / 24 * 2 * res] for res in residue_molecular_dynamics], "<f8")
const_fermion_action = 1.5149456635663205e-01
residue_fermion_action = [
    6.0458038124269137e-03,
    1.5247824256426594e-02,
    3.7602970968999533e-02,
    9.4912662473817022e-02,
    2.4258769955049089e-01,
    6.3038339460769466e-01,
    1.7214395396814584e00,
    5.5998391201760676e00,
    3.6431278685560251e01,
]
offset_fermion_action = [
    1.1680369733380888e-02,
    2.4528542115263768e-02,
    7.1946920045447768e-02,
    2.3818970133124504e-01,
    8.2029368099576661e-01,
    2.8788453123211895e00,
    1.0425969224178980e01,
    4.1948608522841752e01,
    2.6570653748247554e02,
]


class TwoFlavorHISQ(FermionAction):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        tadpole_coeff: float = 1.0,
        naik_epsilon: float = 0.0,
    ) -> None:
        super().__init__(latt_info)
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)

        kappa = 1 / 2
        self.num_flavor = 1

        self.dirac = HISQ(latt_info, mass, kappa, tol, maxiter, tadpole_coeff, naik_epsilon, None)
        self.phi = LatticeStaggeredFermion(latt_info)
        self.gauge_param = self.dirac.gauge_param
        self.invert_param = self.dirac.invert_param

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE  # This is set to compute action
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT

    def updateFatAndLong(self, return_v_link: bool):
        u_link = LatticeGauge(self.latt_info)
        v_link = LatticeGauge(self.latt_info) if return_v_link else None
        w_link = LatticeGauge(self.latt_info)
        fatlink = LatticeGauge(self.latt_info)
        longlink = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)

        computeKSLinkQuda(
            v_link.data_ptrs if return_v_link else nullptr,
            nullptr,
            w_link.data_ptrs,
            u_link.data_ptrs,
            self.dirac.fat7_coeff,
            self.gauge_param,
        )
        computeKSLinkQuda(
            fatlink.data_ptrs,
            longlink.data_ptrs,
            nullptr,
            w_link.data_ptrs,
            self.dirac.level2_coeff,
            self.gauge_param,
        )

        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.type = QudaLinkType.QUDA_ASQTAD_FAT_LINKS
        loadGaugeQuda(fatlink.data_ptrs, self.gauge_param)
        self.gauge_param.type = QudaLinkType.QUDA_ASQTAD_LONG_LINKS
        self.gauge_param.ga_pad = self.latt_info.ga_pad * 3
        self.gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_NO
        loadGaugeQuda(longlink.data_ptrs, self.gauge_param)
        self.gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
        self.gauge_param.ga_pad = self.latt_info.ga_pad
        self.gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_MILC
        self.gauge_param.use_resident_gauge = 1

        return u_link, v_link, w_link

    def action(self, new_gauge: bool) -> float:
        self.updateFatAndLong(False)
        self.invert_param.compute_action = 1
        self.dirac.invertMultiShiftPC(self.phi, offset_molecular_dynamics, residue_molecular_dynamics)
        self.invert_param.compute_action = 0
        return self.action2()

    def action2(self):
        import cupy as cp

        x = self.dirac.invertMultiShiftPC(self.phi, offset_fermion_action, residue_fermion_action, const_fermion_action)
        return cp.linalg.norm(x.even) ** 2  # - const_molecular_dynamics * cp.linalg.norm(self.phi.odd) ** 2

    def force(self, dt, new_gauge: bool):
        u_link, v_link, w_link = self.updateFatAndLong(True)
        num_offset = len(offset_molecular_dynamics)
        xx = self.dirac.invertMultiShiftPC(self.phi, offset_molecular_dynamics, residue_molecular_dynamics)
        for i in range(num_offset):
            dslashQuda(xx[i].odd_ptr, xx[i].even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        computeHISQForceQuda(
            nullptr,
            dt,
            self.dirac.level2_coeff,
            self.dirac.fat7_coeff,
            w_link.data_ptrs,
            v_link.data_ptrs,
            u_link.data_ptrs,
            xx.data_ptrs,
            num_offset,
            0,
            coeff_molecular_dynamics,
            self.gauge_param,
        )

    def sample(self, noise: LatticeStaggeredFermion, new_gauge: bool):
        self.updateFatAndLong(False)
        x = self.dirac.invertMultiShiftPC(noise, offset_pseudo_fermion, residue_pseudo_fermion, const_pseudo_fermion)
        self.phi.odd = noise.even
        self.phi.even = x.even
