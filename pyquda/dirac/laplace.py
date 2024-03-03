from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType

from . import StaggeredDirac, general


class Laplace(StaggeredDirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        laplace3D: int,
    ) -> None:
        super().__init__(latt_info)
        self.mg_instance = None
        self.newQudaGaugeParam()
        self.newQudaMultigridParam()
        self.newQudaInvertParam(laplace3D)

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(self):
        self.mg_param = None
        self.mg_inv_param = None

    def newQudaInvertParam(self, laplace3D: int):
        invert_param = general.newQudaInvertParam(0, 0.125, 0, 0, 0.0, 1.0, None)
        invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
        invert_param.laplace3D = laplace3D
        self.invert_param = invert_param

    def loadGauge(self, gauge: LatticeGauge):
        general.loadGauge(gauge, self.gauge_param)

    def destroy(self):
        pass
