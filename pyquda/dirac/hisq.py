from typing import List

import numpy

from ..pointer import Pointers
from ..pyquda import newMultigridQuda, destroyMultigridQuda, computeKSLinkQuda
from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType, QudaInverterType, QudaReconstructType, QudaPrecision

from . import StaggeredDirac, general


class HISQ(StaggeredDirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        kappa: float,
        tol: float,
        maxiter: int,
        tadpole_coeff: float = 1.0,
        naik_epsilon: float = 0.0,
        geo_block_size: List[List[int]] = None,
    ) -> None:
        super().__init__(latt_info)
        # Using half with multigrid doesn't work
        if geo_block_size is not None:
            self._setPrecision(sloppy=max(self.precision.sloppy, QudaPrecision.QUDA_SINGLE_PRECISION))
        self._setReconstruct(
            cuda=max(self.reconstruct.cuda, QudaReconstructType.QUDA_RECONSTRUCT_NO),
            sloppy=max(self.reconstruct.sloppy, QudaReconstructType.QUDA_RECONSTRUCT_NO),
            precondition=max(self.reconstruct.precondition, QudaReconstructType.QUDA_RECONSTRUCT_NO),
            eigensolver=max(self.reconstruct.eigensolver, QudaReconstructType.QUDA_RECONSTRUCT_NO),
        )
        self.mg_instance = None
        self.newCoeff(tadpole_coeff)
        self.newQudaGaugeParam(tadpole_coeff, naik_epsilon)
        self.newQudaMultigridParam(geo_block_size, mass, kappa, 0.25, 16, 1e-6, 1000, 0, 8)
        self.newQudaInvertParam(mass, kappa, tol, maxiter)

    def newCoeff(self, tadpole_coeff: float):
        u1 = 1.0 / tadpole_coeff
        u2 = u1 * u1
        u4 = u2 * u2
        u6 = u4 * u2
        self.fat7_coeff = numpy.asarray(
            [  # First path: create V, W links
                (1.0 / 8.0),  # one link
                u2 * (0.0),  # Naik
                u2 * (-1.0 / 8.0) * 0.5,  # simple staple
                u4 * (1.0 / 8.0) * 0.25 * 0.5,  # displace link in two directions
                u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),  # displace link in three directions
                u4 * (0.0),  # Lepage term
            ],
            "<f8",
        )
        self.level2_coeff = numpy.asarray(
            [  # Second path: create X, long links
                ((1.0 / 8.0) + (2.0 * 6.0 / 16.0) + (1.0 / 8.0)),  # one link
                # One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik
                (-1.0 / 24.0),  # Naik
                (-1.0 / 8.0) * 0.5,  # simple staple
                (1.0 / 8.0) * 0.25 * 0.5,  # displace link in two directions
                (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),  # displace link in three directions
                (-2.0 / 16.0),  # Lepage term, correct O(a^2) 2x ASQTAD
            ],
            "<f8",
        )
        self.level3_coeff = numpy.asarray(
            [  # Paths for epsilon corrections. Not used if n_naiks = 1.
                (1.0 / 8.0),  # one link b/c of Naik
                (-1.0 / 24.0),  # Naik
                0.0,  # simple staple
                0.0,  # displace link in two directions
                0.0,  # displace link in three directions
                0.0,  # Lepage term
            ],
            "<f8",
        )

    def newQudaGaugeParam(self, tadpole_coeff: float, naik_epsilon: float):
        gauge_param = general.newQudaGaugeParam(
            self.latt_info, tadpole_coeff, naik_epsilon, self.precision, self.reconstruct
        )
        self.gauge_param = gauge_param

    def newQudaMultigridParam(
        self,
        geo_block_size: List[List[int]],
        mass: float,
        kappa: float,
        coarse_tol: float,
        coarse_maxiter: int,
        setup_tol: float,
        setup_maxiter: int,
        nu_pre: int,
        nu_post: int,
    ):
        if geo_block_size is not None:
            mg_param, mg_inv_param = general.newQudaMultigridParam(
                mass,
                kappa,
                geo_block_size,
                coarse_tol,
                coarse_maxiter,
                setup_tol,
                setup_maxiter,
                nu_pre,
                nu_post,
                self.precision,
            )
            mg_inv_param.dslash_type = QudaDslashType.QUDA_ASQTAD_DSLASH
        else:
            mg_param, mg_inv_param = None, None
        self.mg_param = mg_param
        self.mg_inv_param = mg_inv_param

    def newQudaInvertParam(self, mass: float, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(mass, kappa, tol, maxiter, 0.0, 1.0, self.mg_param, self.precision)
        invert_param.dslash_type = QudaDslashType.QUDA_ASQTAD_DSLASH
        if self.mg_param is not None:
            invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
        else:
            invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param = invert_param

    def computeFatLong(self, gauge: LatticeGauge):
        nullptr = Pointers("void", 0)
        inlink = gauge.copy()
        ulink = LatticeGauge(gauge.latt_info)
        fatlink = LatticeGauge(gauge.latt_info)
        longlink = LatticeGauge(gauge.latt_info)

        # gauge_param.use_resident_gauge = 0
        # loadGaugeQuda(inlink.data_ptrs, gauge_param)  # Save the original gauge for the smeared source.
        # gauge_param.use_resident_gauge = 1

        # t boundary will be applied by the staggered phase.
        inlink.staggeredPhase()
        self.gauge_param.staggered_phase_applied = 1

        # Chroma uses periodic boundary condition to do the SU(3) projection.
        # But I think it's wrong.
        # gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        computeKSLinkQuda(
            nullptr,
            nullptr,
            ulink.data_ptrs,
            inlink.data_ptrs,
            self.fat7_coeff,
            self.gauge_param,
        )
        computeKSLinkQuda(
            fatlink.data_ptrs,
            longlink.data_ptrs,
            nullptr,
            ulink.data_ptrs,
            self.level2_coeff,
            self.gauge_param,
        )

        return fatlink, longlink

    def loadGauge(self, gauge: LatticeGauge):
        fatlink, longlink = self.computeFatLong(gauge)
        general.loadFatLongGauge(fatlink, longlink, self.gauge_param)
        if self.mg_param is not None:
            if self.mg_instance is not None:
                self.destroy()
            self.mg_instance = newMultigridQuda(self.mg_param)
            self.invert_param.preconditioner = self.mg_instance

    def destroy(self):
        if self.mg_instance is not None:
            destroyMultigridQuda(self.mg_instance)
            self.mg_instance = None
