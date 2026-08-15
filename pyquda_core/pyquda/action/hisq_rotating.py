from typing import List, Optional

import numpy as np

from ..enum_quda import (
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaParity,
    QudaSolutionType,
    QudaSolveType,
    QudaTboundary,
)
from ..field import LatticeGauge, LatticeInfo, LatticeMom, LatticeStaggeredFermion, MultiLatticeStaggeredFermion
from ..quda import computeHISQForceQuda, computeHISQRotatingForceQuda, dslashQuda, saveGaugeQuda
from ..dirac import getGlobalReconstruct
from ..dirac.hisq_rotating import HISQRotatingDirac
from .abstract import RationalParam, StaggeredFermionAction
from .hisq import HISQAction, MultiHISQAction, nullptr


class HISQRotatingAction(StaggeredFermionAction):
    """Even-pseudofermion HISQ action in a rotating frame."""

    dirac: HISQRotatingDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        rational_param: RationalParam,
        tol: float,
        maxiter: int,
        naik_epsilon: float = 0.0,
        angular_velocity: float = 0.0,
        mass: float = 0.0,
        cache_rotation_links: bool = True,
    ) -> None:
        """Create the even-pseudofermion rotating HISQ action.

        The rotation axis is fixed at the shift-center-half midpoint of even
        global x and y extents.

        Args:
            latt_info: Anti-periodic fermion lattice geometry.
            rational_param: Sampling, action, and force rational approximations.
            tol: Multi-shift solver residual tolerance.
            maxiter: Maximum solver iterations.
            naik_epsilon: HISQ Naik correction coefficient.
            angular_velocity: Rotation angular velocity in lattice units.
            mass: Mass already absorbed into the rational approximation when zero.
            cache_rotation_links: Retain orbital/spin path matrices instead of level-2 X.

        For every unique solver precision ``p``, the link cache adds
        ``288 * sizeof(real_p) * V`` device bytes. A double-only solve costs
        ``2304 * V`` bytes; a double/single MG solve costs ``3456 * V``.
        It avoids 37728 FLOPs per output site and right-hand side:
        ``37728 * V`` for a full-site Dslash, or ``18864 * V`` for one
        checkerboard. Disable it when conserving memory matters more.
        """
        self.angular_velocity = float(angular_velocity)
        if not np.isfinite(self.angular_velocity):
            raise ValueError("angular velocity must be finite")
        dirac = HISQRotatingDirac(
            latt_info,
            mass,
            tol,
            maxiter,
            self.angular_velocity,
            naik_epsilon,
            cache_rotation_links,
        )
        super().__init__(latt_info, dirac)

        # Reuse coefficient bookkeeping without inheriting HISQAction: the HMC
        # driver must not fold this action into ordinary MultiHISQAction.
        HISQAction.setForceParam(self, rational_param)
        self.quark = None
        self.phi = LatticeStaggeredFermion(latt_info)
        self.eta = LatticeStaggeredFermion(latt_info)

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION

    def updateFatLong(self) -> None:
        """Build and load level-2 links for a rational solve."""
        thin_link = LatticeGauge(self.latt_info)
        t_boundary = self.gauge_param.t_boundary
        staggered_phase_applied = self.gauge_param.staggered_phase_applied
        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.staggered_phase_applied = 0
        try:
            saveGaugeQuda(thin_link.data_ptrs, self.gauge_param)
        finally:
            self.gauge_param.t_boundary = t_boundary
            self.gauge_param.staggered_phase_applied = staggered_phase_applied

        # RotatingHMC keeps the resident thin gauge physical and periodic.
        # HISQ links carry MILC phases and fermion APBC locally instead.
        u_link = self.dirac.computeULink(thin_link)
        w_link = self.dirac.computeWLink(u_link)
        level2_fat, level2_long = self.dirac.computeXLink(w_link)
        self.dirac.setFatLongGauge(level2_fat, level2_long, w_link)

    def updateFatLongReturn(self):
        """Build force-chain links, load level-2 links, and return all stages.

        Returns:
            ``(level2_fat, u_link, v_link, w_link)`` for the rotating force.
        """
        thin_link = LatticeGauge(self.latt_info)
        t_boundary = self.gauge_param.t_boundary
        staggered_phase_applied = self.gauge_param.staggered_phase_applied
        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.staggered_phase_applied = 0
        try:
            saveGaugeQuda(thin_link.data_ptrs, self.gauge_param)
        finally:
            self.gauge_param.t_boundary = t_boundary
            self.gauge_param.staggered_phase_applied = staggered_phase_applied

        u_link = self.dirac.computeULink(thin_link)
        v_link, w_link = self.dirac.computeVWLink(u_link)
        level2_fat, level2_long = self.dirac.computeXLink(w_link)
        self.dirac.setFatLongGauge(level2_fat, level2_long, w_link)
        return level2_fat, u_link, v_link, w_link

    def invertMultiShift(self, mode):
        """Run the inherited multi-shift solve without leaking temporary state.

        Args:
            mode: One of the base action's ``sample``, ``action``, or ``force`` modes.

        Returns:
            The value returned by ``StaggeredFermionAction.invertMultiShift``.
        """
        state = (
            self.invert_param.cuda_prec_sloppy,
            self.invert_param.cuda_prec_precondition,
            self.invert_param.solution_type,
            self.invert_param.dagger,
        )
        try:
            return super().invertMultiShift(mode)
        finally:
            (
                self.invert_param.cuda_prec_sloppy,
                self.invert_param.cuda_prec_precondition,
                self.invert_param.solution_type,
                self.invert_param.dagger,
            ) = state

    def sample(self):
        """Draw an even pseudofermion with the configured rational approximation."""
        if self.quark is None:
            self.quark = MultiLatticeStaggeredFermion(self.latt_info, self.max_num_offset)
        self.sampleEta()
        self.eta.data[1] = 0
        self.updateFatLong()
        self.invertMultiShift("sample")

    def action(self, use_force_param: bool) -> float:
        """Evaluate the pseudofermion action.

        Args:
            use_force_param: Use the force rational approximation and QUDA action accumulation.

        Returns:
            Pseudofermion action value.
        """
        if self.quark is None:
            self.quark = MultiLatticeStaggeredFermion(self.latt_info, self.max_num_offset)
        self.updateFatLong()
        if use_force_param:
            compute_action = self.invert_param.compute_action
            self.invert_param.compute_action = 1
            try:
                self.invertMultiShift("force")
            finally:
                self.invert_param.compute_action = compute_action
            return float(self.invert_param.action[0])

        self.invertMultiShift("action")
        return float(self.eta.even.norm2())

    def force(self, dt: float, mom: Optional[LatticeMom] = None):
        """Add the epsilon-corrected rotating HISQ force.

        Args:
            dt: Molecular-dynamics step multiplying the force.
            mom: Optional explicit momentum destination; resident momentum is used when omitted.
        """
        assert self.quark is not None
        level2_fat, u_link, v_link, w_link = self.updateFatLongReturn()
        self.invertMultiShift("force")

        # Native dslash dispatch fills the odd component from each solved even
        # field; the packed field is then consumed by the HISQ force kernel.
        for i in range(self.num):
            dslashQuda(self.quark[i].odd_ptr, self.quark[i].even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)

        momentum_state = None
        if mom is not None:
            tmp = LatticeMom(mom.latt_info)
            momentum_state = (
                self.gauge_param.use_resident_mom,
                self.gauge_param.make_resident_mom,
                self.gauge_param.return_result_mom,
            )
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1

        reconstruct = self.gauge_param.reconstruct
        self.gauge_param.reconstruct = getGlobalReconstruct("staggered").cuda
        try:
            common_args = (
                nullptr if mom is None else tmp.data_ptrs,
                dt,
                self.dirac.path_coeff_2,
                self.dirac.path_coeff_1,
            )
            field_args = (
                w_link.data_ptrs,
                v_link.data_ptrs,
                u_link.data_ptrs,
                self.quark.data_ptrs,
                self.num,
                self.num_naik,
                self.coeff,
            )
            if not self.dirac.rotation_enabled:
                computeHISQForceQuda(*common_args, *field_args, self.gauge_param)
            else:
                computeHISQRotatingForceQuda(
                    *common_args,
                    level2_fat.data_ptrs,
                    *field_args,
                    self.angular_velocity,
                    self.gauge_param,
                )
        finally:
            self.gauge_param.reconstruct = reconstruct
            if momentum_state is not None:
                (
                    self.gauge_param.use_resident_mom,
                    self.gauge_param.make_resident_mom,
                    self.gauge_param.return_result_mom,
                ) = momentum_state

        if mom is not None:
            mom += tmp


class MultiHISQRotatingAction(MultiHISQAction):
    """Fuse rotating HISQ pseudofermions with identical lattice and rotation settings."""

    dirac: HISQRotatingDirac

    def __init__(self, latt_info: LatticeInfo, pseudo_fermions: List[HISQRotatingAction]) -> None:
        """Fuse rotating pseudofermions into one smearing backward pass.

        Global and local lattice sizes, temporal boundary condition, color
        count, angular velocity, and rotation-link-cache setting must match.
        Masses, rational approximations, and Naik epsilon values may differ.

        Args:
            latt_info: Shared anti-periodic fermion lattice geometry.
            pseudo_fermions: Rotating HISQ actions satisfying the fields listed above.
        """
        if not pseudo_fermions:
            raise ValueError("at least one rotating HISQ pseudofermion is required")
        if not all(type(action) is HISQRotatingAction for action in pseudo_fermions):
            raise TypeError("MultiHISQRotatingAction only accepts HISQRotatingAction instances")

        reference = pseudo_fermions[0]
        reference_geometry = (
            tuple(reference.latt_info.global_size),
            tuple(reference.latt_info.size),
            reference.latt_info.t_boundary,
            reference.latt_info.Nc,
        )
        reference_rotation = (reference.angular_velocity, reference.dirac.cache_rotation_links)
        for action in pseudo_fermions[1:]:
            geometry = (
                tuple(action.latt_info.global_size),
                tuple(action.latt_info.size),
                action.latt_info.t_boundary,
                action.latt_info.Nc,
            )
            rotation = (action.angular_velocity, action.dirac.cache_rotation_links)
            if geometry != reference_geometry:
                raise ValueError(
                    "rotating HISQ pseudofermions must use identical global and local lattice sizes, "
                    "temporal boundary conditions, and color counts"
                )
            if rotation != reference_rotation:
                raise ValueError(
                    "rotating HISQ pseudofermions must use identical angular velocities and "
                    "rotation-link-cache settings"
                )

        super().__init__(latt_info, pseudo_fermions)
        self.angular_velocity = reference.angular_velocity

    def prepareFatLong(self) -> None:
        """Build shared pure level-2 links for fused rational solves."""
        thin_link = LatticeGauge(self.latt_info)
        t_boundary = self.gauge_param.t_boundary
        staggered_phase_applied = self.gauge_param.staggered_phase_applied
        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.staggered_phase_applied = 0
        try:
            saveGaugeQuda(thin_link.data_ptrs, self.gauge_param)
        finally:
            self.gauge_param.t_boundary = t_boundary
            self.gauge_param.staggered_phase_applied = staggered_phase_applied

        u_link = self.dirac.computeULink(thin_link)
        w_link = self.dirac.computeWLink(u_link)
        self.level2_fat, self.level2_long = self.dirac.computeXLink(w_link)
        self.w_link = w_link
        if self.dirac.rotation_enabled:
            self.dirac._loadRotatingLinks(self.level2_fat)
        self.current_naik_epsilon = None

    def prepareFatLongReturn(self):
        """Build shared force-chain links and return all smearing stages.

        Returns:
            ``(level2_fat, u_link, v_link, w_link)`` for the fused force.
        """
        thin_link = LatticeGauge(self.latt_info)
        t_boundary = self.gauge_param.t_boundary
        staggered_phase_applied = self.gauge_param.staggered_phase_applied
        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.staggered_phase_applied = 0
        try:
            saveGaugeQuda(thin_link.data_ptrs, self.gauge_param)
        finally:
            self.gauge_param.t_boundary = t_boundary
            self.gauge_param.staggered_phase_applied = staggered_phase_applied

        u_link = self.dirac.computeULink(thin_link)
        v_link, w_link = self.dirac.computeVWLink(u_link)
        self.level2_fat, self.level2_long = self.dirac.computeXLink(w_link)
        self.w_link = w_link
        if self.dirac.rotation_enabled:
            self.dirac._loadRotatingLinks(self.level2_fat)
        self.current_naik_epsilon = None
        return self.level2_fat, u_link, v_link, w_link

    def updateFatLong(self, pseudo_fermion: HISQRotatingAction):
        """Load one pseudofermion's epsilon-corrected links.

        Args:
            pseudo_fermion: Action whose Naik epsilon selects the loaded links.
        """
        dirac = pseudo_fermion.dirac
        if self.current_naik_epsilon != dirac.naik_epsilon:
            fatlink, longlink = dirac.computeXLinkEpsilon(self.level2_fat, self.level2_long, self.w_link)
            dirac.loadFatLongGauge(fatlink, longlink)
            self.current_naik_epsilon = dirac.naik_epsilon

    def force(self, dt: float, mom: Optional[LatticeMom] = None):
        """Add the fused rotating HISQ force.

        Args:
            dt: Molecular-dynamics force scale.
            mom: Optional explicit momentum destination; resident momentum is used when omitted.
        """
        level2_fat, u_link, v_link, w_link = self.prepareFatLongReturn()
        num_current = 0
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.quark = self.quark[num_current : num_current + pseudo_fermion.num]
            pseudo_fermion.invertMultiShift("force")
            for i in range(pseudo_fermion.num):
                dslashQuda(
                    self.quark[num_current + i].odd_ptr,
                    self.quark[num_current + i].even_ptr,
                    pseudo_fermion.invert_param,
                    QudaParity.QUDA_ODD_PARITY,
                )
            num_current += pseudo_fermion.num

        momentum_state = None
        if mom is not None:
            tmp = LatticeMom(mom.latt_info)
            momentum_state = (
                self.gauge_param.use_resident_mom,
                self.gauge_param.make_resident_mom,
                self.gauge_param.return_result_mom,
            )
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1

        reconstruct = self.gauge_param.reconstruct
        self.gauge_param.reconstruct = getGlobalReconstruct("staggered").cuda
        try:
            common_args = (
                nullptr if mom is None else tmp.data_ptrs,
                dt,
                self.dirac.path_coeff_2,
                self.dirac.path_coeff_1,
            )
            field_args = (
                w_link.data_ptrs,
                v_link.data_ptrs,
                u_link.data_ptrs,
                self.quark.data_ptrs,
                self.num,
                self.num_naik,
                self.coeff,
            )
            if not self.dirac.rotation_enabled:
                computeHISQForceQuda(*common_args, *field_args, self.gauge_param)
            else:
                computeHISQRotatingForceQuda(
                    *common_args,
                    level2_fat.data_ptrs,
                    *field_args,
                    self.angular_velocity,
                    self.gauge_param,
                )
        finally:
            self.gauge_param.reconstruct = reconstruct
            if momentum_state is not None:
                (
                    self.gauge_param.use_resident_mom,
                    self.gauge_param.make_resident_mom,
                    self.gauge_param.return_result_mom,
                ) = momentum_state

        if mom is not None:
            mom += tmp
