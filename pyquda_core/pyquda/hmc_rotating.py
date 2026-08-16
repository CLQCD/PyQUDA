"""HMC orchestration for rotating gauge and staggered-fermion actions."""

from typing import List, Optional, Union
from warnings import warn

from .action.abstract import Action, FermionAction, StaggeredFermionAction
from .dirac import GaugeDirac
from .enum_quda import QudaBoolean, QudaTboundary
from .field import LatticeInfo
from .hmc import HMC, Integrator


class RotatingHMC(HMC):
    """Keep the resident thin gauge physical while HISQ builds phased links.

    ``latt_info`` describes the torus gauge field and must be periodic.  A
    ``HISQRotatingAction`` passed in ``monomials`` owns a separate lattice-info
    object with anti-periodic fermion time boundary conditions.  Both objects
    must have the same geometry.
    """

    @staticmethod
    def _warn_inconsistent_actions(monomials, hmc_inner):
        """Reject incompatible actions and warn about suspicious rotation mixtures.

        Args:
            monomials: Actions owned by this HMC level.
            hmc_inner: Optional nested HMC level whose actions share resident fields.
        """
        from .action.gauge_rotating import TreeImprovedRotatingGaugeAction
        from .action.hisq import HISQAction, MultiHISQAction
        from .action.hisq_rotating import HISQRotatingAction, MultiHISQRotatingAction

        actions = list(monomials)
        inner = hmc_inner
        while inner is not None:
            actions.extend(inner.gauge_monomials)
            actions.extend(inner.fermion_monomials)
            inner = inner.hmc_inner

        rotating_types = (HISQRotatingAction, MultiHISQRotatingAction)
        ordinary_hisq = [
            action
            for action in actions
            if isinstance(action, (HISQAction, MultiHISQAction))
            and not isinstance(action, rotating_types)
        ]
        if ordinary_hisq:
            raise TypeError(
                "RotatingHMC requires HISQRotatingAction, including for zero angular velocity; "
                "ordinary HISQ actions assume a differently phased resident gauge"
            )

        gauge_actions = [
            action for action in actions if isinstance(action, TreeImprovedRotatingGaugeAction)
        ]
        fermion_actions = [action for action in actions if isinstance(action, rotating_types)]
        fermion_configurations = {
            (
                float(action.angular_velocity),
                bool(action.dirac.cache_rotation_links),
            )
            for action in fermion_actions
        }
        if len(fermion_configurations) > 1:
            warn(
                "rotating HISQ actions use different angular velocities or link-cache policies; "
                "only compatible actions will be fused",
                RuntimeWarning,
                stacklevel=3,
            )
        if gauge_actions and fermion_actions:
            angular_velocities = {
                float(action.angular_velocity) for action in gauge_actions + fermion_actions
            }
            if len(angular_velocities) != 1:
                warn(
                    "rotating gauge and HISQ actions use different angular velocities",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def __init__(
        self,
        latt_info: LatticeInfo,
        monomials: List[Union[Action, FermionAction, StaggeredFermionAction]],
        integrator: Integrator,
        hmc_inner: Optional[HMC] = None,
    ) -> None:
        """Create an HMC owner for an unphased periodic resident gauge field.

        Args:
            latt_info: Periodic gauge-field lattice geometry.
            monomials: Gauge and rotating HISQ actions at this integrator level.
            integrator: PyQUDA molecular-dynamics integrator.
            hmc_inner: Optional faster nested HMC level.
        """
        if latt_info.t_boundary != QudaTboundary.QUDA_PERIODIC_T:
            raise ValueError("RotatingHMC requires periodic gauge-field lattice_info")
        self._warn_inconsistent_actions(monomials, hmc_inner)
        super().__init__(latt_info, monomials, integrator, hmc_inner)

        for monomial in self.fermion_monomials:
            if monomial.latt_info.global_size != latt_info.global_size:
                raise ValueError("gauge and fermion monomials must use the same lattice geometry")

        # The generic staggered HMC stores a MILC-phased resident thin gauge.
        # Rotating gauge actions need the physical torus field, so use the
        # ordinary gauge owner here and let each rotating HISQ action create
        # its own phased/APBC links from that field.
        self.is_staggered = False
        self._dirac = GaugeDirac(latt_info)
        self.gauge_param = self._dirac.gauge_param
        self.obs_param.remove_staggered_phase = QudaBoolean.QUDA_BOOLEAN_FALSE

        # Every nested level updates the same resident physical gauge and
        # momentum.  Sharing the parameter object keeps the reconstruction
        # convention identical across those updates.
        inner = self.hmc_inner
        while inner is not None:
            inner.gauge_param = self.gauge_param
            inner = inner.hmc_inner

    def fuseFermionAction(self):
        """Combine rotating HISQ actions with identical global and local lattice
        sizes, temporal boundary condition, color count, angular velocity, and
        rotation-link-cache setting into MultiHISQRotatingAction instances.

        Masses, rational approximations, and Naik epsilon values may differ.
        """
        super().fuseFermionAction()
        from .action.hisq_rotating import HISQRotatingAction, MultiHISQRotatingAction

        rotating = [action for action in self.fermion_monomials if type(action) is HISQRotatingAction]
        groups = {}
        for action in rotating:
            key = (
                tuple(action.latt_info.global_size),
                tuple(action.latt_info.size),
                action.latt_info.t_boundary,
                action.latt_info.Nc,
                action.angular_velocity,
                action.dirac.cache_rotation_links,
            )
            groups.setdefault(key, []).append(action)

        fused = [
            MultiHISQRotatingAction(group[0].latt_info, group) if len(group) > 1 else group[0]
            for group in groups.values()
        ]
        self.fermion_monomials = fused + [
            action for action in self.fermion_monomials if type(action) is not HISQRotatingAction
        ]


__all__ = ["RotatingHMC"]
