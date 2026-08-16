from typing import List, Optional, Union
from weakref import finalize

import numpy as np

from ..enum_quda import QudaLinkType, QudaPrecision
from ..field import LatticeGauge, LatticeInfo
from ..quda import (
    activateStaggeredRotatingLinkContextQuda,
    createStaggeredRotatingLinkContextQuda,
    destroyStaggeredRotatingLinkContextQuda,
    loadHISQRotatingOrbitalSpinLinkCacheQuda,
    loadRotatingXGaugeQuda,
    saveHISQRotatingOrbitalSpinLinkCacheQuda,
)
from . import general
from .abstract import Multigrid
from .hisq import HISQDirac


class HISQRotatingPathLinks:
    """PyQUDA snapshots of the cached orbital and spin transporters.

    The entries are references into four ordinary ``LatticeGauge`` snapshots,
    not aliases of QUDA's native Dslash cache. Mutating a snapshot therefore
    cannot change the operator. The tuple order is documented by each
    property and uses ``False``/``True`` hop signs.
    """

    def __init__(
        self,
        vxxtau_minus_t: LatticeGauge,
        vxxtau_plus_t: LatticeGauge,
        vxyt_minus_t: LatticeGauge,
        vxyt_plus_t: LatticeGauge,
    ) -> None:
        """Store the four direction-packed cache snapshots.

        Args:
            vxxtau_minus_t: VXXTau slots with a negative temporal hop.
            vxxtau_plus_t: VXXTau slots with a positive temporal hop.
            vxyt_minus_t: VXYT slots with a negative temporal hop.
            vxyt_plus_t: VXYT slots with a positive temporal hop.
        """
        self._vxxtau_minus_t = vxxtau_minus_t
        self._vxxtau_plus_t = vxxtau_plus_t
        self._vxyt_minus_t = vxyt_minus_t
        self._vxyt_plus_t = vxyt_plus_t

    @property
    def xxt(self):
        """Return xxt links ordered by ``(plus_x, plus_t)`` bit index."""
        return (
            self._vxxtau_minus_t[1],
            self._vxxtau_minus_t[3],
            self._vxxtau_plus_t[1],
            self._vxxtau_plus_t[3],
        )

    @property
    def yyt(self):
        """Return yyt links ordered by ``(plus_y, plus_t)`` bit index."""
        return (
            self._vxxtau_minus_t[0],
            self._vxxtau_minus_t[2],
            self._vxxtau_plus_t[0],
            self._vxxtau_plus_t[2],
        )

    @property
    def xyt(self):
        """Return xyt links ordered by ``(plus_x, plus_y, plus_t)`` bit index."""
        return tuple(self._vxyt_minus_t[index] for index in range(4)) + tuple(
            self._vxyt_plus_t[index] for index in range(4)
        )


class HISQRotatingDirac(HISQDirac):
    """Native QUDA HISQ operator with a rotating-frame correction."""

    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        angular_velocity: float,
        naik_epsilon: float = 0.0,
        cache_rotation_links: bool = True,
        multigrid: Union[List[List[int]], Multigrid, None] = None,
    ) -> None:
        """Create a rotating HISQ operator.

        The rotation axis is fixed at the shift-center-half midpoint of even
        global x and y extents.

        With ``cache_rotation_links=True``, QUDA stores the 16 full 3x3
        complex path matrices used at each site and unique solver precision
        ``p``. For four-volume ``V``, the total is
        ``288 * V * sum_p(sizeof(real_p))`` bytes, excluding allocator
        alignment: ``2304 * V`` for double only and ``3456 * V`` for
        double/single. The cache removes 37728 FLOPs per output site and
        right-hand side: ``37728 * V`` for a full-site Dslash, or
        ``18864 * V`` for one checkerboard. MILC phase sign flips are not
        included. Disable it to trade recomputation for lower device memory.

        When ``multigrid`` is provided, QUDA builds an ordinary HISQ
        coarse operator, while null-vector setup and the fine smoother use
        the complete rotating operator. The hierarchy preconditions an outer
        rotating GCR whose true-residual check guarantees that the converged
        solution is the rotating HISQ one.

        On the local ``16^4`` double-precision regression, caching adds
        144 MiB. It reduced the rotating-correction sub-kernel from 56.1 ms
        to 1.49 ms (about 38x), while constructing the cache took 22.6 ms per
        gauge and precision refresh. These timings are not end-to-end solver
        timings.

        Args:
            latt_info: Anti-periodic fermion lattice geometry.
            mass: Bare staggered mass; use zero when absorbed into a rational approximation.
            tol: Solver residual tolerance.
            maxiter: Maximum solver iterations.
            angular_velocity: Rotation angular velocity in lattice units.
            naik_epsilon: HISQ Naik correction coefficient.
            cache_rotation_links: Cache orbital/spin path matrices when true.
            multigrid: Existing multigrid object or block sizes for a new hierarchy.
        """
        if any(extent % 2 for extent in latt_info.global_size[:2]):
            raise ValueError("rotating HISQ requires even global x and y extents")

        self.angular_velocity = float(angular_velocity)
        if not np.isfinite(self.angular_velocity):
            raise ValueError("angular velocity must be finite")
        self.rotation_enabled = abs(self.angular_velocity) > np.finfo(np.float64).eps
        self.cache_rotation_links = bool(cache_rotation_links)
        self._rotating_link_context = 0
        self._rotating_link_context_finalizer = None
        self._rotating_path_links: Optional[HISQRotatingPathLinks] = None

        if self.rotation_enabled:
            self._rotating_link_context = createStaggeredRotatingLinkContextQuda()
            if not self._rotating_link_context:
                raise RuntimeError("QUDA failed to create the rotating HISQ link context")
            self._rotating_link_context_finalizer = finalize(
                self,
                destroyStaggeredRotatingLinkContextQuda,
                self._rotating_link_context,
            )

        super().__init__(latt_info, mass, tol, maxiter, naik_epsilon, multigrid)
        self.invert_param.angular_velocity = self.angular_velocity

        if self.multigrid.inv_param is not None:
            # Keep the assembled coarse operator ordinary HISQ, but generate
            # null vectors and smooth on the fine grid with the exact rotating
            # operator. The rotating resident links are loaded before MG setup.
            self.multigrid.inv_param.angular_velocity = self.angular_velocity

        # Preserve the machine-precision golden path without MG. Multigrid
        # uses QUDA's standard mixed-precision roles; the native rotation
        # loader creates one level-2 X field/cache per unique precision.
        if multigrid is None:
            double = QudaPrecision.QUDA_DOUBLE_PRECISION
            self.setPrecision(
                cuda=double,
                sloppy=double,
                refinement_sloppy=double,
                precondition=double,
                eigensolver=double,
            )
        else:
            # Rotating Dslash is instantiated for double and single. Keep the
            # ordinary HISQ MG hierarchy and rotating sloppy operator single.
            self.setPrecision(precondition=QudaPrecision.QUDA_SINGLE_PRECISION)

    def _loadRotatingLinks(self, level2_x: LatticeGauge) -> None:
        """Load cache-on or cache-off native links and activate this context.

        Args:
            level2_x: Pure level-2 X links before the epsilon correction.
        """
        if not self.rotation_enabled:
            return
        if not self._rotating_link_context:
            self._rotating_link_context = createStaggeredRotatingLinkContextQuda()
            if not self._rotating_link_context:
                raise RuntimeError("QUDA failed to recreate the rotating HISQ link context")
            self._rotating_link_context_finalizer = finalize(
                self,
                destroyStaggeredRotatingLinkContextQuda,
                self._rotating_link_context,
            )
        link_type = self.gauge_param.type
        use_resident_gauge = self.gauge_param.use_resident_gauge
        try:
            self.gauge_param.type = QudaLinkType.QUDA_ASQTAD_FAT_LINKS
            self.gauge_param.use_resident_gauge = 0
            if self.cache_rotation_links:
                loadHISQRotatingOrbitalSpinLinkCacheQuda(
                    self._rotating_link_context, level2_x.data_ptrs, self.gauge_param
                )
            else:
                loadRotatingXGaugeQuda(
                    self._rotating_link_context, level2_x.data_ptrs, self.gauge_param
                )
            activateStaggeredRotatingLinkContextQuda(self._rotating_link_context)
            self._rotating_path_links = None
        finally:
            self.gauge_param.type = link_type
            self.gauge_param.use_resident_gauge = use_resident_gauge

    def setFatLongGauge(self, level2_x: LatticeGauge, level2_long: LatticeGauge, w_link: LatticeGauge):
        """Load rotating and ordinary epsilon-corrected HISQ links.

        Args:
            level2_x: Pure level-2 fat links used by the rotating paths.
            level2_long: Pure level-2 long links.
            w_link: Reunitarized level-1 links used by the epsilon correction.

        Returns:
            The epsilon-corrected fat and long links loaded into QUDA.
        """
        if self.rotation_enabled:
            self._loadRotatingLinks(level2_x)
        fatlink, longlink = self.computeXLinkEpsilon(level2_x, level2_long, w_link)
        self.loadFatLongGauge(fatlink, longlink)
        return fatlink, longlink

    def getRotatingPathLinks(self) -> HISQRotatingPathLinks:
        """Return lazy PyQUDA snapshots of xxt, yyt, and xyt path links.

        The first request allocates four ``LatticeGauge`` fields, or 144 MiB
        on the ``16^4`` double-precision test. Cache-off mode builds a temporary
        native path cache only for this export. Subsequent calls reuse the
        snapshots until the gauge is refreshed.

        Returns:
            Structured references to four xxt, four yyt, and eight xyt links.

        Raises:
            RuntimeError: If no nonzero-rotation gauge has been loaded.
        """
        if not self.rotation_enabled or not self._rotating_link_context:
            raise RuntimeError("rotating path links require a loaded nonzero-rotation operator")
        if self._rotating_path_links is not None:
            return self._rotating_path_links

        vxxtau_minus_t = LatticeGauge(self.latt_info)
        vxxtau_plus_t = LatticeGauge(self.latt_info)
        vxyt_minus_t = LatticeGauge(self.latt_info)
        vxyt_plus_t = LatticeGauge(self.latt_info)
        link_type = self.gauge_param.type
        use_resident_gauge = self.gauge_param.use_resident_gauge
        try:
            self.gauge_param.type = QudaLinkType.QUDA_ASQTAD_FAT_LINKS
            self.gauge_param.use_resident_gauge = 0
            saveHISQRotatingOrbitalSpinLinkCacheQuda(
                vxxtau_minus_t.data_ptrs,
                vxxtau_plus_t.data_ptrs,
                vxyt_minus_t.data_ptrs,
                vxyt_plus_t.data_ptrs,
                self._rotating_link_context,
                self.gauge_param,
            )
        finally:
            self.gauge_param.type = link_type
            self.gauge_param.use_resident_gauge = use_resident_gauge
        self._rotating_path_links = HISQRotatingPathLinks(
            vxxtau_minus_t, vxxtau_plus_t, vxyt_minus_t, vxyt_plus_t
        )
        return self._rotating_path_links

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        """Build and load all HISQ links from a physical thin gauge.

        Args:
            gauge: Unphased physical thin gauge field.
            thin_update_only: Forwarded to multigrid hierarchy refresh.
        """
        u_link = self.computeULink(gauge)
        w_link = self.computeWLink(u_link)
        level2_x, level2_long = self.computeXLink(w_link)
        self.setFatLongGauge(level2_x, level2_long, w_link)
        general.loadMultigrid(self.multigrid, self.invert_param, thin_update_only)

    def freeGauge(self):
        """Release ordinary resident fields and this object's rotating context."""
        super().freeGauge()
        finalizer = self._rotating_link_context_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()
        self._rotating_link_context = 0
        self._rotating_link_context_finalizer = None
        self._rotating_path_links = None
