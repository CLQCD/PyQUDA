"""Unified tree-improved *rotating* gauge action (TORUS + SHIFT-CENTER-HALF).

One PyQUDA Action class combining CLGLib's tree-improved gauge action and the
rotating action.  S = S0 + Omega*S1 + Omega^2*S2, with

  S0 = betaOverN * sum [ (3 - ReTr plaq) + Cr*(6 - ReTr rect1 - ReTr rect2) ]
  S1 = rotBetaOverN * chair terms (linear in shifted coord)
  S2 = rotBetaOverN * [ 4-plaquette(clover) terms + chair V132 ]

with betaOverN = beta/Nc, rotBetaOverN = betaOverN*rot_beta_scale (0.6),
Cr = -0.05, center C_mu = L_mu//2, shifted coord f = (coord - C + 0.5).

The force is the exact MD derivative of this energy, decomposed into per-loop
staples with *site-dependent* coefficients.  PyQUDA marshals those paths and
compact field/anchor metadata into QUDA's rotation-specific site-wise kernels.
The coordinate basis is generated once on device using rank-aware global
coordinates and remains resident for both energy and force.

NOTE this class does not use gauge.py's forcePath (which asserts a symmetric,
direction-independent coefficient set that the rotating action violates by
construction); it builds its own per-direction path + coefficient tables.
"""

from typing import List, Optional, Tuple
from weakref import finalize

import numpy

from ..field import LatticeInfo, LatticeMom
from ..enum_quda import QudaTboundary
from ..quda import (
    computeGaugeRotatingActionQuda,
    computeGaugeRotatingForceQuda,
    createGaugeRotatingContextQuda,
    destroyGaugeRotatingContextQuda,
)
from ..dirac import GaugeDirac

from .abstract import Action, LoopParam
from .gauge import GaugeAction, nullptr

_PLANES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

_FIELD_CONSTANT = -1
_FIELD_X = 0
_FIELD_Y = 1
_FIELD_X2 = 2
_FIELD_Y2 = 3
_FIELD_R2 = 4
_FIELD_XY = 5


def _plaq(mu, nu):
    """Return the positive-oriented plaquette in the ``mu``-``nu`` plane.

    Args:
        mu: First direction in PyQUDA x/y/z/t order.
        nu: Second direction in PyQUDA x/y/z/t order.
    """
    return [(mu, +1), (nu, +1), (mu, -1), (nu, -1)]


def _steps_to_quda(steps):
    """Encode signed path steps for ``LoopParam``.

    Args:
        steps: ``(direction, sign)`` path entries.
    """
    return [mu if sign > 0 else 4 + mu for mu, sign in steps]


def _steps_to_quda_force(steps):
    """Encode signed steps for QUDA's direct gauge-force interface.

    Args:
        steps: ``(direction, sign)`` path entries.
    """
    return [mu if sign > 0 else 7 - mu for mu, sign in steps]


def _rect(mu, nu):
    """Return the two 1-by-2 rectangles in the ``mu``-``nu`` plane.

    Args:
        mu: First plane direction.
        nu: Second plane direction.
    """
    return (
        [(mu, +1), (mu, +1), (nu, +1), (mu, -1), (mu, -1), (nu, -1)],
        [(mu, +1), (nu, +1), (nu, +1), (mu, -1), (nu, -1), (nu, -1)],
    )


def _clover_leaves(mu, nu):
    """Return the four plaquette leaves around one clover anchor.

    Args:
        mu: First plane direction.
        nu: Second plane direction.
    """
    return [
        [(mu, +1), (nu, +1), (mu, -1), (nu, -1)],
        [(mu, -1), (nu, -1), (mu, +1), (nu, +1)],
        [(nu, +1), (mu, -1), (nu, -1), (mu, +1)],
        [(nu, -1), (mu, +1), (nu, +1), (mu, -1)],
    ]


def _chair_loops(mu, nu, rho):
    """Return the eight signed chair loops for three distinct directions.

    Args:
        mu: First chair direction.
        nu: Second chair direction.
        rho: Third chair direction.
    """
    A = [(mu, +1), (nu, +1), (mu, -1)]
    B = [(mu, -1), (nu, +1), (mu, +1)]
    C = [(rho, +1), (nu, -1), (rho, -1)]
    D = [(rho, -1), (nu, -1), (rho, +1)]
    E = [(mu, +1), (nu, -1), (mu, -1)]
    Fp = [(mu, -1), (nu, -1), (mu, +1)]
    G = [(rho, +1), (nu, +1), (rho, -1)]
    H = [(rho, -1), (nu, +1), (rho, +1)]
    return [
        (A + C, +1.0), (A + D, -1.0), (B + C, -1.0), (B + D, +1.0),
        (E + G, +1.0), (E + H, -1.0), (Fp + G, -1.0), (Fp + H, +1.0),
    ]


# unit displacement in numpy-axis order (t,z,y,x) for each mu (0=x..3=t)
_E_TZYX = [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]


def _anchors(steps):
    """Return each link's forward-anchor site relative to the loop origin.

    Args:
        steps: Signed loop steps.

    Returns:
        Anchor offsets in PyQUDA t/z/y/x order.
    """
    pos = [0, 0, 0, 0]
    anch = []
    for (mu, s) in steps:
        if s > 0:
            anch.append(tuple(pos))
            pos = [pos[d] + _E_TZYX[mu][d] for d in range(4)]
        else:
            pos = [pos[d] - _E_TZYX[mu][d] for d in range(4)]
            anch.append(tuple(pos))
    return anch


def _staples_from_loop(steps):
    """Return (dir, staple, anchor) entries for one closed loop.

    anchor is the differentiated link position relative to the loop origin.
    QUDA evaluates every force path from the differentiated forward link's
    endpoint; the anchor is only used to evaluate a site-dependent loop
    coefficient at x - anchor.

    Args:
        steps: Signed steps of one closed loop.

    Returns:
        ``(direction, staple, anchor)`` entries for every link occurrence.
    """
    L = len(steps)
    out = []
    steps_bwd = [(mu, -s) for (mu, s) in reversed(steps)]
    anch_fwd = _anchors(steps)
    anch_bwd = _anchors(steps_bwd)
    for j in range(L):
        mu, s = steps[j]
        if s > 0:
            out.append((mu, steps[j + 1 :] + steps[:j], anch_fwd[j]))
        else:
            jb = (L - 1) - j
            out.append((mu, steps_bwd[jb + 1 :] + steps_bwd[:jb], anch_bwd[jb]))
    return out


def _energy_loop_specs(latt_info: LatticeInfo, beta, angular_velocity, c_rect, rot_beta_scale):
    """Assemble action loops and their coordinate-basis coefficients.

    Args:
        latt_info: Gauge lattice geometry and color count.
        beta: Tree-improved gauge coupling.
        angular_velocity: Rotation angular velocity in lattice units.
        c_rect: Rectangle coefficient relative to plaquettes.
        rot_beta_scale: Coupling multiplier for rotation-dependent loops.

    Returns:
        ``(steps, scalar coefficient, coordinate type)`` entries.
    """
    bon = beta / latt_info.Nc
    rbon = bon * rot_beta_scale
    table: List[Tuple[list, float, int]] = []

    for mu, nu in _PLANES:
        table.append((_plaq(mu, nu), -bon, _FIELD_CONSTANT))
        r1, r2 = _rect(mu, nu)
        table.append((r1, -c_rect * bon, _FIELD_CONSTANT))
        table.append((r2, -c_rect * bon, _FIELD_CONSTANT))

    s1f = angular_velocity * rbon
    for (mu, nu, rho, scalar, coordinate_type) in [
        (3, 0, 1, -0.125, _FIELD_X),
        (3, 2, 1, -0.125, _FIELD_X),
        (3, 1, 0, +0.125, _FIELD_Y),
        (3, 2, 0, +0.125, _FIELD_Y),
    ]:
        for steps, sgn in _chair_loops(mu, nu, rho):
            table.append((steps, s1f * sgn * scalar, coordinate_type))

    s2f = angular_velocity * angular_velocity * rbon
    for (mu, nu, coordinate_type) in [(1, 2, _FIELD_X2), (0, 2, _FIELD_Y2), (0, 1, _FIELD_R2)]:
        for leaf in _clover_leaves(mu, nu):
            table.append((leaf, s2f * (-0.25), coordinate_type))
    for steps, sgn in _chair_loops(0, 2, 1):
        table.append((steps, s2f * sgn * (-0.125), _FIELD_XY))

    return table


def _force_terms(loop_specs):
    """Differentiate closed-loop specifications into link-owned staples.

    Args:
        loop_specs: Output of ``_energy_loop_specs``.

    Returns:
        Direction, staple, anchor, coefficient, and coordinate type per occurrence.
    """
    terms = []
    for steps, coeff, coordinate_type in loop_specs:
        for direction, staple, offset in _staples_from_loop(steps):
            terms.append((direction, staple, offset, coeff, coordinate_type))
    return terms


class TreeImprovedRotatingGaugeAction(Action):
    """Unified tree-improved rotating gauge action, TORUS + SHIFT-CENTER-HALF."""

    dirac: GaugeDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        beta: float,
        angular_velocity: float,
        c_rect: float = -0.05,
        rot_beta_scale: float = 0.6,
    ):
        """Create a tree-improved gauge action in the rotating frame.

        Args:
            latt_info: Periodic gauge-field lattice geometry.
            beta: Tree-improved gauge coupling in the ``10/g^2`` convention.
            angular_velocity: Rotation angular velocity in lattice units.
            c_rect: Rectangle coefficient relative to the plaquette term.
            rot_beta_scale: Coupling multiplier for rotation-dependent loops.
        """
        if latt_info.t_boundary != QudaTboundary.QUDA_PERIODIC_T:
            raise NotImplementedError("only torus gauge boundary conditions are supported")
        if any(extent % 2 for extent in latt_info.global_size[:2]):
            raise ValueError("rotating gauge action requires even global x and y extents")
        super().__init__(latt_info, GaugeDirac(latt_info))
        angular_velocity = float(angular_velocity)
        if not numpy.isfinite(angular_velocity):
            raise ValueError("angular velocity must be finite")
        self.beta = beta
        self.angular_velocity = angular_velocity
        self.rotation_enabled = abs(angular_velocity) > numpy.finfo(numpy.float64).eps
        self.c_rect = c_rect
        self.rot_beta_scale = rot_beta_scale

        self.loop_specs = _energy_loop_specs(
            latt_info, beta, angular_velocity, c_rect, rot_beta_scale
        )
        if not self.rotation_enabled:
            self.loop_specs = self.loop_specs[: 3 * len(_PLANES)]

        base_paths = []
        base_coeff = []
        for mu, nu in _PLANES:
            base_paths.append(_steps_to_quda(_plaq(mu, nu)))
            base_coeff.append(1.0)
            r1, r2 = _rect(mu, nu)
            base_paths.extend((_steps_to_quda(r1), _steps_to_quda(r2)))
            base_coeff.extend((c_rect, c_rect))
        self.base_action = GaugeAction(latt_info, LoopParam(base_paths, base_coeff), beta)
        self.base_action.gauge_param = self.gauge_param

        self.force_terms = _force_terms(self.loop_specs)
        self._quda_context = 0
        self._quda_context_finalizer = None

        # Coordinates live in a compact, rank-aware scalar basis instead of
        # materializing one volume-sized coefficient field per path
        # derivative. The interface is deliberately rotation-specific; a
        # broader scalar-field abstraction is left to a later use case.
        radius = [0, 0, 0, 0]
        for _, _, offset_tzyx, _, field_index in self.force_terms:
            if field_index == _FIELD_CONSTANT:
                continue
            for direction, value in enumerate(offset_tzyx[::-1]):
                radius[direction] = max(radius[direction], abs(int(value)))
        self._scalar_radius = numpy.asarray(radius, dtype=numpy.int32)

    def _ensure_quda_context(self):
        """Create the persistent device coordinate/path context on first use.

        Returns:
            Opaque QUDA context handle.
        """
        if self._quda_context:
            return self._quda_context

        action_lengths = numpy.asarray(
            [len(steps) for steps, _, _ in self.loop_specs], dtype=numpy.int32
        )
        action_paths = numpy.full(
            (len(self.loop_specs), int(action_lengths.max())), -1, dtype=numpy.int32
        )
        action_coeff = numpy.empty(len(self.loop_specs), dtype=numpy.float64)
        action_field_index = numpy.empty(len(self.loop_specs), dtype=numpy.int32)
        for index, (steps, coefficient, field_index) in enumerate(self.loop_specs):
            encoded = _steps_to_quda_force(steps)
            action_paths[index, : len(encoded)] = encoded
            action_coeff[index] = coefficient
            action_field_index[index] = field_index

        terms = [
            [term for term in self.force_terms if term[0] == direction]
            for direction in range(4)
        ]
        force_num_paths = max(map(len, terms))
        force_max_length = max(
            len(staple)
            for direction_terms in terms
            for _, staple, _, _, _ in direction_terms
        )
        force_paths = numpy.full(
            (4, force_num_paths, force_max_length), -1, dtype=numpy.int32
        )
        force_lengths = numpy.zeros((4, force_num_paths), dtype=numpy.int32)
        force_coeff = numpy.zeros((4, force_num_paths), dtype=numpy.float64)
        force_field_index = numpy.full((4, force_num_paths), -1, dtype=numpy.int32)
        force_field_offset = numpy.zeros((4, force_num_paths, 4), dtype=numpy.int32)
        for direction, direction_terms in enumerate(terms):
            for index, (_, staple, offset_tzyx, coefficient, field_index) in enumerate(
                direction_terms
            ):
                displacement = [int(mu == direction) for mu in range(4)]
                for mu, sign in staple:
                    displacement[mu] += sign
                if displacement != [0, 0, 0, 0]:
                    raise ValueError(
                        f"force staple for direction {direction} is not closed: {staple}"
                    )

                encoded = _steps_to_quda_force(staple)
                force_paths[direction, index, : len(encoded)] = encoded
                force_lengths[direction, index] = len(encoded)
                force_coeff[direction, index] = -float(coefficient)
                force_field_index[direction, index] = field_index
                if field_index != _FIELD_CONSTANT:
                    force_field_offset[direction, index] = numpy.asarray(
                        offset_tzyx[::-1], dtype=numpy.int32
                    )

        local_dim = numpy.asarray(self.latt_info.size, dtype=numpy.int32)
        self._quda_context = createGaugeRotatingContextQuda(
            local_dim,
            self._scalar_radius,
            action_paths,
            action_lengths,
            action_coeff,
            action_field_index,
            force_paths,
            force_lengths,
            force_coeff,
            force_field_index,
            force_field_offset,
        )
        if not self._quda_context:
            raise RuntimeError("QUDA failed to create the rotating gauge context")

        # Registered after initQuda's atexit callback, so context storage is
        # released before QUDA shuts down even if HMC keeps an action cycle.
        self._quda_context_finalizer = finalize(
            self, destroyGaugeRotatingContextQuda, self._quda_context
        )
        return self._quda_context

    def close(self):
        """Release the device context; subsequent use recreates it lazily."""
        finalizer = self._quda_context_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()
            self._quda_context = 0

    def action(self) -> float:
        """Return the resident gauge field's globally reduced action energy."""
        if not self.rotation_enabled:
            # Ordinary GaugeAction omits gauge-independent constants, whereas
            # the rotating action follows CLGLib's absolute-energy convention.
            # Restoring the plaquette/rectangle constant keeps Omega -> 0
            # continuous without changing the force or HMC energy differences.
            volume = int(numpy.prod(self.latt_info.global_size))
            constant = self.beta * volume * (6.0 + 12.0 * self.c_rect)
            return self.base_action.action() + constant
        return computeGaugeRotatingActionQuda(self._ensure_quda_context())

    def force(self, dt: float, mom: Optional[LatticeMom] = None):
        """Add the rotating tree-improved gauge force.

        Args:
            dt: Molecular-dynamics force scale.
            mom: Optional explicit momentum destination; resident momentum is used when omitted.
        """
        if not self.rotation_enabled:
            self.base_action.force(dt, mom)
            return

        gauge_state = None
        if mom is not None:
            gauge_state = (
                self.gauge_param.use_resident_mom,
                self.gauge_param.make_resident_mom,
                self.gauge_param.return_result_mom,
            )
            # The caller owns the explicit force destination. Do not replace
            # the HMC momentum resident field while accumulating into it.
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1
        try:
            computeGaugeRotatingForceQuda(
                nullptr if mom is None else mom.data_ptrs,
                self._ensure_quda_context(),
                dt,
                self.gauge_param,
            )
        finally:
            if gauge_state is not None:
                (
                    self.gauge_param.use_resident_mom,
                    self.gauge_param.make_resident_mom,
                    self.gauge_param.return_result_mom,
                ) = gauge_state
