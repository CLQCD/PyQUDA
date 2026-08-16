"""Check combined rotating tree-improved gauge plus Nf=2+1 HISQ HMC.

``PYQUDA_HMC_TEST=small`` (the default) runs one deterministic complete
trajectory and compares ``h_diff``, ``u0``, and ``abs(Polyakov loop)`` with
per-grid references. Unpartitioned dimensions are four sites; partitioned
dimensions are twelve sites so each local extent remains suitable for the
HISQ three-link stencil. ``PYQUDA_HMC_TEST=full`` instead runs the finalized
global ``12^3 x 4`` identity-start test: 200 warmup trajectories followed by
300 production attempts, retaining every third post-Metropolis state. It
checks the Hamiltonian and acceptance distributions and compares ensemble
observables with independent CLGLib results.

Run from the repository root. ``PYQUDA_TEST_GRID`` is ordered as x,y,z,t and
its product must equal the MPI rank count. Single GPU, one rank::

    CUDA_VISIBLE_DEVICES=0 PYQUDA_HMC_TEST=small python tests/test_hmc_hisq_rotating.py
    CUDA_VISIBLE_DEVICES=0 PYQUDA_HMC_TEST=full python tests/test_hmc_hisq_rotating.py

Single GPU, two MPI ranks sharing that device::

    CUDA_VISIBLE_DEVICES=0 PYQUDA_HMC_TEST=small PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hmc_hisq_rotating.py
    CUDA_VISIBLE_DEVICES=0 PYQUDA_HMC_TEST=full PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hmc_hisq_rotating.py

Two GPUs, one MPI rank per device::

    CUDA_VISIBLE_DEVICES=0,1 PYQUDA_HMC_TEST=small PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hmc_hisq_rotating.py
    CUDA_VISIBLE_DEVICES=0,1 PYQUDA_HMC_TEST=full PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hmc_hisq_rotating.py

Repeat the two-rank test with grids ``1,2,1,1``, ``1,1,2,1``, and
``1,1,1,2`` for y, z, and t decomposition. Combined diagonal coverage uses
``2,1,1,2`` and ``1,2,1,2`` with four ranks, and ``2,2,1,2`` with eight
ranks. Fewer visible GPUs than ranks intentionally exercises the same grids
with local ranks sharing devices through MPS. The full mode can be much
slower when multiple ranks share one GPU.
"""

from math import exp, hypot, prod
import os
import sys
from time import perf_counter
from warnings import catch_warnings, simplefilter

from check_pyquda import test_dir

sys.path.insert(0, os.path.dirname(test_dir))

from mpi4py import MPI  # noqa: E402
from pyquda.action.abstract import RationalParam  # noqa: E402
from pyquda.action.gauge_rotating import TreeImprovedRotatingGaugeAction  # noqa: E402
from pyquda.action.hisq import HISQAction  # noqa: E402
from pyquda.action.hisq_rotating import HISQRotatingAction, MultiHISQRotatingAction  # noqa: E402
from pyquda.enum_quda import QudaVerbosity  # noqa: E402
from pyquda.hmc import HMC, O4Nf5Ng0V  # noqa: E402
from pyquda.hmc_rotating import RotatingHMC  # noqa: E402
from pyquda_utils import core  # noqa: E402


LATTICE = [12, 12, 12, 4]
ANGULAR_VELOCITY = 0.09
BETA = 7.3
TRAJECTORY_LENGTH = 2.0
WARMUP_TRAJECTORIES = 200
PRODUCTION_CONFIGURATIONS = 100
PRODUCTION_SKIP = 3
PRODUCTION_TRAJECTORIES = PRODUCTION_CONFIGURATIONS * PRODUCTION_SKIP
OUTER_STEPS = 8
INNER_STEPS = 1
SEED = 20260818

H_DIFF_TIGHT_CUT = 0.18
MIN_H_DIFF_BELOW_TIGHT_CUT = 285
H_DIFF_LOOSE_CUT = 0.3
MIN_H_DIFF_BELOW_LOOSE_CUT = 297
MIN_ACCEPTED_TRAJECTORIES = 286

# Independent CLGLib identity-start reference: 200 warmup trajectories, then
# 300 production attempts with Skip=3 and 100 retained configurations. Block
# SEM uses blocks of 10 retained configurations:
#
# observable    mean             block SEM
# u0            0.89484963       6.80e-5
# mean(abs(P))  0.7403923294     7.90e-3
CLG_MEAN_U0 = 0.89484963
CLG_MEAN_ABS_POLYAKOV = 0.7403923294
U0_TOL = 1e-4
ABS_POLYAKOV_TOL = 3e-2

SMALL_ANGULAR_VELOCITY = 0.1
SMALL_OUTER_STEPS = 2
SMALL_INNER_STEPS = 1
SMALL_CASES = {
    (1, 1, 1, 1): ([4, 4, 4, 4], -1.105390655148, 0.952114029196, 1.918925601062),
    (2, 1, 1, 1): ([12, 4, 4, 4], -1.651015565350e1, 0.955369085745, 1.828971962525),
    (1, 2, 1, 1): ([4, 12, 4, 4], -1.196108814320e1, 0.956555159804, 1.467035080512),
    (1, 1, 2, 1): ([4, 4, 12, 4], -1.480844164951e1, 0.955740649068, 1.793035040662),
    (1, 1, 1, 2): ([4, 4, 4, 12], -1.462094023846e1, 0.954874593058, 0.536208985201),
    (2, 1, 1, 2): ([12, 4, 4, 12], -3.298284094106e1, 0.954615264569, 0.562899111640),
    (1, 2, 1, 2): ([4, 12, 4, 12], -3.532007190335e1, 0.955083417275, 0.510117869852),
    (2, 2, 1, 2): ([12, 12, 4, 12], -1.918313865796e2, 0.956221111876, 0.610456278525),
}

LIGHT_MASS = 0.2
LIGHT_EPSILON = 0.0
LIGHT_MC = [
    5.739346341805533,
    -0.03419769943763456,
    -0.3193521020535132,
    -2.688803150811836,
    -27.216658544425936,
    -920.7511362574797,
    0.2623979195244254,
    0.9826039738340274,
    4.869608306515427,
    27.994567620185045,
    294.4355685338594,
]
LIGHT_MD = [
    0.02776289005513841,
    0.22927211190842767,
    0.3967959651684371,
    0.865918265080505,
    2.138865372006634,
    7.957659217978743,
    0.1890424552003735,
    0.5638844456519323,
    2.6256286026533258,
    14.235187993255236,
    105.21350312565073,
]

HEAVY_MASS = 0.5
HEAVY_EPSILON = -0.151482468311921
HEAVY_MC = [
    2.5231100953713903,
    -0.09216846063286101,
    -0.5043857059446171,
    -2.4936059422760617,
    -15.553974026643278,
    -299.30360454144,
    1.3312281904269978,
    3.330761197264888,
    11.451089542594072,
    48.816174734797514,
    390.7069268509098,
]
HEAVY_MD = [
    0.15556995570686036,
    0.21131352059790415,
    0.5653450314355664,
    1.5589535536758405,
    5.159098534639067,
    33.713826756845634,
    1.1902873472731925,
    2.712590944759212,
    8.945996641708259,
    36.39597573265889,
    236.63704175329474,
]


def _rational_param(mc, md):
    """Build the degree-five rational approximation used by production HMC."""
    return RationalParam(
        norm_force=md[0],
        residue_force=md[1:6],
        offset_force=md[6:11],
        norm_sample=mc[0],
        residue_sample=mc[1:6],
        offset_sample=mc[6:11],
        norm_action=md[0],
        residue_action=md[1:6],
        offset_action=md[6:11],
    )


LIGHT_RATIONAL_PARAM = _rational_param(LIGHT_MC, LIGHT_MD)
HEAVY_RATIONAL_PARAM = _rational_param(HEAVY_MC, HEAVY_MD)


def _as_complex(value):
    """Normalize a scalar or two-real observable to Python complex."""
    if isinstance(value, complex):
        return value
    return complex(value[0], value[1])


def _check_configuration_policy(
    gauge_latt, fermion_latt, gauge_action, rotating_action, angular_velocity
):
    """Check HMC geometry and mixed-action validation behavior."""
    ordinary = HISQAction(
        fermion_latt,
        LIGHT_RATIONAL_PARAM,
        tol=1e-10,
        maxiter=10000,
        naik_epsilon=0.0,
    )
    inconsistent = HISQRotatingAction(
        fermion_latt,
        LIGHT_RATIONAL_PARAM,
        tol=1e-10,
        maxiter=10000,
        naik_epsilon=0.0,
        angular_velocity=angular_velocity + 0.01,
        cache_rotation_links=False,
        mass=0.0,
    )
    try:
        RotatingHMC(
            gauge_latt,
            [gauge_action, ordinary, rotating_action],
            O4Nf5Ng0V(1),
        )
    except TypeError as error:
        assert "ordinary HISQ" in str(error)
    else:
        raise AssertionError("RotatingHMC accepted an ordinary HISQ action")

    with catch_warnings(record=True) as caught:
        simplefilter("always", RuntimeWarning)
        warning_hmc = RotatingHMC(
            gauge_latt,
            [gauge_action, rotating_action, inconsistent],
            O4Nf5Ng0V(1),
        )

    messages = [str(item.message) for item in caught]
    assert any("only compatible actions will be fused" in message for message in messages)
    assert any("different angular velocities" in message for message in messages)
    assert rotating_action in warning_hmc.fermion_monomials
    assert inconsistent in warning_hmc.fermion_monomials


def _trajectory(hmc: RotatingHMC, metropolis: bool):
    """Run one production-length trajectory and return its observables."""
    hmc.gaussMom()
    hmc.samplePhi()
    old_gauge = core.LatticeGauge(hmc.latt_info)
    hmc.saveGauge(old_gauge)

    h_old = hmc.momAction() + hmc.gaugeAction() + hmc.fermionAction()
    hmc.integrate(TRAJECTORY_LENGTH, 2e-15)
    h_new = hmc.momAction() + hmc.gaugeAction() + hmc.fermionAction()
    h_diff = h_new - h_old

    accepted = True
    if metropolis:
        accepted = hmc.accept(h_diff)
        if not accepted:
            hmc.loadGauge(old_gauge)

    u0 = hmc.plaquette()[0] ** 0.25
    polyakov = _as_complex(hmc.polyakovLoop())
    return h_diff, accepted, u0, polyakov


def _make_hmc(grid, lattice, angular_velocity, outer_steps, inner_steps, cache_tag):
    """Create an HMC instance for the requested lattice, parameters, and grid."""
    multi_rank = MPI.COMM_WORLD.Get_size() > 1
    grid_suffix = "" if grid == [1, 1, 1, 1] else "-" + "x".join(map(str, grid))
    resource_path = os.path.join(
        test_dir, ".cache", f"quda-rotating-hmc-{cache_tag}" + grid_suffix
    )
    os.makedirs(resource_path, exist_ok=True)
    core.init(
        grid,
        resource_path=resource_path,
        enable_mps=multi_rank,
        enable_p2p=0 if multi_rank else 3,
    )
    gauge_latt = core.LatticeInfo(lattice, t_boundary=1, anisotropy=1.0)
    fermion_latt = core.LatticeInfo(lattice, t_boundary=-1, anisotropy=1.0)
    gauge_action = TreeImprovedRotatingGaugeAction(
        gauge_latt, beta=BETA, angular_velocity=angular_velocity
    )
    light = HISQRotatingAction(
        fermion_latt,
        LIGHT_RATIONAL_PARAM,
        tol=1e-10,
        maxiter=10000,
        naik_epsilon=LIGHT_EPSILON,
        angular_velocity=angular_velocity,
        mass=0.0,
    )
    heavy = HISQRotatingAction(
        fermion_latt,
        HEAVY_RATIONAL_PARAM,
        tol=1e-10,
        maxiter=10000,
        naik_epsilon=HEAVY_EPSILON,
        angular_velocity=angular_velocity,
        mass=0.0,
    )
    _check_configuration_policy(
        gauge_latt, fermion_latt, gauge_action, light, angular_velocity
    )

    inner_hmc = HMC(gauge_latt, [gauge_action], O4Nf5Ng0V(inner_steps))
    hmc = RotatingHMC(
        gauge_latt, [light, heavy], O4Nf5Ng0V(outer_steps), hmc_inner=inner_hmc
    )
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
    hmc.initialize(SEED, core.LatticeGauge(gauge_latt))
    return gauge_action, light, heavy, inner_hmc, hmc


def _check_hmc_structure(gauge_action, light, heavy, inner_hmc, hmc):
    """Check that rotating actions were fused without changing gauge nesting."""
    assert hmc.gauge_monomials == []
    assert len(hmc.fermion_monomials) == 1
    assert isinstance(hmc.fermion_monomials[0], MultiHISQRotatingAction)
    assert hmc.fermion_monomials[0].pseudo_fermions == [light, heavy]
    assert inner_hmc.gauge_monomials == [gauge_action]


def main():
    """Run the selected quick or production HMC test on the MPI grid."""
    mode = os.environ.get("PYQUDA_HMC_TEST", "small").lower()
    if mode not in ("small", "full"):
        raise ValueError("PYQUDA_HMC_TEST must be 'small' or 'full'")

    encoded_grid = os.environ.get("PYQUDA_TEST_GRID")
    grid = (
        [1, 1, 1, 1]
        if encoded_grid is None
        else [int(value) for value in encoded_grid.split(",")]
    )
    if len(grid) != 4 or prod(grid) != MPI.COMM_WORLD.Get_size():
        raise ValueError(
            f"PYQUDA_TEST_GRID={encoded_grid!r} does not match "
            f"{MPI.COMM_WORLD.Get_size()} MPI size"
        )

    rank = MPI.COMM_WORLD.Get_rank()
    if mode == "small":
        case = SMALL_CASES.get(tuple(grid))
        if case is None:
            raise ValueError(f"unsupported small rotating HMC grid {grid}")
        lattice, h_diff_ref, u0_ref, abs_polyakov_ref = case
        gauge_action, light, heavy, inner_hmc, hmc = _make_hmc(
            grid,
            lattice,
            SMALL_ANGULAR_VELOCITY,
            SMALL_OUTER_STEPS,
            SMALL_INNER_STEPS,
            "small",
        )
        _check_hmc_structure(gauge_action, light, heavy, inner_hmc, hmc)
        h_diff, _, u0, polyakov = _trajectory(hmc, metropolis=False)
        passed = (
            abs(h_diff - h_diff_ref) < 1e-7
            and abs(u0 - u0_ref) < 1e-10
            and abs(abs(polyakov) - abs_polyakov_ref) < 1e-9
        )
        if rank == 0:
            print(
                f"small HMC MPI grid={tuple(grid)} h_diff={h_diff:+.12e} "
                f"u0={u0:.12f} abs_polyakov={abs(polyakov):.12f}",
                flush=True,
            )
        if not MPI.COMM_WORLD.bcast(passed, root=0):
            raise AssertionError(f"small rotating HMC failed for MPI grid {tuple(grid)}")
        return

    gauge_action, light, heavy, inner_hmc, hmc = _make_hmc(
        grid,
        LATTICE,
        ANGULAR_VELOCITY,
        OUTER_STEPS,
        INNER_STEPS,
        "full",
    )
    _check_hmc_structure(gauge_action, light, heavy, inner_hmc, hmc)

    for index in range(WARMUP_TRAJECTORIES):
        start = perf_counter()
        h_diff, _, u0, polyakov = _trajectory(hmc, metropolis=False)
        if rank == 0:
            print(
                f"warmup {index + 1:03d}/{WARMUP_TRAJECTORIES:03d} "
                f"h_diff={h_diff:+.8e} u0={u0:.9f} "
                f"abs_polyakov={abs(polyakov):.9f} elapsed={perf_counter() - start:.2f}s",
                flush=True,
            )

    production_h_diff = []
    production_u0 = []
    production_polyakov = []
    accepted = 0
    for index in range(PRODUCTION_TRAJECTORIES):
        start = perf_counter()
        h_diff, accept, u0, polyakov = _trajectory(hmc, metropolis=True)
        production_h_diff.append(h_diff)
        accepted += int(accept)

        keep = (index + 1) % PRODUCTION_SKIP == 0
        if keep:
            production_u0.append(u0)
            production_polyakov.append(polyakov)

        if rank == 0:
            print(
                f"production {index + 1:03d}/{PRODUCTION_TRAJECTORIES:03d} "
                f"h_diff={h_diff:+.8e} p_accept={exp(min(-h_diff, 0.0)):.6f} "
                f"accepted={accept} u0={u0:.9f} "
                f"polyakov={polyakov.real:+.9f}{polyakov.imag:+.9f}i "
                f"abs_polyakov={hypot(polyakov.real, polyakov.imag):.9f} "
                f"kept={keep} "
                f"elapsed={perf_counter() - start:.2f}s",
                flush=True,
            )

    max_abs_h_diff = max(abs(value) for value in production_h_diff)
    assert len(production_u0) == PRODUCTION_CONFIGURATIONS
    assert len(production_polyakov) == PRODUCTION_CONFIGURATIONS
    below_tight_cut = sum(abs(value) < H_DIFF_TIGHT_CUT for value in production_h_diff)
    below_loose_cut = sum(abs(value) < H_DIFF_LOOSE_CUT for value in production_h_diff)
    mean_u0 = sum(production_u0) / PRODUCTION_CONFIGURATIONS
    mean_polyakov = sum(production_polyakov) / PRODUCTION_CONFIGURATIONS
    mean_abs_polyakov = sum(abs(value) for value in production_polyakov) / PRODUCTION_CONFIGURATIONS
    passed = (
        accepted >= MIN_ACCEPTED_TRAJECTORIES
        and below_tight_cut >= MIN_H_DIFF_BELOW_TIGHT_CUT
        and below_loose_cut >= MIN_H_DIFF_BELOW_LOOSE_CUT
        and abs(mean_u0 - CLG_MEAN_U0) < U0_TOL
        and abs(mean_abs_polyakov - CLG_MEAN_ABS_POLYAKOV) < ABS_POLYAKOV_TOL
    )
    if rank == 0:
        print(
            f"MPI grid={tuple(grid)} production acceptance={accepted}/{PRODUCTION_TRAJECTORIES} "
            f"max_abs_h_diff={max_abs_h_diff:.8e} "
            f"below_{H_DIFF_TIGHT_CUT:.2f}={below_tight_cut}/{PRODUCTION_TRAJECTORIES} "
            f"below_{H_DIFF_LOOSE_CUT:.1f}={below_loose_cut}/{PRODUCTION_TRAJECTORIES} "
            f"mean_u0={mean_u0:.9f} "
            f"mean_polyakov={mean_polyakov.real:+.9f}{mean_polyakov.imag:+.9f}i "
            f"mean_abs_polyakov={mean_abs_polyakov:.9f}",
            flush=True,
        )
    if not MPI.COMM_WORLD.bcast(passed, root=0):
        raise AssertionError(f"rotating HMC production failed for MPI grid {tuple(grid)}")


if __name__ == "__main__":
    main()
