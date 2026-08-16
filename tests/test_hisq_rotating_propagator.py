"""Check full, even-odd, multishift, cached, and MG rotating HISQ solves.

The test passes when ten sites from each double-precision ``16^4`` solution
match the hardcoded offline CLGLib propagator within ``5e-14``; full and
even-odd solutions, cache-on and cache-off paths, multishift residuals, and
multigrid gauge reloads must satisfy their similarly tight assertions. The
same global references are gathered and checked for each listed MPI grid.

Run from the repository root. ``PYQUDA_TEST_GRID`` is ordered as x,y,z,t and
its product must equal the MPI rank count. Single GPU, one rank::

    CUDA_VISIBLE_DEVICES=0 python tests/test_hisq_rotating_propagator.py

Single GPU, two MPI ranks sharing that device::

    CUDA_VISIBLE_DEVICES=0 PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hisq_rotating_propagator.py

Two GPUs, one MPI rank per device::

    CUDA_VISIBLE_DEVICES=0,1 PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hisq_rotating_propagator.py

Repeat the two-rank test with grids ``1,2,1,1``, ``1,1,2,1``, and
``1,1,1,2`` for y, z, and t decomposition. Combined diagonal coverage uses
``2,1,1,2`` and ``1,2,1,2`` with four ranks, and ``2,2,1,2`` with eight
ranks. Fewer visible GPUs than ranks intentionally exercises the same grids
with local ranks sharing devices through MPS.
"""

from check_pyquda import test_dir

import os
import sys

sys.path.insert(0, os.path.dirname(test_dir))

import numpy  # noqa: E402
from mpi4py import MPI  # noqa: E402

from pyquda.action.abstract import RationalParam  # noqa: E402
from pyquda.action.hisq_rotating import HISQRotatingAction  # noqa: E402
from pyquda.dirac.hisq_rotating import HISQRotatingDirac  # noqa: E402
from pyquda.enum_quda import (  # noqa: E402
    QUDA_MAX_MG_LEVEL,
    QudaInverterType,
    QudaMatPCType,
    QudaPrecision,
    QudaSolutionType,
    QudaSolveType,
)
from pyquda.field import (  # noqa: E402
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
)
from pyquda.hmc import O2Nf1Ng0V  # noqa: E402
from pyquda.dirac.hisq import HISQDirac  # noqa: E402
from pyquda.hmc_rotating import RotatingHMC  # noqa: E402
from pyquda_utils import core  # noqa: E402


_SITES = [
    (0, 0, 0, 0),
    (15, 0, 0, 0),
    (0, 15, 0, 0),
    (0, 0, 15, 0),
    (0, 0, 0, 15),
    (1, 2, 3, 4),
    (5, 7, 9, 11),
    (8, 8, 8, 8),
    (14, 13, 12, 11),
    (3, 10, 6, 15),
]

# Complete three-color values from an offline double-precision CLGLib solve
# using the same gauge and source files.  CLGLib is not a test dependency.
_CLG_SOLUTION = numpy.asarray(
    [
        [
            0.18848010581084271 - 0.15034946892798054j,
            0.12764911807853099 - 0.024221460377494623j,
            0.05032275362602761 - 0.12687145174498993j,
        ],
        [
            -0.29632354812974909 + 0.065587760743907583j,
            0.46431239747043918 - 0.46487032822356433j,
            0.045304684270500682 - 0.30460582039687301j,
        ],
        [
            0.46629575272377843 - 0.059493267938674307j,
            0.16676743610911135 + 0.18963837189754162j,
            0.058032331475246737 - 0.057082555772502756j,
        ],
        [
            0.24999524836755485 - 0.52063017734769723j,
            -0.12452101079617402 + 0.1070924681548569j,
            0.001286831298684203 - 0.12820771840196207j,
        ],
        [
            -0.49651416293553918 - 0.20210898624154572j,
            0.34133689229561415 - 0.0022782761186126402j,
            0.12635486017754416 + 0.34084190864397601j,
        ],
        [
            -0.11888947965034873 - 0.10936612307649979j,
            0.078974415676665702 + 0.2044067961300455j,
            -0.10741742350506971 - 0.23456837816946915j,
        ],
        [
            -0.2239993442490949 + 0.35999480423690378j,
            -0.040184671580775877 - 0.25552519321276618j,
            0.061884758513678938 + 0.19154924985318605j,
        ],
        [
            0.44062886034252358 + 0.045977895890735372j,
            0.36135172844242724 + 0.10741097063482978j,
            0.030164759777169297 - 0.0039050141882574709j,
        ],
        [
            0.27436760337757654 - 0.485662250118224j,
            -0.1956678529751476 + 0.011230204872786368j,
            0.12974554086982856 - 0.06465462352461307j,
        ],
        [
            -0.02517070601947068 - 0.38476199011870577j,
            -0.41685839073966563 + 0.13874880213227983j,
            -0.42404697096478172 + 0.30821041075874184j,
        ],
    ],
    dtype=numpy.complex128,
)

_VALIDATED_MPI_GRIDS = {
    (2, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
    (2, 1, 1, 2),
    (1, 2, 1, 2),
    (2, 2, 1, 2),
}


def _test_grid():
    """Return and validate the requested MPI process grid."""
    mpi_size = MPI.COMM_WORLD.Get_size()
    encoded = os.environ.get("PYQUDA_TEST_GRID")
    grid = [mpi_size, 1, 1, 1] if encoded is None else [int(value) for value in encoded.split(",")]
    if len(grid) != 4 or int(numpy.prod(grid)) != mpi_size:
        raise ValueError(f"PYQUDA_TEST_GRID={encoded!r} does not match {mpi_size} MPI ranks")
    if mpi_size > 1 and tuple(grid) not in _VALIDATED_MPI_GRIDS:
        raise ValueError(f"unsupported rotating propagator MPI grid {grid}")
    return grid


def _selected_solution(latt, solution):
    """Gather the ten global-site solution matrices used by the CLGLib golden."""
    solution_lexico = solution.lexico()
    offsets = [coord * extent for coord, extent in zip(latt.grid_coord, latt.size)]
    local = {}
    for index, (x, y, z, t) in enumerate(_SITES):
        coordinates = [x, y, z, t]
        if all(offsets[d] <= coordinates[d] < offsets[d] + latt.size[d] for d in range(4)):
            local[index] = numpy.asarray(
                solution_lexico[
                    t - offsets[3],
                    z - offsets[2],
                    y - offsets[1],
                    x - offsets[0],
                ]
            ).copy()

    selected_by_index = {}
    for rank_values in latt.mpi_comm.allgather(local):
        selected_by_index.update(rank_values)
    if len(selected_by_index) != len(_SITES):
        raise RuntimeError(f"collected {len(selected_by_index)} of {len(_SITES)} golden sites")
    return numpy.asarray([selected_by_index[index] for index in range(len(_SITES))])


def _solve_multishift(latt, gauge, source):
    """Check that the rotating action reaches QUDA's multi-shift solver."""
    offsets = [0.02, 0.2, 1.0]
    action = HISQRotatingAction(
        latt,
        RationalParam(residue_force=[1.0] * len(offsets), offset_force=offsets),
        tol=1e-13,
        maxiter=10000,
        naik_epsilon=0.0,
        angular_velocity=0.2,
        mass=0.0,
    )
    action.phi = source
    hmc = RotatingHMC(gauge.latt_info, [action], O2Nf1Ng0V(1))
    hmc.initialize(97532, gauge)
    action.quark = MultiLatticeStaggeredFermion(latt, action.max_num_offset)
    action.updateFatLong()
    action.invertMultiShift("force")

    true_residuals = numpy.asarray(action.invert_param.true_res_offset[: len(offsets)])
    if latt.mpi_rank == 0:
        print(
            "multishift true residuals: "
            + ", ".join(f"{residual:.16e}" for residual in true_residuals)
        )
    assert numpy.max(true_residuals) < 5e-13


def _new_multigrid_dirac(latt):
    """Construct the rotation-aware fine-smoother multigrid configuration."""
    dirac = HISQRotatingDirac(
        latt,
        mass=0.11,
        tol=5e-16,
        maxiter=10000,
        angular_velocity=0.2,
        naik_epsilon=0.0,
        multigrid=[[8, 8, 4, 4]],
    )
    dirac.invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    dirac.multigrid.param.n_vec = [3] + [24] * (QUDA_MAX_MG_LEVEL - 1)
    dirac.multigrid.setParam(
        coarse_maxiter=8,
        setup_tol=1e-3,
        setup_maxiter=100,
        smoother_nu_post=2,
    )
    return dirac


def _check_multigrid_reload(latt, gauge, source):
    """Compare a refreshed multigrid hierarchy with a freshly built one."""
    reloaded_gauge = core.LatticeGauge(gauge.latt_info)
    reloaded_gauge.gauss(24681358, 0.2)

    reloaded_dirac = _new_multigrid_dirac(latt)
    try:
        reloaded_dirac.loadGauge(gauge)
        reloaded_dirac.loadGauge(reloaded_gauge, thin_update_only=True)
        reloaded_solution = reloaded_dirac.invert(source)
    finally:
        reloaded_dirac.freeGauge()

    fresh_dirac = _new_multigrid_dirac(latt)
    try:
        fresh_dirac.loadGauge(reloaded_gauge)
        fresh_solution = fresh_dirac.invert(source)
    finally:
        fresh_dirac.freeGauge()

    difference = (
        (reloaded_solution - fresh_solution).norm2() / fresh_solution.norm2()
    ) ** 0.5
    if latt.mpi_rank == 0:
        print(f"multigrid gauge-reload relative solution difference: {difference:.16e}")
    assert difference < 5e-14


def _check_near_zero_angular_velocity(latt, gauge, source):
    """Verify that sub-epsilon angular velocity uses the ordinary HISQ path."""
    rotating = HISQRotatingDirac(
        latt,
        mass=0.11,
        tol=1e-15,
        maxiter=10000,
        angular_velocity=numpy.finfo(numpy.float64).eps / 2,
    )
    ordinary = HISQDirac(latt, mass=0.11, tol=1e-15, maxiter=10000)
    double = QudaPrecision.QUDA_DOUBLE_PRECISION

    solutions = []
    for dirac in (rotating, ordinary):
        dirac.setPrecision(
            cuda=double,
            sloppy=double,
            refinement_sloppy=double,
            precondition=double,
            eigensolver=double,
        )
        dirac.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        dirac.invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
        dirac.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        dirac.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        with dirac.useGauge(gauge):
            solutions.append(dirac.invert(source))

    difference = (
        (solutions[0] - solutions[1]).norm2() / solutions[1].norm2()
    ) ** 0.5
    if latt.mpi_rank == 0:
        print(f"near-zero-angular-velocity/ordinary HISQ relative solution difference: {difference:.16e}")
    assert not rotating.rotation_enabled
    assert difference < 5e-14


def _solve(
    latt,
    gauge,
    source,
    even_odd: bool,
    multigrid: bool = False,
    cache_rotation_links: bool = True,
    export_path_links: bool = False,
):
    """Solve one operator variant and optionally export its rotating paths."""
    multigrid_blocks = [[8, 8, 4, 4]] if multigrid else None
    dirac = _new_multigrid_dirac(latt) if multigrid else HISQRotatingDirac(
        latt, mass=0.11, tol=1e-15, maxiter=10000, angular_velocity=0.2,
        naik_epsilon=0.0, multigrid=multigrid_blocks,
        cache_rotation_links=cache_rotation_links)
    dirac.invert_param.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
    if multigrid:
        # The coarse operator remains ordinary HISQ; setup and fine smoothing
        # use the complete rotating operator.
        assert dirac.multigrid.inv_param.angular_velocity == dirac.angular_velocity
        tag = "multigrid"
    elif even_odd:
        dirac.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        dirac.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        dirac.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        tag = "even-odd-cached" if cache_rotation_links else "even-odd-on-the-fly"
    else:
        dirac.invert_param.inv_type = QudaInverterType.QUDA_BICGSTAB_INVERTER
        dirac.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
        tag = "full-system"

    with dirac.useGauge(gauge):
        solution = dirac.invert(source)
        residual = dirac.mat(solution) - source
        path_links = dirac.getRotatingPathLinks() if export_path_links else None

    relative_residual = (residual.norm2() / source.norm2()) ** 0.5
    selected = _selected_solution(latt, solution)
    selected_error = numpy.max(numpy.abs(selected - _CLG_SOLUTION))
    if latt.mpi_rank == 0:
        print(f"{tag} full relative residual: {relative_residual:.16e}")
        print(f"{tag} CLGLib selected-site max abs error: {selected_error:.16e}")
    assert relative_residual < 5e-14
    numpy.testing.assert_allclose(selected, _CLG_SOLUTION, rtol=0.0, atol=5e-14)
    return solution, path_links


def _check_path_link_exports(cached, uncached):
    """Compare lazy xxt/yyt/xyt snapshots from cache-on and cache-off contexts."""
    for family in ("xxt", "yyt", "xyt"):
        cached_links = getattr(cached, family)
        uncached_links = getattr(uncached, family)
        assert len(cached_links) == len(uncached_links)
        for index, (cached_link, uncached_link) in enumerate(zip(cached_links, uncached_links)):
            cached_array = numpy.asarray(cached_link.lexico())
            uncached_array = numpy.asarray(uncached_link.lexico())
            error = numpy.max(numpy.abs(cached_array - uncached_array))
            if cached_link.latt_info.mpi_rank == 0:
                print(f"{family}[{index}] cached/on-the-fly max abs error: {error:.16e}")
            assert error < 5e-14


def main():
    """Run full, even-odd, cache, multi-shift, and fine-MG regressions."""
    grid = _test_grid()
    grid_suffix = "" if grid == [1, 1, 1, 1] else "-" + "x".join(map(str, grid))
    resource_path = os.path.join(test_dir, ".cache", "quda-rotating-propagator-golden" + grid_suffix)
    os.makedirs(resource_path, exist_ok=True)
    multi_rank = MPI.COMM_WORLD.Get_size() > 1
    core.init(
        grid,
        resource_path=resource_path,
        enable_mps=multi_rank,
        enable_p2p=0 if multi_rank else 3,
    )
    gauge_latt = core.LatticeInfo([16, 16, 16, 16], t_boundary=1, anisotropy=1.0)
    latt = core.LatticeInfo([16, 16, 16, 16], t_boundary=-1, anisotropy=1.0)
    gauge = core.LatticeGauge(gauge_latt)
    gauge.gauss(24681357, 0.2)

    rng = numpy.random.default_rng(97531)
    source_global = (
        rng.standard_normal((16, 16, 16, 16, 3))
        + 1j * rng.standard_normal((16, 16, 16, 16, 3))
    ) / numpy.sqrt(2.0)
    x0, y0, z0, t0 = [coord * extent for coord, extent in zip(latt.grid_coord, latt.size)]
    source_lexico = source_global[
        t0 : t0 + latt.Lt,
        z0 : z0 + latt.Lz,
        y0 : y0 + latt.Ly,
        x0 : x0 + latt.Lx,
    ]
    source = LatticeStaggeredFermion(latt, latt.evenodd(source_lexico, False, "numpy"))
    source.toDevice()

    full_solution, _ = _solve(latt, gauge, source, even_odd=False)
    even_odd_solution, cached_paths = _solve(
        latt, gauge, source, even_odd=True, export_path_links=True
    )
    uncached_solution, uncached_paths = _solve(
        latt, gauge, source, even_odd=True, cache_rotation_links=False,
        export_path_links=True,
    )
    _check_path_link_exports(cached_paths, uncached_paths)
    _check_near_zero_angular_velocity(latt, gauge, source)
    _solve_multishift(latt, gauge, source)
    multigrid_solution, _ = _solve(latt, gauge, source, even_odd=True, multigrid=True)
    _check_multigrid_reload(latt, gauge, source)
    even_odd_difference = (
        (full_solution - even_odd_solution).norm2() / even_odd_solution.norm2()
    ) ** 0.5
    multigrid_difference = (
        (full_solution - multigrid_solution).norm2() / multigrid_solution.norm2()
    ) ** 0.5
    uncached_difference = (
        (even_odd_solution - uncached_solution).norm2() / uncached_solution.norm2()
    ) ** 0.5
    if latt.mpi_rank == 0:
        print(f"full-system/even-odd relative solution difference: {even_odd_difference:.16e}")
        print(f"full-system/multigrid relative solution difference: {multigrid_difference:.16e}")
        print(f"cached/on-the-fly relative solution difference: {uncached_difference:.16e}")
    assert even_odd_difference < 5e-14
    assert multigrid_difference < 5e-14
    assert uncached_difference < 5e-14


if __name__ == "__main__":
    main()
