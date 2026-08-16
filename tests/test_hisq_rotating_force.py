"""Check rotating gauge energy/force and epsilon-corrected HISQ force.

The serial test passes when the absolute gauge energy matches the offline
CLGLib value within ``5e-11`` and ten complete gauge-force and HISQ-force link
matrices match hardcoded CLGLib values within ``1e-14``. It also checks the
zero-angular-velocity fast paths and rotating-context lifecycle. MPI cases
compare energy and force signatures with one-rank references within their
declared ``2e-11``/``2e-13`` tolerances.

Run from the repository root. ``PYQUDA_TEST_GRID`` is ordered as x,y,z,t and
its product must equal the MPI rank count. Single GPU, one rank::

    CUDA_VISIBLE_DEVICES=0 python tests/test_hisq_rotating_force.py

Single GPU, two MPI ranks sharing that device::

    CUDA_VISIBLE_DEVICES=0 PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hisq_rotating_force.py

Two GPUs, one MPI rank per device::

    CUDA_VISIBLE_DEVICES=0,1 PYQUDA_TEST_GRID=2,1,1,1 mpiexec --bind-to none -n 2 python tests/test_hisq_rotating_force.py

Repeat the two-rank test with grids ``1,2,1,1``, ``1,1,2,1``, and
``1,1,1,2`` for y, z, and t decomposition. Combined diagonal coverage uses
``2,1,1,2`` and ``1,2,1,2`` with four ranks, and ``2,2,1,2`` with eight
ranks. Fewer visible GPUs than ranks intentionally exercises the same grids
with local ranks sharing devices through MPS.
"""

from unittest.mock import patch

from check_pyquda import test_dir

import os
import sys

sys.path.insert(0, os.path.dirname(test_dir))

import numpy  # noqa: E402
from mpi4py import MPI  # noqa: E402

from pyquda.action.abstract import RationalParam  # noqa: E402
from pyquda.action.gauge_rotating import TreeImprovedRotatingGaugeAction  # noqa: E402
from pyquda.action.hisq_rotating import HISQRotatingAction  # noqa: E402
from pyquda.field import (  # noqa: E402
    LatticeMom,
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
)
from pyquda.hmc import O2Nf1Ng0V  # noqa: E402
from pyquda.hmc_rotating import RotatingHMC  # noqa: E402
from pyquda_utils import core  # noqa: E402


_MPI_CASES = {
    (2, 1, 1, 1): (
        [12, 4, 4, 4],
        5815.5889234942979,
        [347.0965975022321, 228.23969285519038, 113.47884536141405,
         371.4422889327106, 5.1880467992241677],
        [2.7351717294999536, 0.99593261802114208, -2.3666815956355487,
         -0.8328835137101529, 4.1303579636095664],
    ),
    (1, 2, 1, 1): (
        [4, 12, 4, 4],
        5865.3274724257753,
        [349.40563218805244, 157.66430855709658, -40.30563385881757,
         427.3537499861526, -36.91841378742325],
        [2.8298822985798187, 0.7663682049238763, -1.0535600776090486,
         0.2716714740827939, 3.755038876146239],
    ),
    (1, 1, 1, 2): (
        [4, 4, 4, 12],
        5594.3828272625051,
        [331.69187692688575, -10.914019855017031, 388.7452875115338,
         396.66288225207404, -5.106543359038511],
        [3.3395254392528906, 3.8724301874402762, -2.451397896866448,
         0.9433196630151897, 1.5564237334752788],
    ),
    (2, 1, 1, 2): (
        [12, 4, 4, 12],
        17560.861207941056,
        [603.1592963819601, -252.81169144328996, -81.40217178213508,
         342.79169318388267, 499.1021993910382],
        [5.343944131260036, 1.3355091905681749, 5.809277843108058,
         9.63159160008422, 5.303359158431304],
    ),
    (1, 1, 2, 1): (
        [4, 4, 12, 4],
        5578.7737610685854,
        [331.1639752743967, 7.101740405545598, 74.20419389960537,
         212.30769167553356, -51.97654357474745],
        [2.78564936102055, 0.8010308283605507, 1.7272447323945292,
         3.5278761491491264, 0.9237694709062955],
    ),
    (1, 2, 1, 2): (
        [4, 12, 4, 12],
        17623.201895419184,
        [604.4305317708535, -776.9473490756739, 4.934439869905603,
         -104.1476483456376, 508.7893113127949],
        [5.624500123820403, -1.4320289410232134, 1.863516984781485,
         10.340714428279927, 9.391205657625447],
    ),
    (2, 2, 1, 2): (
        [12, 12, 4, 12],
        54730.003828340406,
        [1090.4028989050198, -1002.9541615867228, 63.45225750636442,
         262.7245255204228, -159.4063384894883],
        [8.915828319576937, 2.9457141732515617, -2.7977441337928157,
         6.246505917405027, 3.86727883069464],
    ),
}
_CLG_GAUGE_ACTION = 1856.5969359823
_CLG_ZERO_GAUGE_ACTION = 1840.2635338586
_LATTICE = [4, 4, 4, 4]
_BETA = 7.3
_ANGULAR_VELOCITY = 0.13
_EPSILON = -0.151482468311921
_GAUGE_SEED = 24681357
_GAUGE_WIDTH = 0.2
_FERMION_SEED = 86420975
_GAUGE_MOMENTUM_FACTOR = 2.0

# Degree-5 MD approximation to (x + 4 * 0.5^2)^(-1/4).  The physical
# mass is absorbed into the offsets, so the rotating Dirac mass is zero.
_MD = [
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

# Deterministic sample without replacement from all (mu, x, y, z, t) links.
_LINKS = [
    (1, 3, 2, 2, 2),
    (3, 0, 0, 0, 0),
    (3, 2, 2, 2, 0),
    (1, 1, 3, 2, 2),
    (2, 0, 2, 1, 0),
    (1, 2, 3, 1, 2),
    (2, 2, 3, 3, 1),
    (2, 2, 3, 3, 0),
    (0, 1, 2, 0, 1),
    (0, 0, 1, 3, 0),
]

# Complete closed thin-link matrices from an offline double-precision CLGLib
# CActionFermionHISQCombined calculation on the identical gauge and even
# pseudofermion.  CLGLib is not a runtime test dependency.
_CLG_FORCE = numpy.asarray(
    [
        [
            [0.0029415412268955195j, -0.003870571587488974 + 0.012279580154974885j,
             -0.00048678105393306715 + 0.002496531566381072j],
            [0.003870571587488974 + 0.012279580154974885j, -0.00010995121310410235j,
             -0.010624857745664705 + 0.009092325156481503j],
            [0.00048678105393306715 + 0.002496531566381072j,
             0.010624857745664705 + 0.009092325156481503j, -0.002831590013791417j],
        ],
        [
            [-0.007177697210976042j, 0.003450129844036128 - 0.00834034103787875j,
             -0.004136500362891156 - 0.016659300141404692j],
            [-0.003450129844036128 - 0.00834034103787875j, 0.0011541447865065866j,
             -0.006293355626813482 - 0.0019769810890913554j],
            [0.004136500362891156 - 0.016659300141404692j,
             0.006293355626813482 - 0.0019769810890913554j, 0.006023552424469456j],
        ],
        [
            [0.014913804464644647j, 0.0056616015266667315 + 0.0001615305955068977j,
             -0.01276437181672509 + 0.007767388053273362j],
            [-0.0056616015266667315 + 0.0001615305955068977j, 0.008669455485771204j,
             0.011127449380659717 - 0.0008350020864597561j],
            [0.01276437181672509 + 0.007767388053273362j,
             -0.011127449380659717 - 0.0008350020864597561j, -0.023583259950415852j],
        ],
        [
            [-0.0016498909001551824j, 0.0024232707692559276 + 0.0038852931102910356j,
             -0.0012048554003100949 + 0.00589742507017257j],
            [-0.0024232707692559276 + 0.0038852931102910356j, -0.0032334725153278726j,
             0.004648887356864223 + 0.00146981702947149j],
            [0.0012048554003100949 + 0.00589742507017257j,
             -0.004648887356864223 + 0.00146981702947149j, 0.004883363415483056j],
        ],
        [
            [-0.007354396594927288j, -0.005064943860532596 + 0.0050312854055965285j,
             0.012689431507253162 - 0.013485575899497376j],
            [0.005064943860532596 + 0.0050312854055965285j, -0.010623071754123477j,
             0.0006334870949638151 - 0.0025458930429117206j],
            [-0.012689431507253162 - 0.013485575899497376j,
             -0.0006334870949638151 - 0.0025458930429117206j, 0.017977468349050764j],
        ],
        [
            [0.002492160414608109j, 0.010436920806995122 + 0.003476386686119418j,
             -0.004977514449572569 - 0.00320362362726137j],
            [-0.010436920806995122 + 0.003476386686119418j, -0.002220653307546649j,
             -0.000737881464730097 + 0.002948868124172266j],
            [0.004977514449572569 - 0.00320362362726137j,
             0.000737881464730097 + 0.002948868124172266j, -0.00027150710706145935j],
        ],
        [
            [0.021094771101925237j, 0.0001670056848164887 + 0.0000846753996512897j,
             0.003984794824586853 - 0.0027933215058197096j],
            [-0.0001670056848164887 + 0.0000846753996512897j, -0.0036886332443101313j,
             -0.004965886546755215 - 0.0020393245486199137j],
            [-0.003984794824586853 - 0.0027933215058197096j,
             0.004965886546755215 - 0.0020393245486199137j, -0.017406137857615106j],
        ],
        [
            [-0.016553993651965738j, -0.007385184388227346 - 0.0027743687533916665j,
             -0.006146885648747326 + 0.006805243985123568j],
            [0.007385184388227346 - 0.0027743687533916665j, 0.008041557517699867j,
             -0.009210299661911 + 0.003083442738190236j],
            [0.006146885648747326 + 0.006805243985123568j,
             0.009210299661911 + 0.003083442738190236j, 0.008512436134265873j],
        ],
        [
            [-0.0008397954117404655j, 0.0018841781634227513 - 0.00017020737134274986j,
             -0.011776294286257753 - 0.001432583841047063j],
            [-0.0018841781634227513 - 0.00017020737134274986j, -0.020597556171819617j,
             -0.0029751622278737003 + 0.0010407090064107434j],
            [0.011776294286257753 - 0.001432583841047063j,
             0.0029751622278737003 + 0.0010407090064107434j, 0.021437351583560083j],
        ],
        [
            [-0.007891210480549984j, -0.0047169151408516545 - 0.00448134542630072j,
             -0.0012689439872915148 + 0.01286609787249536j],
            [0.0047169151408516545 - 0.00448134542630072j, 0.010201595779625672j,
             0.005340597037062991 + 0.003964031401228444j],
            [0.0012689439872915148 + 0.01286609787249536j,
             -0.005340597037062991 + 0.003964031401228444j, -0.002310385299075689j],
        ],
    ],
    dtype=numpy.complex128,
)

# Complete closed thin-link matrices from the offline CLGLib production
# CActionGaugePlaquetteRotating force on the same random tree-improved gauge.
_CLG_GAUGE_FORCE = numpy.asarray(
    [
        [
            [0.694105987242928j, -1.3553345832891193 + 0.2724316248846048j,
             1.2960460959568758 + 1.7070197200347155j],
            [1.3553345832891193 + 0.2724316248846048j, -1.3270631361583067j,
             0.4998130321938723 - 1.4393610911526287j],
            [-1.2960460959568758 + 1.7070197200347155j,
             -0.4998130321938723 - 1.4393610911526287j, 0.6329571489153786j],
        ],
        [
            [0.5023595471713113j, -1.2973409102917566 + 0.9679724704703901j,
             1.0312623927437417 - 1.0924824115036904j],
            [1.2973409102917566 + 0.9679724704703901j, -1.6928940759057145j,
             0.3167515123921979 - 1.179368740542082j],
            [-1.0312623927437417 - 1.0924824115036904j,
             -0.3167515123921979 - 1.179368740542082j, 1.1905345287344031j],
        ],
        [
            [-1.6127948780504644j, 1.2621917820150008 + 0.8335818741369574j,
             -0.45169987039142045 + 0.06025328763729419j],
            [-1.2621917820150008 + 0.8335818741369574j, 0.21441049508844795j,
             0.5870191354090661 - 0.48699581664006375j],
            [0.45169987039142045 + 0.06025328763729419j,
             -0.5870191354090661 - 0.48699581664006375j, 1.3983843829620166j],
        ],
        [
            [0.14185781988796123j, 1.05867373248884 - 0.055994417218119596j,
             -0.9478522183269789 - 0.1482986242390381j],
            [-1.05867373248884 - 0.055994417218119596j, -0.539844363944072j,
             -0.9179453143702558 + 0.942976893019484j],
            [0.9478522183269789 - 0.1482986242390381j,
             0.9179453143702558 + 0.942976893019484j, 0.3979865440561107j],
        ],
        [
            [-0.8546885500827365j, 0.20165677248592975 + 0.64037216511288j,
             0.587055522460233 - 0.297119403435815j],
            [-0.20165677248592975 + 0.64037216511288j, 0.8929078341035697j,
             -1.933320039831549 - 0.38522150616860973j],
            [-0.587055522460233 - 0.297119403435815j,
             1.933320039831549 - 0.38522150616860973j, -0.03821928402083323j],
        ],
        [
            [1.092858985698569j, 2.0821364160482636 + 1.2063852776991775j,
             1.0888995114550952 - 0.1415670083466824j],
            [-2.0821364160482636 + 1.2063852776991775j, -1.9327020193618942j,
             0.7117392960968147 - 2.440452755212268j],
            [-1.0888995114550952 - 0.1415670083466824j,
             -0.7117392960968147 - 2.440452755212268j, 0.8398430336633252j],
        ],
        [
            [-1.9179255113491076j, -0.23619534877985965 - 1.4649339901141247j,
             0.08024794173383627 - 0.5234410768671987j],
            [0.23619534877985965 - 1.4649339901141247j, 2.0106344531781803j,
             -0.811258267622601 - 0.25328217499570715j],
            [-0.08024794173383627 - 0.5234410768671987j,
             0.811258267622601 - 0.25328217499570715j, -0.0927089418290728j],
        ],
        [
            [0.1274942166183937j, 0.1755222253620286 - 0.9896713057567759j,
             -0.968771924814995 + 0.0022991484778367016j],
            [-0.1755222253620286 - 0.9896713057567759j, 0.11927371076077674j,
             -0.36295484364233843 - 1.2890088445221486j],
            [0.968771924814995 + 0.0022991484778367016j,
             0.36295484364233843 - 1.2890088445221486j, -0.24676792737917044j],
        ],
        [
            [-0.28210280426584877j, -0.6495317401203835 + 2.4647878570868427j,
             1.2397378482539434 - 0.1273066103135242j],
            [0.6495317401203835 + 2.4647878570868427j, 0.23250394451735595j,
             -0.9558707681978389 + 0.26449840044538986j],
            [-1.2397378482539434 - 0.1273066103135242j,
             0.9558707681978389 + 0.26449840044538986j, 0.04959885974849282j],
        ],
        [
            [-0.07745725770166545j, 0.910723592183432 - 0.2791542121690213j,
             -0.6975923150412917 + 0.8559874771829172j],
            [-0.910723592183432 - 0.2791542121690213j, 0.4629309176809081j,
             -1.7521334237067991 + 0.6320394920372869j],
            [0.6975923150412917 + 0.8559874771829172j,
             1.7521334237067991 + 0.6320394920372869j, -0.38547365997924266j],
        ],
    ],
    dtype=numpy.complex128,
)


def _decode_momentum(momentum: numpy.ndarray) -> numpy.ndarray:
    """Expand QUDA's ten-real anti-Hermitian momentum representation."""
    matrix = numpy.zeros(momentum.shape[:-1] + (3, 3), dtype=numpy.complex128)
    matrix[..., 0, 0] = 1j * momentum[..., 6]
    matrix[..., 0, 1] = momentum[..., 0] + 1j * momentum[..., 1]
    matrix[..., 0, 2] = momentum[..., 2] + 1j * momentum[..., 3]
    matrix[..., 1, 0] = -momentum[..., 0] + 1j * momentum[..., 1]
    matrix[..., 1, 1] = 1j * momentum[..., 7]
    matrix[..., 1, 2] = momentum[..., 4] + 1j * momentum[..., 5]
    matrix[..., 2, 0] = -momentum[..., 2] + 1j * momentum[..., 3]
    matrix[..., 2, 1] = -momentum[..., 4] + 1j * momentum[..., 5]
    matrix[..., 2, 2] = 1j * momentum[..., 8]
    return matrix


def _assert_invalid_rotation_configs(gauge_latt, fermion_latt, rational):
    """Check the supported torus and finite-velocity policy."""
    try:
        TreeImprovedRotatingGaugeAction(gauge_latt, _BETA, numpy.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite gauge angular velocity was accepted")

    try:
        TreeImprovedRotatingGaugeAction(fermion_latt, _BETA, _ANGULAR_VELOCITY)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("non-torus gauge boundary was accepted")

    try:
        HISQRotatingAction(
            fermion_latt,
            rational,
            tol=1e-13,
            maxiter=10000,
            naik_epsilon=0.0,
            angular_velocity=numpy.inf,
            mass=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite fermion angular velocity was accepted")

def _test_grid():
    """Return and validate the requested MPI process grid."""
    size = MPI.COMM_WORLD.Get_size()
    encoded = os.environ.get("PYQUDA_TEST_GRID", "1,1,1,1")
    grid = [int(value) for value in encoded.split(",")]
    if len(grid) != 4 or numpy.prod(grid) != size:
        raise ValueError(f"PYQUDA_TEST_GRID={encoded!r} does not match {size} MPI ranks")
    return grid


def _gather_momentum(latt, momentum, global_lattice):
    """Gather one local momentum field into global t/z/y/x order."""
    local = latt.lexico(momentum.getHost(), True)
    pieces = latt.mpi_comm.gather((latt.grid_coord, local), root=0)
    if latt.mpi_rank != 0:
        return None

    lx, ly, lz, lt = global_lattice
    result = numpy.empty((4, lt, lz, ly, lx, 10), dtype=local.dtype)
    for rank_coord, rank_force in pieces:
        x0, y0, z0, t0 = [
            rank_coord[d] * rank_force.shape[4 - d] for d in range(4)
        ]
        local_lt, local_lz, local_ly, local_lx = rank_force.shape[1:5]
        result[
            :,
            t0 : t0 + local_lt,
            z0 : z0 + local_lz,
            y0 : y0 + local_ly,
            x0 : x0 + local_lx,
        ] = rank_force
    return result


def _force_signature(force, seed):
    """Return deterministic global force projections for MPI regression."""
    flat = force.ravel()
    rng = numpy.random.default_rng(seed)
    signature = [numpy.linalg.norm(flat)]
    for _ in range(4):
        signs = rng.choice(numpy.asarray([-1.0, 1.0]), size=flat.size)
        signature.append(numpy.dot(signs, flat))
    return numpy.asarray(signature)


def _mpi_main(grid):
    """Run the domain-decomposed rotating gauge and HISQ force checks."""
    case = _MPI_CASES.get(tuple(grid))
    if case is None:
        raise ValueError(f"unsupported rotating force MPI grid {grid}")
    lattice, gauge_action_ref, gauge_force_ref, hisq_force_ref = case
    suffix = "-" + "x".join(map(str, grid))
    resource_path = os.path.join(test_dir, ".cache", "quda-rotating-force-mpi" + suffix)
    os.makedirs(resource_path, exist_ok=True)
    core.init(
        grid,
        resource_path=resource_path,
        enable_mps=True,
        enable_p2p=0,
    )
    gauge_latt = core.LatticeInfo(lattice, t_boundary=1, anisotropy=1.0)
    fermion_latt = core.LatticeInfo(lattice, t_boundary=-1, anisotropy=1.0)
    gauge = core.LatticeGauge(gauge_latt)
    gauge.gauss(_GAUGE_SEED, _GAUGE_WIDTH)

    lx, ly, lz, lt = lattice
    rng = numpy.random.default_rng(_FERMION_SEED)
    phi_global = (
        rng.standard_normal((lt, lz, ly, lx, 3))
        + 1j * rng.standard_normal((lt, lz, ly, lx, 3))
    ) / numpy.sqrt(2.0)
    coordinates = numpy.indices((lt, lz, ly, lx))
    phi_global[numpy.sum(coordinates, axis=0) % 2 == 1] = 0.0
    x0, y0, z0, t0 = [
        coord * extent
        for coord, extent in zip(fermion_latt.grid_coord, fermion_latt.size)
    ]
    phi_local = phi_global[
        t0 : t0 + fermion_latt.Lt,
        z0 : z0 + fermion_latt.Lz,
        y0 : y0 + fermion_latt.Ly,
        x0 : x0 + fermion_latt.Lx,
    ]
    phi = LatticeStaggeredFermion(
        fermion_latt, fermion_latt.evenodd(phi_local, False, "numpy")
    )
    phi.toDevice()

    rational = RationalParam(
        norm_force=_MD[0],
        residue_force=_MD[1:6],
        offset_force=_MD[6:11],
        norm_sample=0.0,
        residue_sample=[1.0],
        offset_sample=[1.0],
        norm_action=0.0,
        residue_action=[1.0],
        offset_action=[1.0],
    )
    gauge_action = TreeImprovedRotatingGaugeAction(
        gauge_latt, _BETA, _ANGULAR_VELOCITY
    )
    action = HISQRotatingAction(
        fermion_latt,
        rational,
        tol=1e-13,
        maxiter=10000,
        naik_epsilon=_EPSILON,
        angular_velocity=_ANGULAR_VELOCITY,
        mass=0.0,
    )
    hmc = RotatingHMC(gauge_latt, [gauge_action, action], O2Nf1Ng0V(1))
    hmc.initialize(20260803, gauge)
    action.phi = phi
    action.quark = MultiLatticeStaggeredFermion(
        fermion_latt, action.max_num_offset
    )

    gauge_action_value = gauge_action.action()
    gauge_momentum = LatticeMom(gauge_latt)
    gauge_action.force(1.0, gauge_momentum)
    gauge_force = _gather_momentum(gauge_latt, gauge_momentum, lattice)

    momentum = LatticeMom(gauge_latt)
    action.force(1.0, momentum)
    hisq_force = _gather_momentum(gauge_latt, momentum, lattice)
    if gauge_latt.mpi_rank == 0:
        gauge_signature = _force_signature(gauge_force, 20260806)
        hisq_signature = _force_signature(hisq_force, 20260807)
        print(f"MPI_CASE {tuple(grid)} gauge_action={gauge_action_value:.17g}")
        print(f"MPI_CASE {tuple(grid)} gauge_force={gauge_signature.tolist()}")
        print(f"MPI_CASE {tuple(grid)} hisq_force={hisq_signature.tolist()}")
        numpy.testing.assert_allclose(gauge_action_value, gauge_action_ref, rtol=0.0, atol=2e-11)
        numpy.testing.assert_allclose(
            gauge_signature, numpy.asarray(gauge_force_ref), rtol=2e-13, atol=2e-13
        )
        numpy.testing.assert_allclose(
            hisq_signature, numpy.asarray(hisq_force_ref), rtol=2e-13, atol=2e-13
        )


def main():
    """Run the serial CLGLib golden and policy regressions."""
    grid = _test_grid()
    if MPI.COMM_WORLD.Get_size() > 1:
        _mpi_main(grid)
        return

    resource_path = os.path.join(test_dir, ".cache", "quda-rotating-force-golden")
    os.makedirs(resource_path, exist_ok=True)
    core.init(grid, resource_path=resource_path)
    gauge_latt = core.LatticeInfo(_LATTICE, t_boundary=1, anisotropy=1.0)
    fermion_latt = core.LatticeInfo(_LATTICE, t_boundary=-1, anisotropy=1.0)

    gauge = core.LatticeGauge(gauge_latt)
    gauge.gauss(_GAUGE_SEED, _GAUGE_WIDTH)

    rng = numpy.random.default_rng(_FERMION_SEED)
    phi_lexico = (
        rng.standard_normal((4, 4, 4, 4, 3))
        + 1j * rng.standard_normal((4, 4, 4, 4, 3))
    ) / numpy.sqrt(2.0)
    coordinates = numpy.indices((4, 4, 4, 4))
    phi_lexico[numpy.sum(coordinates, axis=0) % 2 == 1] = 0.0
    phi = LatticeStaggeredFermion(
        fermion_latt, fermion_latt.evenodd(phi_lexico, False, "numpy")
    )
    phi.toDevice()

    rational = RationalParam(
        norm_force=_MD[0],
        residue_force=_MD[1:6],
        offset_force=_MD[6:11],
        norm_sample=0.0,
        residue_sample=[1.0],
        offset_sample=[1.0],
        norm_action=0.0,
        residue_action=[1.0],
        offset_action=[1.0],
    )
    _assert_invalid_rotation_configs(gauge_latt, fermion_latt, rational)
    gauge_action = TreeImprovedRotatingGaugeAction(gauge_latt, _BETA, _ANGULAR_VELOCITY)
    action = HISQRotatingAction(
        fermion_latt,
        rational,
        tol=1e-13,
        maxiter=10000,
        naik_epsilon=_EPSILON,
        angular_velocity=_ANGULAR_VELOCITY,
        mass=0.0,
    )
    hmc = RotatingHMC(gauge_latt, [gauge_action, action], O2Nf1Ng0V(1))
    hmc.initialize(20260803, gauge)
    action.phi = phi
    action.quark = MultiLatticeStaggeredFermion(
        fermion_latt, action.max_num_offset
    )

    # Force calls temporarily override momentum residency.  Verify that all
    # incoming flags, including make_resident_mom, are restored even when the
    # QUDA entry point returns without producing a field.
    probe_momentum = LatticeMom(gauge_latt)
    incoming_state = (1, 1, 0)
    gauge_action.gauge_param.use_resident_mom = incoming_state[0]
    gauge_action.gauge_param.make_resident_mom = incoming_state[1]
    gauge_action.gauge_param.return_result_mom = incoming_state[2]
    with patch("pyquda.action.gauge_rotating.computeGaugeRotatingForceQuda") as force_call:
        gauge_action.force(1.0, probe_momentum)
    assert force_call.call_count == 1
    assert (
        gauge_action.gauge_param.use_resident_mom,
        gauge_action.gauge_param.make_resident_mom,
        gauge_action.gauge_param.return_result_mom,
    ) == incoming_state
    gauge_action.gauge_param.use_resident_mom = 0
    gauge_action.gauge_param.make_resident_mom = 0
    gauge_action.gauge_param.return_result_mom = 0

    # Sub-epsilon angular velocity must stay on the ordinary QUDA path and
    # must not allocate the rotation-specific coordinate/path context.
    # Materialize QUDA's lazy extended resident field first; the upstream
    # ordinary GaugeAction assumes a preceding gauge observable has done so.
    hmc.plaquette()
    near_zero = numpy.finfo(numpy.float64).eps / 2
    zero_action = TreeImprovedRotatingGaugeAction(gauge_latt, _BETA, near_zero)
    assert zero_action._quda_context == 0
    base_energy = zero_action.base_action.action()
    base_constant = _BETA * numpy.prod(_LATTICE) * (6.0 + 12.0 * zero_action.c_rect)
    numpy.testing.assert_allclose(
        zero_action.action(), base_energy + base_constant, rtol=0.0, atol=1e-13
    )
    numpy.testing.assert_allclose(
        zero_action.action(), _CLG_ZERO_GAUGE_ACTION, rtol=0.0, atol=5e-11
    )
    zero_momentum = LatticeMom(gauge_latt)
    base_momentum = LatticeMom(gauge_latt)
    zero_action.force(1.0, zero_momentum)
    zero_action.base_action.force(1.0, base_momentum)
    numpy.testing.assert_allclose(
        zero_momentum.getHost(),
        base_momentum.getHost(),
        rtol=0.0,
        atol=1e-14,
    )
    assert zero_action._quda_context == 0

    # Sub-epsilon angular velocity must disable the rotating fat-link cache
    # and dispatch force evaluation through the ordinary HISQ entry point.
    zero_hisq = HISQRotatingAction(
        gauge_latt,
        rational,
        tol=1e-13,
        maxiter=10000,
        naik_epsilon=_EPSILON,
        angular_velocity=near_zero,
        mass=0.0,
    )
    zero_phi = LatticeStaggeredFermion(
        gauge_latt, gauge_latt.evenodd(phi_lexico, False, "numpy")
    )
    zero_phi.toDevice()
    zero_hisq.phi = zero_phi
    zero_hisq.quark = MultiLatticeStaggeredFermion(
        gauge_latt, zero_hisq.max_num_offset
    )
    assert not zero_hisq.dirac.rotation_enabled
    with patch(
        "pyquda.dirac.hisq_rotating.loadRotatingXGaugeQuda"
    ) as load_rotating_links:
        zero_hisq.dirac._loadRotatingLinks(None)
    load_rotating_links.assert_not_called()

    with patch(
        "pyquda.action.hisq_rotating.computeHISQForceQuda"
    ) as ordinary_force, patch(
        "pyquda.action.hisq_rotating.computeHISQRotatingForceQuda"
    ) as rotating_force:
        zero_hisq.force(1.0)
    assert ordinary_force.call_count == 1
    rotating_force.assert_not_called()

    # Context destruction is idempotent and a closed action can build a fresh
    # device context after the QUDA resident state has been reused.
    action_before_close = gauge_action.action()
    assert gauge_action._quda_context
    gauge_action.close()
    gauge_action.close()
    assert gauge_action._quda_context == 0
    action_after_rebuild = gauge_action.action()
    assert gauge_action._quda_context
    numpy.testing.assert_allclose(action_after_rebuild, action_before_close, rtol=0.0, atol=1e-13)

    # Offline CLGLib production result on this exact seeded gauge.  The
    # runtime test intentionally has no CLGLib or reference-file dependency.
    numpy.testing.assert_allclose(
        gauge_action.action(),
        _CLG_GAUGE_ACTION,
        rtol=0.0,
        atol=5e-11,
    )

    gauge_momentum = LatticeMom(gauge_latt)
    gauge_action.force(1.0, gauge_momentum)
    gauge_momentum_lexico = gauge_latt.lexico(gauge_momentum.getHost(), True)
    gauge_force = _decode_momentum(gauge_momentum_lexico)
    selected_gauge = numpy.asarray(
        [gauge_force[mu, t, z, y, x] for mu, x, y, z, t in _LINKS]
    )
    numpy.testing.assert_allclose(
        selected_gauge,
        _GAUGE_MOMENTUM_FACTOR * _CLG_GAUGE_FORCE,
        rtol=0.0,
        atol=1e-14,
    )

    momentum = LatticeMom(gauge_latt)
    action.force(1.0, momentum)
    momentum_lexico = gauge_latt.lexico(momentum.getHost(), True)
    force = _decode_momentum(momentum_lexico)
    selected = numpy.asarray(
        [force[mu, t, z, y, x] for mu, x, y, z, t in _LINKS]
    )

    # GaugeMomentumFactor=2 is applied when CLGLib consumes its closed force.
    # QUDA's decoded momentum matrix is therefore twice the stored CLGLib
    # matrix for the same physical force.
    numpy.testing.assert_allclose(
        selected,
        _GAUGE_MOMENTUM_FACTOR * _CLG_FORCE,
        rtol=0.0,
        atol=1e-14,
    )
    gauge_action.close()


if __name__ == "__main__":
    main()
