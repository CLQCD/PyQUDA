from __future__ import annotations  # TYPE_CHECKING
from os import environ
from typing import TYPE_CHECKING, List, Literal, NamedTuple
from warnings import warn, filterwarnings

if TYPE_CHECKING:
    from typing import Protocol, TypeVar
    from _typeshed import SupportsFlush, SupportsWrite

    _T_contra = TypeVar("_T_contra", contravariant=True)

    class SupportsWriteAndFlush(SupportsWrite[_T_contra], SupportsFlush, Protocol[_T_contra]):
        pass


from mpi4py import MPI

__version__ = "0.5.6"
from . import pyquda as quda
from .field import LatticeInfo


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_MPI_COMM: MPI.Comm = None
_MPI_SIZE: int = 1
_MPI_RANK: int = 0
_GRID_SIZE: List[int] = [1, 1, 1, 1]
_GRID_COORD: List[int] = [0, 0, 0, 0]
_DEFAULT_LATTICE: LatticeInfo = None
_CUDA_BACKEND: Literal["cupy", "torch"] = "cupy"
_GPUID: int = 0
_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def getRankFromCoord(coord: List[int], grid: List[int]) -> int:
    x, y, z, t = grid
    return ((coord[0] * y + coord[1]) * z + coord[2]) * t + coord[3]


def getCoordFromRank(rank: int, grid: List[int]) -> List[int]:
    x, y, z, t = grid
    return [rank // t // z // y, rank // t // z % y, rank // t % z, rank % t]


def printRoot(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: SupportsWriteAndFlush[str] | None = None,
    flush: bool = False,
):
    if _MPI_RANK == 0:
        print(*values, sep=sep, end=end, file=file, flush=flush)


def _initEnviron(**kwargs):
    def _setEnviron(env, key, value):
        if value is not None:
            if env in environ:
                warn(f"WARNING: Both {env} and init({key}) are set", RuntimeWarning)
            environ[env] = value
        if env in environ:
            printRoot(f"INFO: Using {env}={environ[env]}")

    for key in kwargs.keys():
        _setEnviron(f"QUDA_{key.upper()}", key, kwargs[key])


def _initEnvironWarn(**kwargs):
    def _setEnviron(env, key, value):
        if value is not None:
            if env in environ:
                warn(f"WARNING: Both {env} and init({key}) are set", RuntimeWarning)
            environ[env] = value
        else:
            if env not in environ:
                warn(f"WARNING: Neither {env} nor init({key}) is set", RuntimeWarning)
        if env in environ:
            printRoot(f"INFO: Using {env}={environ[env]}")

    for key in kwargs.keys():
        _setEnviron(f"QUDA_{key.upper()}", key, kwargs[key])


def init(
    grid_size: List[int] = None,
    latt_size: List[int] = None,
    t_boundary: Literal[1, -1] = None,
    anisotropy: float = None,
    *,
    backend: Literal["cupy", "torch"] = "cupy",
    resource_path: str = None,
    enable_mps: str = None,
):
    """
    Initialize MPI along with the QUDA library.
    """
    filterwarnings("default", "", DeprecationWarning)
    global _MPI_COMM, _MPI_SIZE, _MPI_RANK, _GRID_SIZE, _GRID_COORD, _DEFAULT_LATTICE
    if _MPI_COMM is None:
        import atexit
        from platform import node as gethostname

        Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]
        _MPI_COMM = MPI.COMM_WORLD
        _MPI_SIZE = _MPI_COMM.Get_size()
        _MPI_RANK = _MPI_COMM.Get_rank()
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        assert _MPI_SIZE == Gx * Gy * Gz * Gt
        printRoot(f"INFO: Using gird {_GRID_SIZE}")

        if latt_size is not None:
            assert t_boundary is not None and anisotropy is not None
            _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)
            printRoot(f"INFO: Using default LatticeInfo({latt_size}, {t_boundary}, {anisotropy})")

        _initEnvironWarn(resource_path=resource_path)
        _initEnviron(enable_mps=enable_mps)

        global _CUDA_BACKEND, _GPUID, _COMPUTE_CAPABILITY

        assert backend in ["cupy", "torch"], f"Unsupported backend {backend}"
        if backend == "cupy":
            from cupy import cuda
            from . import malloc_pyquda
        elif backend == "torch":
            from torch import cuda
        else:
            raise ImportError("Either CuPy or PyTorch is needed to handle the field data")
        _CUDA_BACKEND = backend

        gpuid = 0
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)
        for i in range(_MPI_RANK):
            if hostname == hostname_recv_buf[i]:
                gpuid += 1

        if backend == "cupy":
            device_count = cuda.runtime.getDeviceCount()
        elif backend == "torch":
            device_count = cuda.device_count()
        if gpuid >= device_count:
            if "QUDA_ENABLE_MPS" in environ and environ["QUDA_ENABLE_MPS"] == "1":
                gpuid %= device_count
        _GPUID = gpuid

        if backend == "cupy":
            cuda.Device(gpuid).use()
            cc = cuda.Device(gpuid).compute_capability
            _COMPUTE_CAPABILITY = _ComputeCapability(int(cc[:-1]), int(cc[-1]))
            allocator = cuda.PythonFunctionAllocator(malloc_pyquda.pyquda_cupy_malloc, malloc_pyquda.pyquda_cupy_free)
            cuda.set_allocator(allocator.malloc)
        elif backend == "torch":
            cuda.set_device(gpuid)
            cc = cuda.get_device_capability(gpuid)
            _COMPUTE_CAPABILITY = _ComputeCapability(cc[0], cc[1])

        quda.initCommsGridQuda(4, _GRID_SIZE)
        quda.initQuda(_GPUID)
        atexit.register(quda.endQuda)
    else:
        warn("WARNING: PyQuda is already initialized", RuntimeWarning)


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridSize():
    return _GRID_SIZE


def getGridCoord():
    return _GRID_COORD


def setDefaultLattice(latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
    global _DEFAULT_LATTICE
    _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)


def getDefaultLattice():
    assert _DEFAULT_LATTICE is not None
    return _DEFAULT_LATTICE


def getGPUID():
    return _GPUID


def getCUDABackend():
    return _CUDA_BACKEND


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY
