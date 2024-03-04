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
_GPUID: int = -1
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
    backend: Literal["cupy", "torch"] = "cupy",
    *,
    resource_path: str = None,
    enable_tuning: str = None,
    enable_tuning_shared: str = None,
    tune_version_check: str = None,
    tuning_rank: str = None,
    profile_output_base: str = None,
    enable_target_profile: str = None,
    do_not_profile: str = None,
    enable_trace: str = None,
    enable_mps: str = None,
    rank_verbosity: str = None,
    enable_managed_memory: str = None,
    enable_managed_prefetch: str = None,
    enable_device_memory_pool: str = None,
    enable_pinned_memory_pool: str = None,
    enable_p2p: str = None,
    enable_p2p_max_access_rank: str = None,
    enable_gdr: str = None,
    enable_gdr_blacklist: str = None,
    enable_nvshmem: str = None,
    deterministic_reduce: str = None,
    allow_jit: str = None,
    device_reset: str = None,
    reorder_location: str = None,
    enable_force_monitor: str = None,
):
    """
    Initialize MPI along with the QUDA library.
    """
    filterwarnings("default", "", DeprecationWarning)
    global _MPI_COMM, _MPI_SIZE, _MPI_RANK, _GRID_SIZE, _GRID_COORD
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

        _initEnvironWarn(resource_path=resource_path)
        _initEnviron(
            resource_path=resource_path,
            enable_tuning=enable_tuning,
            enable_tuning_shared=enable_tuning_shared,
            tune_version_check=tune_version_check,
            tuning_rank=tuning_rank,
            profile_output_base=profile_output_base,
            enable_target_profile=enable_target_profile,
            do_not_profile=do_not_profile,
            enable_trace=enable_trace,
            enable_mps=enable_mps,
            rank_verbosity=rank_verbosity,
            enable_managed_memory=enable_managed_memory,
            enable_managed_prefetch=enable_managed_prefetch,
            enable_device_memory_pool=enable_device_memory_pool,
            enable_pinned_memory_pool=enable_pinned_memory_pool,
            enable_p2p=enable_p2p,
            enable_p2p_max_access_rank=enable_p2p_max_access_rank,
            enable_gdr=enable_gdr,
            enable_gdr_blacklist=enable_gdr_blacklist,
            enable_nvshmem=enable_nvshmem,
            deterministic_reduce=deterministic_reduce,
            allow_jit=allow_jit,
            device_reset=device_reset,
            reorder_location=reorder_location,
            enable_force_monitor=enable_force_monitor,
        )

        global _DEFAULT_LATTICE, _CUDA_BACKEND, _GPUID, _COMPUTE_CAPABILITY

        if latt_size is not None:
            assert t_boundary is not None and anisotropy is not None
            _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)
            printRoot(f"INFO: Using default LatticeInfo({latt_size}, {t_boundary}, {anisotropy})")

        if backend == "cupy":
            from cupy import cuda
            from . import malloc_pyquda
        elif backend == "torch":
            from torch import cuda
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")
        _CUDA_BACKEND = backend

        # determine which GPU this rank will use @quda/include/communicator_quda.h
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if _GPUID < 0:
            if _CUDA_BACKEND == "cupy":
                device_count = cuda.runtime.getDeviceCount()
            elif _CUDA_BACKEND == "torch":
                device_count = cuda.device_count()
            if device_count == 0:
                raise RuntimeError("No devices found")

            _GPUID = 0
            for i in range(_MPI_RANK):
                if hostname == hostname_recv_buf[i]:
                    _GPUID += 1

            if _GPUID >= device_count:
                if "QUDA_ENABLE_MPS" in environ and environ["QUDA_ENABLE_MPS"] == "1":
                    _GPUID %= device_count
                    print(f"INFO: MPS enabled, rank={_MPI_RANK} -> gpu={_GPUID}")
                else:
                    raise RuntimeError(f"Too few GPUs available on {hostname}")

        if _CUDA_BACKEND == "cupy":
            cuda.Device(_GPUID).use()
            cc = cuda.Device(_GPUID).compute_capability
            _COMPUTE_CAPABILITY = _ComputeCapability(int(cc[:-1]), int(cc[-1]))
            allocator = cuda.PythonFunctionAllocator(malloc_pyquda.pyquda_cupy_malloc, malloc_pyquda.pyquda_cupy_free)
            cuda.set_allocator(allocator.malloc)
        elif _CUDA_BACKEND == "torch":
            cuda.set_device(_GPUID)
            cc = cuda.get_device_capability(_GPUID)
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


def setGPUID(gpuid: int):
    global _MPI_COMM, _GPUID
    assert _MPI_COMM is None, "setGPUID() should be called before init()"
    assert gpuid >= 0
    _GPUID = gpuid


def getGPUID():
    return _GPUID


def getCUDABackend():
    return _CUDA_BACKEND


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY
