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

from .version import __version__  # noqa: F401
from . import pyquda as quda
from .field import LatticeInfo


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_MPI_COMM: MPI.Comm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_SIZE: List[int] = None
_GRID_COORD: List[int] = [0, 0, 0, 0]
_DEFAULT_LATTICE: LatticeInfo = None
_CUDA_BACKEND: Literal["numpy", "cupy", "torch"] = "cupy"
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
    backend: Literal["numpy", "cupy", "torch"] = "cupy",
    *,
    resource_path: str = "",
    rank_verbosity: List[int] = [0],
    enable_mps: bool = False,
    enable_gdr: bool = False,
    enable_gdr_blacklist: List[int] = [],
    enable_p2p: Literal[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] = 3,
    enable_p2p_max_access_rank: int = 0x7FFFFFFF,
    enable_nvshmem: bool = True,
    allow_jit: bool = False,
    reorder_location: Literal["GPU", "CPU"] = "GPU",
    enable_tuning: bool = True,
    enable_tuning_shared: bool = True,
    tune_version_check: bool = True,
    tuning_rank: int = 0,
    profile_output_base: str = "",
    enable_target_profile: List[int] = [],
    do_not_profile: bool = False,
    enable_trace: Literal[0, 1, 2] = 0,
    enable_force_monitor: bool = False,
    enable_device_memory_pool: bool = True,
    enable_pinned_memory_pool: bool = True,
    enable_managed_memory: bool = False,
    enable_managed_prefetch: bool = False,
    deterministic_reduce: bool = False,
    device_reset: bool = False,
):
    """
    Initialize MPI along with the QUDA library.
    """
    filterwarnings("default", "", DeprecationWarning)
    global _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        import atexit
        from platform import node as gethostname

        Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]
        assert _MPI_SIZE == Gx * Gy * Gz * Gt
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        printRoot(f"INFO: Using gird {_GRID_SIZE}")

        _initEnvironWarn(resource_path=resource_path if resource_path != "" else None)
        _initEnviron(
            rank_verbosity=",".join(rank_verbosity) if rank_verbosity != [0] else None,
            enable_mps="1" if enable_mps else None,
            enable_gdr="1" if enable_gdr else None,
            enable_gdr_blacklist=",".join(enable_gdr_blacklist) if enable_gdr_blacklist != [] else None,
            enable_p2p=str(enable_p2p) if enable_p2p != 3 else None,
            enable_p2p_max_access_rank=(
                str(enable_p2p_max_access_rank) if enable_p2p_max_access_rank < 0x7FFFFFFF else None
            ),
            enable_nvshmem="0" if not enable_nvshmem else None,
            allow_jit="1" if allow_jit else None,
            reorder_location="CPU" if reorder_location == "CPU" else None,
            enable_tuning="0" if not enable_tuning else None,
            enable_tuning_shared="0" if not enable_tuning_shared else None,
            tune_version_check="0" if not tune_version_check else None,
            tuning_rank=str(tuning_rank) if tuning_rank else None,
            profile_output_base=profile_output_base if profile_output_base != "" else None,
            enable_target_profile=",".join(enable_target_profile) if enable_target_profile != [] else None,
            do_not_profile="1" if do_not_profile else None,
            enable_trace="1" if enable_trace else None,
            enable_force_monitor="1" if enable_force_monitor else None,
            enable_device_memory_pool="0" if not enable_device_memory_pool else None,
            enable_pinned_memory_pool="0" if not enable_pinned_memory_pool else None,
            enable_managed_memory="1" if enable_managed_memory else None,
            enable_managed_prefetch="1" if enable_managed_prefetch else None,
            deterministic_reduce="1" if deterministic_reduce else None,
            device_reset="1" if device_reset else None,
        )

        global _DEFAULT_LATTICE, _CUDA_BACKEND, _GPUID, _COMPUTE_CAPABILITY

        if latt_size is not None:
            assert t_boundary is not None and anisotropy is not None
            _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)
            printRoot(f"INFO: Using default LatticeInfo({latt_size}, {t_boundary}, {anisotropy})")

        if backend == "numpy":
            pass
        elif backend == "cupy":
            from cupy import cuda
            # from . import malloc_pyquda
        elif backend == "torch":
            from torch import cuda, set_default_device
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")
        _CUDA_BACKEND = backend

        # determine which GPU this rank will use @quda/include/communicator_quda.h
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if _GPUID < 0:
            if _CUDA_BACKEND == "numpy":
                device_count = 0x7FFFFFFF
            elif _CUDA_BACKEND == "cupy":
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

        if _CUDA_BACKEND == "numpy":
            pass
        elif _CUDA_BACKEND == "cupy":
            cuda.Device(_GPUID).use()
            cc = cuda.Device(_GPUID).compute_capability
            _COMPUTE_CAPABILITY = _ComputeCapability(int(cc[:-1]), int(cc[-1]))
            # allocator = cuda.PythonFunctionAllocator(malloc_pyquda.pyquda_cupy_malloc, malloc_pyquda.pyquda_cupy_free)
            # cuda.set_allocator(allocator.malloc)
        elif _CUDA_BACKEND == "torch":
            set_default_device(f"cuda:{_GPUID}")
            cc = cuda.get_device_capability(_GPUID)
            _COMPUTE_CAPABILITY = _ComputeCapability(cc[0], cc[1])

        quda.initCommsGridQuda(4, _GRID_SIZE)
        quda.initQuda(_GPUID)
        atexit.register(quda.endQuda)
    else:
        warn("WARNING: PyQUDA is already initialized", RuntimeWarning)


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
    global _GPUID
    assert _GRID_SIZE is None, "setGPUID() should be called before init()"
    assert gpuid >= 0
    _GPUID = gpuid


def getGPUID():
    return _GPUID


def getCUDABackend():
    return _CUDA_BACKEND


def setCUDAComputeCapability(major: int, minor: int):
    global _COMPUTE_CAPABILITY
    _COMPUTE_CAPABILITY = _ComputeCapability(major, minor)


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY
