import logging
from os import environ
from sys import stdout
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Tuple, Union

from mpi4py import MPI

from ._version import __version__  # noqa: F401
from . import pyquda as quda
from .field import LatticeInfo


class _MPILogger:
    def __init__(self, root: int = 0) -> None:
        self.root = root
        formatter = logging.Formatter(fmt="{name} {levelname}: {message}", style="{")
        stdout_handler = logging.StreamHandler(stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.WARNING)
        self.logger = logging.getLogger("PyQUDA")
        self.logger.level = logging.DEBUG
        self.logger.handlers = [stdout_handler, stderr_handler]

    def debug(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.debug(msg)

    def info(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.info(msg)

    def warning(self, msg: str, category: Warning):
        if _MPI_RANK == self.root:
            self.logger.warning(msg, exc_info=category(msg), stack_info=True)

    def error(self, msg: str, category: Exception):
        if _MPI_RANK == self.root:
            self.logger.error(msg, exc_info=category(msg), stack_info=True)

    def critical(self, msg: str, category: Exception):
        if _MPI_RANK == self.root:
            self.logger.critical(msg, exc_info=category(msg), stack_info=True)
        raise category(msg)


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_MPI_LOGGER: _MPILogger = _MPILogger()
_MPI_COMM: MPI.Comm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_SIZE: Union[List[int], None] = None
_GRID_COORD: Union[List[int], None] = None
_DEFAULT_LATTICE: Union[LatticeInfo, None] = None
_CUDA_BACKEND: Literal["numpy", "cupy", "torch"] = "cupy"
_HIP: bool = False
_GPUID: int = -1
_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def getRankFromCoord(grid_coord: List[int], grid_size: List[int] = None) -> int:
    Gx, Gy, Gz, Gt = _GRID_SIZE if grid_size is None else grid_size
    gx, gy, gz, gt = grid_coord
    return ((gx * Gy + gy) * Gz + gz) * Gt + gt


def getCoordFromRank(mpi_rank: int, grid_size: List[int] = None) -> List[int]:
    Gx, Gy, Gz, Gt = _GRID_SIZE if grid_size is None else grid_size
    return [mpi_rank // Gt // Gz // Gy, mpi_rank // Gt // Gz % Gy, mpi_rank // Gt % Gz, mpi_rank % Gt]


def _composition4(n):
    """
    Writing n as the sum of 4 natural numbers
    """
    addend: List[Tuple[int, int, int, int]] = []
    for i in range(n + 1):
        for j in range(i, n + 1):
            for k in range(j, n + 1):
                x, y, z, t = i, j - i, k - j, n - k
                addend.append((x, y, z, t))
    return addend


def _factorization4(k: int):
    """
    Writing k as the product of 4 positive numbers
    """
    prime_factor: List[List[Tuple[int, int, int, int]]] = []
    for p in range(2, int(k**0.5) + 1):
        n = 0
        while k % p == 0:
            n += 1
            k //= p
        if n != 0:
            prime_factor.append([(p**x, p**y, p**z, p**t) for x, y, z, t in _composition4(n)])
    if k != 1:
        prime_factor.append([(k**x, k**y, k**z, k**t) for x, y, z, t in _composition4(1)])
    return prime_factor


def _partition(factor: List[List[Tuple[int, int, int, int]]], idx: int, sublatt_size: List[int], grid_size: List[int]):
    if idx == 0:
        factor = _factorization4(factor)
    if idx == len(factor):
        yield grid_size
    else:
        Lx, Ly, Lz, Lt = sublatt_size
        Gx, Gy, Gz, Gt = grid_size
        for x, y, z, t in factor[idx]:
            if Lx % x == 0 and Ly % y == 0 and Lz % z == 0 and Lt % t == 0:
                yield from _partition(
                    factor, idx + 1, [Lx // x, Ly // y, Lz // z, Lt // t], [Gx * x, Gy * y, Gz * z, Gt * t]
                )


def _getDefaultGrid(mpi_size: int, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    latt_vol = Lx * Ly * Lz * Lt
    latt_surf = [latt_vol // latt_size[dir] for dir in range(4)]
    min_comm, min_grid = latt_vol, []
    assert latt_vol % mpi_size == 0, "lattice volume must be divisible by MPI size"
    assert Lx % 2 == 0 and Ly % 2 == 0 and Lz % 2 == 0 and Lt % 2 == 0, "lattice size must be even in all directions"
    for grid_size in _partition(mpi_size, 0, [Lx // 2, Ly // 2, Lz // 2, Lt // 2], [1, 1, 1, 1]):
        comm = [latt_surf[dir] * grid_size[dir] for dir in range(4) if grid_size[dir] > 1]
        if sum(comm) < min_comm:
            min_comm, min_grid = sum(comm), [grid_size]
        elif sum(comm) == min_comm:
            min_grid.append(grid_size)
    if min_grid == []:
        _MPI_LOGGER.critical(
            f"Cannot get the proper GPU grid for lattice size {latt_size} with {mpi_size} MPI processes"
        )
    return min(min_grid)


def _initEnviron(**kwargs):
    def _setEnviron(env, key, value):
        if value is not None:
            if env in environ:
                _MPI_LOGGER.warning(f"Both {env} and init({key}) are set", RuntimeWarning)
            environ[env] = value
        if env in environ:
            _MPI_LOGGER.info(f"Using {env}={environ[env]}")

    for key in kwargs.keys():
        _setEnviron(f"QUDA_{key.upper()}", key, kwargs[key])


def _initEnvironWarn(**kwargs):
    def _setEnviron(env, key, value):
        if value is not None:
            if env in environ:
                _MPI_LOGGER.warning(f"Both {env} and init({key}) are set", RuntimeWarning)
            environ[env] = value
        else:
            if env not in environ:
                _MPI_LOGGER.warning(f"Neither {env} nor init({key}) is set", RuntimeWarning)
        if env in environ:
            _MPI_LOGGER.info(f"Using {env}={environ[env]}")

    for key in kwargs.keys():
        _setEnviron(f"QUDA_{key.upper()}", key, kwargs[key])


def initGPU(backend: Literal["numpy", "cupy", "torch"] = None, gpuid: int = -1):
    global _CUDA_BACKEND, _HIP, _GPUID, _COMPUTE_CAPABILITY

    if isGridInitialized():
        _MPI_LOGGER.critical("initGPU should be called before init", RuntimeError)
    if _GPUID < 0:
        from platform import node as gethostname

        if backend is None:
            backend = environ["PYQUDA_BACKEND"] if "PYQUDA_BACKEND" in environ else "cupy"
        if backend == "numpy":
            cudaGetDeviceCount: Callable[[], int] = lambda: 0x7FFFFFFF
            cudaGetDeviceProperties: Callable[[int], Dict[str, Any]] = lambda device: {"major": 0, "minor": 0}
            cudaSetDevice: Callable[[int], None] = lambda device: None
        elif backend == "cupy":
            import cupy
            from cupy.cuda.runtime import getDeviceCount as cudaGetDeviceCount
            from cupy.cuda.runtime import getDeviceProperties as cudaGetDeviceProperties
            from cupy.cuda.runtime import is_hip

            cudaSetDevice: Callable[[int], None] = lambda device: cupy.cuda.Device(device).use()
            _HIP = is_hip
        elif backend == "torch":
            import torch
            from torch.cuda import device_count as cudaGetDeviceCount
            from torch.cuda import get_device_properties as cudaGetDeviceProperties
            from torch.version import hip

            cudaSetDevice: Callable[[int], None] = lambda device: torch.set_default_device(f"cuda:{device}")
            _HIP = hip is not None
        else:
            _MPI_LOGGER.critical(f"Unsupported CUDA backend {backend}", ValueError)
        _CUDA_BACKEND = backend
        _MPI_LOGGER.info(f"Using CUDA backend {backend}")

        # quda/include/communicator_quda.h
        # determine which GPU this rank will use
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if gpuid < 0:
            device_count = cudaGetDeviceCount()
            if device_count == 0:
                _MPI_LOGGER.critical("No devices found", RuntimeError)

            gpuid = 0
            for i in range(_MPI_RANK):
                if hostname == hostname_recv_buf[i]:
                    gpuid += 1

            if gpuid >= device_count:
                if "QUDA_ENABLE_MPS" in environ and environ["QUDA_ENABLE_MPS"] == "1":
                    gpuid %= device_count
                    print(f"MPS enabled, rank={_MPI_RANK} -> gpu={gpuid}")
                else:
                    _MPI_LOGGER.critical(f"Too few GPUs available on {hostname}", RuntimeError)
        _GPUID = gpuid

        props = cudaGetDeviceProperties(gpuid)
        if hasattr(props, "major") and hasattr(props, "minor"):
            _COMPUTE_CAPABILITY = _ComputeCapability(int(props.major), int(props.minor))
        else:
            _COMPUTE_CAPABILITY = _ComputeCapability(int(props["major"]), int(props["minor"]))

        cudaSetDevice(gpuid)
    else:
        _MPI_LOGGER.warning("GPU is already initialized", RuntimeWarning)


def initQUDA(grid_size: List[int], gpuid: int):
    import atexit

    # if _CUDA_BACKEND == "cupy":
    #     import cupy
    #     from . import malloc_pyquda

    #     allocator = cupy.cuda.PythonFunctionAllocator(
    #         malloc_pyquda.pyquda_device_malloc, malloc_pyquda.pyquda_device_free
    #     )
    #     cupy.cuda.set_allocator(allocator.malloc)

    quda.initCommsGridQuda(4, grid_size)
    quda.initQuda(gpuid)
    atexit.register(quda.endQuda)


def init(
    grid_size: List[int] = None,
    latt_size: List[int] = None,
    t_boundary: Literal[1, -1] = None,
    anisotropy: float = None,
    backend: Literal["numpy", "cupy", "torch"] = None,
    init_quda: bool = True,
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
    global _GRID_SIZE, _GRID_COORD, _DEFAULT_LATTICE
    if _GRID_SIZE is None:
        initGPU(backend)

        use_default_grid = grid_size is None and latt_size is not None
        use_default_latt = latt_size is not None and t_boundary is not None and anisotropy is not None
        if use_default_grid:
            grid_size = _getDefaultGrid(_MPI_SIZE, latt_size)
        Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]
        if _MPI_SIZE != Gx * Gy * Gz * Gt:
            _MPI_LOGGER.critical(f"The MPI size {_MPI_SIZE} does not match the grid size {grid_size}", ValueError)
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        _MPI_LOGGER.info(f"Using the grid size {_GRID_SIZE}")
        if use_default_grid and not use_default_latt:
            _MPI_LOGGER.info(f"Using the lattice size {latt_size} only for getting the default grid size {_GRID_SIZE}")
        if use_default_latt:
            _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)
            _MPI_LOGGER.info(f"Using the default lattice LatticeInfo({latt_size}, {t_boundary}, {anisotropy})")

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

        if init_quda:
            initQUDA(_GRID_SIZE, _GPUID)
    else:
        _MPI_LOGGER.warning("PyQUDA is already initialized", RuntimeWarning)


def getLogger():
    return _MPI_LOGGER


def setLoggerLevel(level: Literal["debug", "info", "warning", "error", "critical"]):
    _MPI_LOGGER.logger.setLevel(level.upper())


def isGPUInitialized():
    return _GPUID >= 0


def isGridInitialized():
    return _GRID_SIZE is not None


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridSize():
    assert _GRID_SIZE is not None, "PyQUDA is not initialized"
    return _GRID_SIZE


def getGridCoord():
    assert _GRID_COORD is not None, "PyQUDA is not initialized"
    return _GRID_COORD


def setDefaultLattice(latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
    global _DEFAULT_LATTICE
    _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)


def getDefaultLattice():
    assert _DEFAULT_LATTICE is not None, "Default lattice is not set"
    return _DEFAULT_LATTICE


def getCUDABackend():
    return _CUDA_BACKEND


def isHIP():
    return _HIP


def getGPUID():
    return _GPUID


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY
