import logging
from os import environ
from sys import stdout
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Sequence, Union

from mpi4py import MPI
from mpi4py.util import dtlib
import numpy

from .version import __version__  # noqa: F401
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


def getRankFromCoord(coord: List[int], grid: List[int]) -> int:
    x, y, z, t = grid
    return ((coord[0] * y + coord[1]) * z + coord[2]) * t + coord[3]


def getCoordFromRank(rank: int, grid: List[int]) -> List[int]:
    x, y, z, t = grid
    return [rank // t // z // y, rank // t // z % y, rank // t % z, rank % t]


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
    global _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        import atexit
        from platform import node as gethostname

        Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]
        if _MPI_SIZE != Gx * Gy * Gz * Gt:
            _MPI_LOGGER.critical(f"the MPI size {_MPI_SIZE} does not match the grid size {grid_size}", ValueError)
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        _MPI_LOGGER.info(f"Using GPU grid {_GRID_SIZE}")

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

        global _DEFAULT_LATTICE, _CUDA_BACKEND, _HIP, _GPUID, _COMPUTE_CAPABILITY

        if latt_size is not None:
            if t_boundary is None or anisotropy is None:
                _MPI_LOGGER.critical("t_boundary and anisotropy should not be None if latt_size is given", ValueError)
            _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)
            _MPI_LOGGER.info(f"Using default LatticeInfo({latt_size}, {t_boundary}, {anisotropy})")

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

        # if _CUDA_BACKEND == "cupy":
        #     from . import malloc_pyquda

        #     allocator = cupy.cuda.PythonFunctionAllocator(
        #         malloc_pyquda.pyquda_device_malloc, malloc_pyquda.pyquda_device_free
        #     )
        #     cupy.cuda.set_allocator(allocator.malloc)

        # quda/include/communicator_quda.h
        # determine which GPU this rank will use
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if _GPUID < 0:
            device_count = cudaGetDeviceCount()
            if device_count == 0:
                _MPI_LOGGER.critical("No devices found", RuntimeError)

            _GPUID = 0
            for i in range(_MPI_RANK):
                if hostname == hostname_recv_buf[i]:
                    _GPUID += 1

            if _GPUID >= device_count:
                if "QUDA_ENABLE_MPS" in environ and environ["QUDA_ENABLE_MPS"] == "1":
                    _GPUID %= device_count
                    print(f"MPS enabled, rank={_MPI_RANK} -> gpu={_GPUID}")
                else:
                    _MPI_LOGGER.critical(f"Too few GPUs available on {hostname}", RuntimeError)

        props = cudaGetDeviceProperties(_GPUID)
        if hasattr(props, "major") and hasattr(props, "minor"):
            _COMPUTE_CAPABILITY = _ComputeCapability(int(props.major), int(props.minor))
        else:
            _COMPUTE_CAPABILITY = _ComputeCapability(int(props["major"]), int(props["minor"]))

        cudaSetDevice(_GPUID)
        quda.initCommsGridQuda(4, _GRID_SIZE)
        quda.initQuda(_GPUID)
        atexit.register(quda.endQuda)
    else:
        _MPI_LOGGER.warning("PyQUDA is already initialized", RuntimeWarning)


def getLogger():
    return _MPI_LOGGER


def setLoggerLevel(level: Literal["debug", "info", "warning", "error", "critical"]):
    _MPI_LOGGER.logger.setLevel(level.upper())


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridSize():
    assert _GRID_SIZE is not None
    return _GRID_SIZE


def getGridCoord():
    assert _GRID_COORD is not None
    return _GRID_COORD


def setDefaultLattice(latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
    global _DEFAULT_LATTICE
    _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)


def getDefaultLattice():
    assert _DEFAULT_LATTICE is not None
    return _DEFAULT_LATTICE


def getCUDABackend():
    return _CUDA_BACKEND


def isHIP():
    return _HIP


def setGPUID(gpuid: int):
    global _GPUID
    assert _GRID_SIZE is None, "setGPUID() should be called before init()"
    assert gpuid >= 0
    _GPUID = gpuid


def getGPUID():
    return _GPUID


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY


def getSublatticeSize(latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert Lx % Gx == 0 and Ly % Gy == 0 and Lz % Gz == 0 and Lt % Gt == 0
    return [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]


def _getSubarray(shape: Sequence[int], axes: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    for j, i in enumerate(axes):
        sizes[i] *= _GRID_SIZE[j]
        starts[i] *= _GRID_COORD[j]
    return sizes, subsizes, starts


def readMPIFile(
    filename: str,
    dtype: str,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
):
    sizes, subsizes, starts = _getSubarray(shape, axes)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = numpy.empty(subsizes, native_dtype)

    fh = MPI.File.Open(_MPI_COMM, filename, MPI.MODE_RDONLY)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Read_all(buf)
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


def writeMPIFile(
    filename: str,
    dtype: str,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
    buf: numpy.ndarray,
):
    sizes, subsizes, starts = _getSubarray(shape, axes)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = buf.view(native_dtype)

    fh = MPI.File.Open(_MPI_COMM, filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()
