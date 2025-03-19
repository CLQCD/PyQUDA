from os import environ
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Union

from ._version import __version__  # noqa: F401
from .field import LatticeInfo
from pyquda_comm import (  # noqa: F401
    initGrid,
    isGridInitialized,
    getLogger,
    setLoggerLevel,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridSize,
    getGridCoord,
    getGridMap,
    setGridMap,
)


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_DEFAULT_LATTICE: Union[LatticeInfo, None] = None
_CUDA_BACKEND: Literal["numpy", "cupy", "torch"] = "cupy"
_HIP: bool = False
_GPUID: int = -1
_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def _setEnviron(**kwargs):
    def _setEnviron(env, key, value):
        if value is not None:
            if env in environ:
                getLogger().warning(f"Both {env} and init({key}) are set", RuntimeWarning)
            environ[env] = value
        if env in environ:
            getLogger().info(f"Using {env}={environ[env]}")

    for key in kwargs.keys():
        _setEnviron(f"QUDA_{key.upper()}", key, kwargs[key])


def _setEnvironWarn(**kwargs):
    def _setEnviron(env, key, value):
        if value is not None:
            if env in environ:
                getLogger().warning(f"Both {env} and init({key}) are set", RuntimeWarning)
            environ[env] = value
        else:
            if env not in environ:
                getLogger().warning(f"Neither {env} nor init({key}) is set", RuntimeWarning)
        if env in environ:
            getLogger().info(f"Using {env}={environ[env]}")

    for key in kwargs.keys():
        _setEnviron(f"QUDA_{key.upper()}", key, kwargs[key])


def initGPU(backend: Literal["numpy", "cupy", "torch"] = None, gpuid: int = -1):
    global _CUDA_BACKEND, _HIP, _GPUID, _COMPUTE_CAPABILITY
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
            getLogger().critical(f"Unsupported CUDA backend {backend}", ValueError)
        _CUDA_BACKEND = backend
        getLogger().info(f"Using CUDA backend {backend}")

        # quda/include/communicator_quda.h
        # determine which GPU this rank will use
        hostname = gethostname()
        hostname_recv_buf = getMPIComm().allgather(hostname)

        if gpuid < 0:
            device_count = cudaGetDeviceCount()
            if device_count == 0:
                getLogger().critical("No devices found", RuntimeError)

            gpuid = 0
            for i in range(getMPIRank()):
                if hostname == hostname_recv_buf[i]:
                    gpuid += 1

            if gpuid >= device_count:
                if "QUDA_ENABLE_MPS" in environ and environ["QUDA_ENABLE_MPS"] == "1":
                    gpuid %= device_count
                    print(f"MPS enabled, rank={getMPIRank():3d} -> gpu={gpuid}")
                else:
                    raise RuntimeError(f"Too few GPUs available on {hostname}")
        _GPUID = gpuid

        props = cudaGetDeviceProperties(gpuid)
        if hasattr(props, "major") and hasattr(props, "minor"):
            _COMPUTE_CAPABILITY = _ComputeCapability(int(props.major), int(props.minor))
        else:
            _COMPUTE_CAPABILITY = _ComputeCapability(int(props["major"]), int(props["minor"]))

        cudaSetDevice(gpuid)
    else:
        getLogger().warning("GPU is already initialized", RuntimeWarning)


def initQUDA(grid_size: List[int], gpuid: int, use_quda_allocator: bool = False):
    import atexit
    from . import pyquda as quda, malloc_pyquda

    if use_quda_allocator:
        if _CUDA_BACKEND == "cupy":
            import cupy

            allocator = cupy.cuda.PythonFunctionAllocator(
                malloc_pyquda.pyquda_device_malloc, malloc_pyquda.pyquda_device_free
            )
            cupy.cuda.set_allocator(allocator.malloc)

    quda.initCommsGridQuda(4, grid_size, getGridMap().encode())
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
    global _DEFAULT_LATTICE
    if getGridSize(False) is None:
        use_default_grid = grid_size is None and latt_size is not None
        use_default_latt = latt_size is not None and t_boundary is not None and anisotropy is not None
        initGrid(grid_size, latt_size)
        if use_default_grid and not use_default_latt:
            getLogger().info(
                f"Using the lattice size {latt_size} only for getting the default grid size {getGridSize()}"
            )
        if use_default_latt:
            _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)
            getLogger().info(f"Using the default lattice LatticeInfo({latt_size}, {t_boundary}, {anisotropy})")

        _setEnvironWarn(resource_path=resource_path if resource_path != "" else None)
        _setEnviron(
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

        initGPU(backend, -1)
        if init_quda:
            initQUDA(getGridSize(), _GPUID)
    else:
        getLogger().warning("PyQUDA is already initialized", RuntimeWarning)


def isGPUInitialized():
    return _GPUID >= 0


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
