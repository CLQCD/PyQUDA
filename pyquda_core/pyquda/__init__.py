from os import environ, path
from typing import List, Literal, Optional

from ._version import __version__  # noqa: F401
from pyquda_comm import (  # noqa: F401
    GridMapType,
    BackendType,
    BackendTargetType,
    initGrid,
    initDevice,
    isGridInitialized,
    isDeviceInitialized,
    getLogger,
    setLoggerLevel,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridMap,
    getGridSize,
    getGridCoord,
    getArrayBackend,
    getArrayDevice,
)
from . import quda_define

_QUDA_INITIALIZED: bool = False


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


def initQUDA(grid_size: List[int], device: int, use_quda_allocator: bool = False):
    import atexit
    from . import pyquda as quda, malloc_pyquda

    global _QUDA_INITIALIZED
    if not isGridInitialized() or not isDeviceInitialized():
        getLogger().critical("initGrid and initDevice should be called before initQUDA", RuntimeError)

    if use_quda_allocator:
        if getArrayBackend() == "cupy":
            import cupy

            allocator = cupy.cuda.PythonFunctionAllocator(
                malloc_pyquda.pyquda_device_malloc, malloc_pyquda.pyquda_device_free
            )
            cupy.cuda.set_allocator(allocator.malloc)

    quda.initCommsGridQuda(4, grid_size, getGridMap().encode())
    quda.initQuda(device)
    atexit.register(quda.endQuda)
    _QUDA_INITIALIZED = True


def init(
    grid_size: Optional[List[int]] = None,
    latt_size: Optional[List[int]] = None,
    grid_map: GridMapType = "default",
    backend: BackendType = "cupy",
    backend_target: BackendTargetType = quda_define.target(),
    init_quda: bool = True,
    *,
    resource_path: str = "",
    rank_verbosity: List[int] = [0],
    enable_mps: bool = False,
    enable_gdr: bool = False,
    enable_gdr_blacklist: List[int] = [],
    enable_p2p: Literal[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] = 3,
    enable_p2p_max_access_rank: int = 0x7FFFFFFF,
    enable_zero_copy: bool = False,
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
    enable_monitor: bool = False,
    enable_monitor_period: int = 1000,
    enable_device_memory_pool: bool = True,
    enable_pinned_memory_pool: bool = True,
    enable_managed_memory: bool = False,
    enable_managed_prefetch: bool = False,
    deterministic_reduce: bool = False,
    device_reset: bool = False,
    max_multi_rhs: int = 0,
):
    """
    Initialize MPI along with the QUDA library.
    """
    if not isGridInitialized() or not isDeviceInitialized():
        initGrid(grid_map, grid_size, latt_size)
        initDevice(backend, backend_target, -1, enable_mps)

    if init_quda:
        if not _QUDA_INITIALIZED:
            _setEnvironWarn(
                resource_path=path.expanduser(path.expandvars(resource_path)) if resource_path != "" else None
            )
            _setEnviron(
                rank_verbosity=",".join([str(i) for i in rank_verbosity]) if rank_verbosity != [0] else None,
                enable_mps="1" if enable_mps else None,
                enable_gdr="1" if enable_gdr else None,
                enable_gdr_blacklist=(
                    ",".join([str(i) for i in enable_gdr_blacklist]) if enable_gdr_blacklist != [] else None
                ),
                enable_p2p=str(enable_p2p) if enable_p2p != 3 else None,
                enable_p2p_max_access_rank=(
                    str(enable_p2p_max_access_rank) if enable_p2p_max_access_rank < 0x7FFFFFFF else None
                ),
                enable_zero_copy="1" if enable_zero_copy else None,
                enable_nvshmem="0" if not enable_nvshmem else None,
                allow_jit="1" if allow_jit else None,
                reorder_location="CPU" if reorder_location == "CPU" else None,
                enable_tuning="0" if not enable_tuning else None,
                enable_tuning_shared="0" if not enable_tuning_shared else None,
                tune_version_check="0" if not tune_version_check else None,
                tuning_rank=str(tuning_rank) if tuning_rank > 0 else None,
                profile_output_base=profile_output_base if profile_output_base != "" else None,
                enable_target_profile=(
                    ",".join([str(i) for i in enable_target_profile]) if enable_target_profile != [] else None
                ),
                do_not_profile="1" if do_not_profile else None,
                enable_trace=str(enable_trace) if enable_trace > 0 else None,
                enable_force_monitor="1" if enable_force_monitor else None,
                enable_monitor="1" if enable_monitor else None,
                enable_monitor_period=str(enable_monitor_period) if enable_monitor_period != 1000 else None,
                enable_device_memory_pool="0" if not enable_device_memory_pool else None,
                enable_pinned_memory_pool="0" if not enable_pinned_memory_pool else None,
                enable_managed_memory="1" if enable_managed_memory else None,
                enable_managed_prefetch="1" if enable_managed_prefetch else None,
                deterministic_reduce="1" if deterministic_reduce else None,
                device_reset="1" if device_reset else None,
                max_multi_rhs=str(max_multi_rhs) if max_multi_rhs > 0 else None,
            )
            initQUDA(getGridSize(), getArrayDevice())
        else:
            getLogger().warning("PyQUDA is already initialized", RuntimeWarning)


def isQUDAInitialized():
    return _QUDA_INITIALIZED
