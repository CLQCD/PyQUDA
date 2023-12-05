from typing import List, Literal, NamedTuple
from warnings import warn

from . import mpi
from . import pyquda as quda, malloc_pyquda


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_GPUID: int = 0
_CUDA_BACKEND: Literal["cupy", "torch"] = "cupy"
_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def setGPUID(gpuid: int):
    global _GPUID
    _GPUID = gpuid


def setCUDABackend(backend: Literal["cupy", "torch"]):
    assert backend in ["cupy", "torch"], f"Unsupported backend {backend}"
    global _CUDA_BACKEND
    _CUDA_BACKEND = backend


def getCUDABackend():
    return _CUDA_BACKEND


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY


def init(grid_size: List[int] = None):
    """
    Initialize MPI along with the QUDA library.

    If grid_size is None, MPI will not applied.
    """
    if mpi.comm is None:
        import atexit
        from .pyquda import initCommsGridQuda, initQuda, endQuda

        if _CUDA_BACKEND == "cupy":
            from cupy import cuda
        elif _CUDA_BACKEND == "torch":
            from torch import cuda
        else:
            raise ImportError("CuPy or PyTorch is needed to handle field data")

        gpuid = 0
        Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]

        try:
            from mpi4py import MPI
            from os import getenv
            from platform import node as gethostname
        except ImportError:
            mpi.comm = 0
        else:
            mpi.comm = MPI.COMM_WORLD
            mpi.rank = mpi.comm.Get_rank()
            mpi.size = mpi.comm.Get_size()
            mpi.grid = [Gx, Gy, Gz, Gt]
            mpi.coord = mpi.coordFromRank(mpi.rank)

            hostname = gethostname()
            hostname_recv_buf = mpi.comm.allgather(hostname)
            for i in range(mpi.rank):
                if hostname == hostname_recv_buf[i]:
                    gpuid += 1

            if _CUDA_BACKEND == "cupy":
                device_count = cuda.runtime.getDeviceCount()
            elif _CUDA_BACKEND == "torch":
                device_count = cuda.device_count()
            if gpuid >= device_count:
                enable_mps_env = getenv("QUDA_ENABLE_MPS")
                if enable_mps_env is not None and enable_mps_env == "1":
                    gpuid %= device_count

        assert Gx * Gy * Gz * Gt == mpi.size
        initCommsGridQuda(4, [Gx, Gy, Gz, Gt])

        global _GPUID, _COMPUTE_CAPABILITY
        gpuid += _GPUID
        _GPUID = gpuid

        if _CUDA_BACKEND == "cupy":
            cuda.Device(gpuid).use()
            cc = cuda.Device(gpuid).compute_capability
            _COMPUTE_CAPABILITY = _ComputeCapability(int(cc[:-1]), int(cc[-1]))
            allocator = cuda.PythonFunctionAllocator(malloc_pyquda.pyquda_cupy_malloc, malloc_pyquda.pyquda_cupy_free)
            cuda.set_allocator(allocator.malloc)
        elif _CUDA_BACKEND == "torch":
            cuda.set_device(gpuid)
            cc = cuda.get_device_capability(gpuid)
            _COMPUTE_CAPABILITY = _ComputeCapability(cc[0], cc[1])

        initQuda(gpuid)
        atexit.register(endQuda)
    else:
        warn("PyQuda is already initialized", RuntimeWarning)
