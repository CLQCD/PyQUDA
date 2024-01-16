from typing import List, Literal, NamedTuple
from warnings import warn

from mpi4py import MPI

__version__ = "0.5.0"
from . import pyquda as quda


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_MPI_COMM: MPI.Comm = None
_MPI_SIZE: int = 1
_MPI_RANK: int = 0
_GRID_SIZE: List[int] = [1, 1, 1, 1]
_GRID_COORD: List[int] = [0, 0, 0, 0]
_GPUID: int = 0
_CUDA_BACKEND: Literal["cupy", "torch"] = "cupy"
_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def getRankFromCoord(coord: List[int], grid: List[int]) -> int:
    x, y, z, t = grid
    return ((coord[0] * y + coord[1]) * z + coord[2]) * t + coord[3]


def getCoordFromRank(rank: int, grid: List[int]) -> List[int]:
    x, y, z, t = grid
    return [rank // t // z // y, rank // t // z % y, rank // t % z, rank % t]


def init(grid_size: List[int] = None, backend: Literal["cupy", "torch"] = "cupy", resource_path: str = None):
    """
    Initialize MPI along with the QUDA library.

    If grid_size is None, MPI will not applied.
    """
    global _MPI_COMM, _MPI_SIZE, _MPI_RANK, _GRID_SIZE, _GRID_COORD
    if _MPI_COMM is None:
        import atexit
        from os import environ
        from platform import node as gethostname

        assert backend in ["cupy", "torch"], f"Unsupported backend {backend}"
        if backend == "cupy":
            from cupy import cuda
            from . import malloc_pyquda
        elif backend == "torch":
            from torch import cuda
        else:
            raise ImportError("CuPy or PyTorch is needed to handle field data")

        gpuid = 0
        Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]

        _MPI_COMM = MPI.COMM_WORLD
        _MPI_SIZE = _MPI_COMM.Get_size()
        _MPI_RANK = _MPI_COMM.Get_rank()
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        assert _MPI_SIZE == Gx * Gy * Gz * Gt

        global _GPUID, _CUDA_BACKEND, _COMPUTE_CAPABILITY

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

        gpuid += _GPUID
        _GPUID = gpuid

        _CUDA_BACKEND = backend

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

        if "QUDA_RESOURCE_PATH" not in environ and resource_path is not None:
            environ["QUDA_RESOURCE_PATH"] = resource_path
        quda.initCommsGridQuda(4, [Gx, Gy, Gz, Gt])
        quda.initQuda(gpuid)
        atexit.register(quda.endQuda)
    else:
        warn("PyQuda is already initialized", RuntimeWarning)


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


def setGPUID(gpuid: int):
    global _GPUID
    _GPUID = gpuid


def getGPUID():
    return _GPUID


def getCUDABackend():
    return _CUDA_BACKEND


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY
