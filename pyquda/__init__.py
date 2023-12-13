from typing import List, Literal, NamedTuple
from warnings import warn

from . import pyquda as quda


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_MPI_COMM = None
_MPI_SIZE: int = 1
_MPI_RANK: int = 0
_GRID_SIZE: List[int] = [1, 1, 1, 1]
_GRID_COORD: List[int] = [0, 0, 0, 0]
_GPUID: int = 0
_CUDA_BACKEND: Literal["cupy", "torch"] = "cupy"
_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def getRankFromCoord(coord: List[int]):
    Gx, Gy, Gz, Gt = _GRID_SIZE
    return ((coord[0] * Gy + coord[1]) * Gz + coord[2]) * Gt + coord[3]


def getCoordFromRank(rank: int):
    Gx, Gy, Gz, Gt = _GRID_SIZE
    return [rank // Gt // Gz // Gy, rank // Gt // Gz % Gy, rank // Gt % Gz, rank % Gt]


def init(grid_size: List[int] = None):
    """
    Initialize MPI along with the QUDA library.

    If grid_size is None, MPI will not applied.
    """
    global _MPI_COMM, _MPI_SIZE, _MPI_RANK, _GRID_SIZE, _GRID_COORD
    if _MPI_COMM is None:
        import atexit
        from .pyquda import initCommsGridQuda, initQuda, endQuda

        if _CUDA_BACKEND == "cupy":
            from cupy import cuda
            from . import malloc_pyquda
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
            _MPI_COMM = 0
        else:
            _MPI_COMM: MPI.Comm = MPI.COMM_WORLD
            _MPI_SIZE = _MPI_COMM.Get_size()
            _MPI_RANK = _MPI_COMM.Get_rank()
            _GRID_SIZE = [Gx, Gy, Gz, Gt]
            _GRID_COORD = getCoordFromRank(_MPI_RANK)

            hostname = gethostname()
            hostname_recv_buf = _MPI_COMM.allgather(hostname)
            for i in range(_MPI_RANK):
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

        assert Gx * Gy * Gz * Gt == _MPI_SIZE
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


def setCUDABackend(backend: Literal["cupy", "torch"]):
    assert backend in ["cupy", "torch"], f"Unsupported backend {backend}"
    global _CUDA_BACKEND
    _CUDA_BACKEND = backend


def getCUDABackend():
    return _CUDA_BACKEND


def getCUDAComputeCapability():
    return _COMPUTE_CAPABILITY


def gather(data, axes: List[int] = [-1, -1, -1, -1], mode: str = None, root: int = 0):
    import numpy

    dtype = data.dtype
    Lt, Lz, Ly, Lx = [data.shape[axis] if axis != -1 else 1 for axis in axes]
    Gx, Gy, Gz, Gt = _GRID_SIZE
    collect = tuple([axis for axis in axes if axis != -1])
    if collect == ():
        collect = (0, -1)
    process = tuple([collect[0] + d for d in range(4) if axes[d] == -1])
    prefix = data.shape[: collect[0]]
    suffix = data.shape[collect[-1] + 1 :]
    Nroots = Lx * Ly * Lz * Lt
    Nprefix = int(numpy.prod(prefix))
    Nsuffix = int(numpy.prod(suffix))
    sendbuf = data.reshape(Nprefix * Nroots * Nsuffix).get()
    if _MPI_RANK == root:
        recvbuf = numpy.zeros((_MPI_SIZE, Nprefix * Nroots * Nsuffix), dtype)
    else:
        recvbuf = None
    if _MPI_COMM is not None:
        _MPI_COMM.Gatherv(sendbuf, recvbuf, root)
    else:
        recvbuf[0] = sendbuf
    if _MPI_RANK == root:
        data = numpy.zeros((Nprefix, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nsuffix), dtype)
        for i in range(_MPI_SIZE):
            gt = i % Gt
            gz = i // Gt % Gz
            gy = i // Gt // Gz % Gy
            gx = i // Gt // Gz // Gy
            data[
                :, gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
            ] = recvbuf[i].reshape(Nprefix, Lt, Lz, Ly, Lx, Nsuffix)
        data = data.reshape(*prefix, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, *suffix)

        mode = "sum" if mode is None else mode
        if mode.lower() == "sum":
            data = data.sum(process)
        elif mode.lower() == "mean":
            data = data.mean(process)
        else:
            raise NotImplementedError(f"{mode} mode in mpi.gather not implemented yet.")
        return data
    else:
        return None
