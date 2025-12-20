import logging
from os import environ
from sys import stdout
from typing import Generator, List, Literal, Optional, Sequence, Tuple, Type, Union, get_args

import numpy
from numpy.typing import NDArray, DTypeLike
from mpi4py import MPI
from mpi4py.util import dtlib

GridMapType = Literal["default", "reversed", "shared"]
from .array import BackendType, BackendTargetType, backendDeviceAPI


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
        self.logger.level = logging.INFO
        self.logger.handlers = [stdout_handler, stderr_handler]

    def debug(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.debug(msg)

    def info(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.info(msg)

    def warning(self, msg: str, category: Type[Warning]):
        if _MPI_RANK == self.root:
            self.logger.warning(msg, exc_info=category(msg), stack_info=True)

    def error(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.error(msg, exc_info=category(msg), stack_info=True)

    def critical(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.critical(msg, exc_info=category(msg), stack_info=True)
        raise category(msg)


_MPI_LOGGER: _MPILogger = _MPILogger()
_MPI_COMM: MPI.Intracomm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_MAP: GridMapType = "default"
"""For MPI, the default node mapping is lexicographical with t varying fastest."""
_GRID_SIZE: Optional[Tuple[int, ...]] = None
_GRID_COORD: Optional[Tuple[int, ...]] = None
_SHARED_RANK_LIST: Optional[List[int]] = None
_ARRAY_BACKEND: BackendType = "cupy"
_ARRAY_BACKEND_TARGET: BackendTargetType = "cuda"
_ARRAY_DEVICE: int = -1


def _defaultRankFromCoord(coords: Sequence[int], dims: Sequence[int]) -> int:
    rank = 0
    for coord, dim in zip(coords, dims):
        rank = rank * dim + coord
    return rank


def _defaultCoordFromRank(rank: int, dims: Sequence[int]) -> List[int]:
    coords = []
    for dim in dims[::-1]:
        coords.append(rank % dim)
        rank = rank // dim
    return coords[::-1]


def getRankFromCoord(grid_coord: List[int]) -> int:
    grid_size = getGridSize()
    if len(grid_coord) != len(grid_size):
        _MPI_LOGGER.critical(
            f"Grid coordinate {grid_coord} and grid size {grid_size} must have the same dimension",
            ValueError,
        )

    if _GRID_MAP == "default":
        mpi_rank = _defaultRankFromCoord(grid_coord, grid_size)
    elif _GRID_MAP == "reversed":
        mpi_rank = _defaultRankFromCoord(grid_coord[::-1], grid_size[::-1])
    elif _GRID_MAP == "shared":
        assert _SHARED_RANK_LIST is not None
        mpi_rank = _SHARED_RANK_LIST.index(_defaultRankFromCoord(grid_coord, grid_size))
    else:
        _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)
    return mpi_rank


def getCoordFromRank(mpi_rank: int) -> List[int]:
    grid_size = getGridSize()

    if _GRID_MAP == "default":
        grid_coord = _defaultCoordFromRank(mpi_rank, grid_size)
    elif _GRID_MAP == "reversed":
        grid_coord = _defaultCoordFromRank(mpi_rank, grid_size[::-1])[::-1]
    elif _GRID_MAP == "shared":
        assert _SHARED_RANK_LIST is not None
        grid_coord = _defaultCoordFromRank(_SHARED_RANK_LIST[mpi_rank], grid_size)
    else:
        _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)
    return grid_coord


def getNeighbourRank():
    grid_size = getGridSize()
    grid_coord = getGridCoord()

    neighbour_forward = []
    neighbour_backward = []
    for d in range(len(grid_size)):
        g, G = grid_coord[d], grid_size[d]
        grid_coord[d] = (g + 1) % G
        neighbour_forward.append(getRankFromCoord(grid_coord))
        grid_coord[d] = (g - 1) % G
        neighbour_backward.append(getRankFromCoord(grid_coord))
        grid_coord[d] = g
    return neighbour_forward + neighbour_backward


def getSublatticeSize(latt_size: Sequence[int], force_even: bool = True):
    grid_size = getGridSize()
    if len(latt_size) != len(grid_size):
        _MPI_LOGGER.critical(
            f"Lattice size {latt_size} and grid size {grid_size} must have the same dimension",
            ValueError,
        )
    if force_even:
        if not all([(GL % (2 * G) == 0 or GL * G == 1) for GL, G in zip(latt_size, grid_size)]):
            _MPI_LOGGER.critical(
                f"lattice size {latt_size} must be divisible by gird size {grid_size}, "
                "and sublattice size must be even in all directions for consistant even-odd preconditioning, "
                "otherwise lattice size and grid size for this direction must be 1",
                ValueError,
            )
    else:
        if not all([(GL % G == 0) for GL, G in zip(latt_size, grid_size)]):
            _MPI_LOGGER.critical(
                f"lattice size {latt_size} must be divisible by gird size {grid_size}",
                ValueError,
            )
    return [GL // G for GL, G in zip(latt_size, grid_size)]


def _composition(n: int, d: int):
    """
    Writing n as the sum of d natural numbers
    """
    addend: List[List[int]] = []
    i = [0 for _ in range(d - 1)] + [n] + [0]
    while i[0] <= n:
        addend.append([i[s] - i[s - 1] for s in range(d)])
        i[d - 2] += 1
        for s in range(d - 2, 0, -1):
            if i[s] == n + 1:
                i[s] = 0
                i[s - 1] += 1
        for s in range(1, d - 1, 1):
            if i[s] < i[s - 1]:
                i[s] = i[s - 1]
    return addend


def _factorization(k: int, d: int):
    """
    Writing k as the product of d positive numbers
    """
    prime_factor: List[List[List[int]]] = []
    for p in range(2, int(k**0.5) + 1):
        n = 0
        while k % p == 0:
            n += 1
            k //= p
        if n != 0:
            prime_factor.append([[p**a for a in addend] for addend in _composition(n, d)])
    if k != 1:
        prime_factor.append([[k**a for a in addend] for addend in _composition(1, d)])
    return prime_factor


def _partition(
    factor: Union[int, List[List[List[int]]]],
    sublatt_size: List[int],
    grid_size: Optional[List[int]] = None,
    idx: int = 0,
) -> Generator[List[int], None, None]:
    if idx == 0:
        assert isinstance(factor, int) and grid_size is None
        grid_size = [1 for _ in range(len(sublatt_size))]
        factor = _factorization(factor, len(sublatt_size))
    assert isinstance(factor, list) and grid_size is not None
    if idx == len(factor):
        yield grid_size
    else:
        for factor_size in factor[idx]:
            for L, x in zip(sublatt_size, factor_size):
                if L % x != 0:
                    break
            else:
                yield from _partition(
                    factor,
                    [L // f for L, f in zip(sublatt_size, factor_size)],
                    [G * f for G, f in zip(grid_size, factor_size)],
                    idx + 1,
                )


def getDefaultGrid(mpi_size: int, latt_size: Sequence[int], evenodd: bool = True):
    Lx, Ly, Lz, Lt = latt_size
    latt_vol = Lx * Ly * Lz * Lt
    latt_surf = [latt_vol // latt_size[dir] for dir in range(4)]
    min_comm, min_grid = latt_vol, []
    assert latt_vol % mpi_size == 0, "lattice volume must be divisible by MPI size"
    if evenodd:
        assert (
            Lx % 2 == 0 and Ly % 2 == 0 and Lz % 2 == 0 and Lt % 2 == 0
        ), "lattice size must be even in all directions for even-odd preconditioning"
        partition = _partition(mpi_size, [Lx // 2, Ly // 2, Lz // 2, Lt // 2])
    else:
        partition = _partition(mpi_size, [Lx, Ly, Lz, Lt])
    for grid_size in partition:
        comm = [latt_surf[dir] * grid_size[dir] for dir in range(4) if grid_size[dir] > 1]
        if sum(comm) < min_comm:
            min_comm, min_grid = sum(comm), [grid_size]
        elif sum(comm) == min_comm:
            min_grid.append(grid_size)
    if min_grid == []:
        _MPI_LOGGER.critical(
            f"Cannot get proper grid for lattice size {latt_size} with {mpi_size} MPI processes", ValueError
        )
    return min(min_grid)


def setSharedRankList(grid_size: Sequence[int]):
    global _SHARED_RANK_LIST
    shared_comm = _MPI_COMM.Split_type(MPI.COMM_TYPE_SHARED)
    shared_size = shared_comm.Get_size()
    shared_rank = shared_comm.Get_rank()
    shared_root = shared_comm.bcast(_MPI_RANK)
    node_rank = _MPI_COMM.allgather(shared_root).index(shared_root)
    assert _MPI_SIZE % shared_size == 0
    node_grid_size = [G for G in grid_size]
    shared_grid_size = [1 for _ in grid_size]
    dim, last_dim = 0, len(grid_size) - 1
    while shared_size > 1:
        for prime in [2, 3, 5]:
            if node_grid_size[dim] % prime == 0 and shared_size % prime == 0:
                node_grid_size[dim] //= prime
                shared_grid_size[dim] *= prime
                shared_size //= prime
                last_dim = dim
                break
        else:
            if last_dim == dim:
                _MPI_LOGGER.critical("GlobalSharedMemory::GetShmDims failed", ValueError)
        dim = (dim + 1) % len(grid_size)
    grid_coord = [
        n * S + s
        for n, S, s in zip(
            _defaultCoordFromRank(node_rank, node_grid_size),
            shared_grid_size,
            _defaultCoordFromRank(shared_rank, shared_grid_size),
        )
    ]
    _SHARED_RANK_LIST = _MPI_COMM.allgather(_defaultRankFromCoord(grid_coord, grid_size))


def initGrid(
    grid_map: GridMapType = "default",
    grid_size: Optional[Sequence[int]] = None,
    latt_size: Optional[Sequence[int]] = None,
    evenodd: bool = True,
):
    global _GRID_MAP, _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        if grid_map not in get_args(GridMapType):
            _MPI_LOGGER.critical(f"Unsupported grid mapping {grid_map}", ValueError)
        _GRID_MAP = grid_map

        if grid_size is None and latt_size is not None:
            grid_size = getDefaultGrid(_MPI_SIZE, latt_size, evenodd)
            _MPI_LOGGER.info(f"Using lattice size {latt_size} to set the default grid size {grid_size}")
        if grid_size is None:
            grid_size = [1, 1, 1, 1]

        if grid_map == "shared":
            setSharedRankList(grid_size)

        _GRID_SIZE = tuple(grid_size)
        _GRID_COORD = tuple(getCoordFromRank(_MPI_RANK))
        _MPI_LOGGER.info(f"Using grid size {_GRID_SIZE}")
    else:
        _MPI_LOGGER.warning("Grid is already initialized", RuntimeWarning)


def initDevice(
    backend: BackendType = "cupy",
    backend_target: BackendTargetType = "cuda",
    device: int = -1,
    enable_mps: bool = False,
):
    global _ARRAY_BACKEND, _ARRAY_BACKEND_TARGET, _ARRAY_DEVICE
    if _ARRAY_DEVICE < 0:
        from platform import node as gethostname

        if backend not in get_args(BackendType):
            _MPI_LOGGER.critical(f"Unsupported Array API backend {backend}", ValueError)
        if backend == "numpy":
            backend_target = "cpu"
        getDeviceCount, setDevice = backendDeviceAPI(backend, backend_target)
        _MPI_LOGGER.info(f"Using Array API backend {backend} with target {backend_target}")
        _ARRAY_BACKEND = backend
        _ARRAY_BACKEND_TARGET = backend_target

        # quda/include/communicator_quda.h
        # determine which GPU this rank will use
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if device < 0:
            device_count = getDeviceCount()
            if device_count == 0:
                _MPI_LOGGER.critical("No devices found", RuntimeError)

            # We initialize gpuid if it's still negative.
            device = 0
            for i in range(_MPI_RANK):
                if hostname == hostname_recv_buf[i]:
                    device += 1

            if device >= device_count:
                if enable_mps or environ.get("QUDA_ENABLE_MPS") == "1":
                    device %= device_count
                    print(f"MPS enabled, rank={_MPI_RANK:3d} -> gpu={device}")
                else:
                    _MPI_LOGGER.critical(f"Too few GPUs available on {hostname}", RuntimeError)
        _ARRAY_DEVICE = device

        setDevice(device)
    else:
        _MPI_LOGGER.warning("Device is already initialized", RuntimeWarning)


def isGridInitialized():
    return _GRID_SIZE is not None


def isDeviceInitialized():
    return _ARRAY_DEVICE >= 0


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


def getGridMap():
    return _GRID_MAP


def getGridSize():
    if _GRID_SIZE is None:
        _MPI_LOGGER.critical("Grid is not initialized", RuntimeError)
    return list(_GRID_SIZE)


def getGridCoord():
    if _GRID_COORD is None:
        _MPI_LOGGER.critical("Grid is not initialized", RuntimeError)
    return list(_GRID_COORD)


def getArrayBackend():
    return _ARRAY_BACKEND


def getArrayBackendTarget():
    return _ARRAY_BACKEND_TARGET


def getArrayDevice():
    return _ARRAY_DEVICE


def getSubarray(dtype: DTypeLike, shape: Sequence[int], axes: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    grid = getGridSize()
    coord = getGridCoord()
    for j, i in enumerate(axes):
        sizes[i] *= grid[j]
        starts[i] *= coord[j]

    dtype_str = numpy.dtype(dtype).str
    native_dtype_str = dtype_str if not dtype_str.startswith(">") else dtype_str.replace(">", "<")
    return native_dtype_str, dtlib.from_numpy_dtype(native_dtype_str).Create_subarray(sizes, subsizes, starts)


def readMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray:
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_RDONLY)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Read_all(buf)
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


def readMPIFileInChunks(
    filename: str, dtype: DTypeLike, offset: int, count: int, shape: Sequence[int], axes: Sequence[int]
) -> Generator[Tuple[int, NDArray], None, None]:
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_RDONLY)
    filetype.Commit()
    for i in range(count):
        fh.Set_view(disp=offset + i * _MPI_SIZE * filetype.size, filetype=filetype)
        fh.Read_all(buf)
        yield i, buf.view(dtype)
    filetype.Free()
    fh.Close()


def writeMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray):
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = buf.view(native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()


def writeMPIFileInChunks(
    filename: str, dtype: DTypeLike, offset: int, count: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray
):
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = buf.view(native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype.Commit()
    for i in range(count):
        fh.Set_view(disp=offset + i * _MPI_SIZE * filetype.size, filetype=filetype)
        yield i  # Waiting for buf
        fh.Write_all(buf)
    filetype.Free()
    fh.Close()
