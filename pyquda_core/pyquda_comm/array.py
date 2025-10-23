from typing import Any, Callable, Dict, Literal, Optional, Sequence

import numpy
from numpy.typing import NDArray, DTypeLike

BackendType = Literal["numpy", "cupy", "torch"]
TorchBackendType = Literal["cuda", "xpu"]


def arrayDeviceAPI(backend: BackendType, torch_backend: Optional[TorchBackendType] = None):
    if backend == "numpy":
        getDeviceCount: Callable[[], int] = lambda: 0x7FFFFFFF
        getDeviceProperties: Callable[[int], Dict[str, Any]] = lambda device: {"major": 0, "minor": 0}
        setDevice: Callable[[int], None] = lambda device: None
        isCUDA: bool = False
    elif backend == "cupy":
        import cupy
        from cupy.cuda.runtime import getDeviceCount
        from cupy.cuda.runtime import getDeviceProperties
        from cupy.cuda.runtime import is_hip

        setDevice: Callable[[int], None] = lambda device: cupy.cuda.Device(device).use()
        isCUDA: bool = not is_hip
    elif backend == "torch":
        import torch
        if torch_backend == "cuda":
            from torch.cuda import device_count as getDeviceCount
            from torch.cuda import get_device_properties as getDeviceProperties
            from torch.version import hip

            setDevice: Callable[[int], None] = lambda device: torch.set_default_device(f"cuda:{device}")
            isCUDA: bool = hip is None
        elif torch_backend == "xpu":
            from torch.xpu import device_count as getDeviceCount
            from torch.xpu import get_device_properties as getDeviceProperties

            setDevice: Callable[[int], None] = lambda device: torch.set_default_device(f"xpu:{device}")
            isCUDA: bool = False

    return getDeviceCount, getDeviceProperties, setDevice, isCUDA


def arrayDType(dtype: DTypeLike, backend: BackendType) -> DTypeLike:
    if backend == "numpy":
        return numpy.dtype(dtype).type
    elif backend == "cupy":
        return numpy.dtype(dtype).type
    elif backend == "torch":
        import torch

        # from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict

        # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
        numpy_to_torch_dtype_dict = {
            numpy.bool_: torch.bool,
            numpy.uint8: torch.uint8,
            numpy.uint16: torch.uint16,
            numpy.uint32: torch.uint32,
            numpy.uint64: torch.uint64,
            numpy.int8: torch.int8,
            numpy.int16: torch.int16,
            numpy.int32: torch.int32,
            numpy.int64: torch.int64,
            numpy.float16: torch.float16,
            numpy.float32: torch.float32,
            numpy.float64: torch.float64,
            numpy.complex64: torch.complex64,
            numpy.complex128: torch.complex128,
        }
        return numpy_to_torch_dtype_dict[numpy.dtype(dtype).type]


def arrayHost(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.asarray(data)
    elif backend == "cupy":
        return data.get()
    elif backend == "torch":
        return data.cpu().numpy()


def arrayHostCopy(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return data.copy()
    elif backend == "cupy":
        return data.get()
    elif backend == "torch":
        return data.cpu().numpy()


def arrayDevice(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.asarray(data)
    elif backend == "cupy":
        import cupy

        return cupy.asarray(data)
    elif backend == "torch":
        import torch

        return torch.as_tensor(data)


def arrayCopy(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return data.copy()
    elif backend == "cupy":
        return data.copy()
    elif backend == "torch":
        return data.clone()


def arrayIsContiguous(data, backend: BackendType) -> bool:
    if backend == "numpy":
        return data.flags.c_contiguous
    elif backend == "cupy":
        return data.flags.c_contiguous
    elif backend == "torch":
        return data.is_contiguous()


def arrayContiguous(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.ascontiguousarray(data)
    elif backend == "cupy":
        import cupy

        return cupy.ascontiguousarray(data)
    elif backend == "torch":
        return data.contiguous()


def arrayNorm2(data, backend: BackendType) -> float:
    if backend == "numpy":
        return numpy.linalg.norm(data).item() ** 2
    elif backend == "cupy":
        import cupy

        return cupy.linalg.norm(data).item() ** 2
    elif backend == "torch":
        import torch

        return torch.linalg.norm(data).item() ** 2


def arrayZeros(shape: Sequence[int], dtype: DTypeLike, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.zeros(shape, dtype)
    elif backend == "cupy":
        import cupy

        return cupy.zeros(shape, dtype)
    elif backend == "torch":
        import torch

        return torch.zeros(shape, dtype=dtype)


def arrayExp(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.exp(data)
    elif backend == "cupy":
        import cupy

        return cupy.exp(data)
    elif backend == "torch":
        import torch

        return torch.exp(data)


def arrayIdentity(n: int, dtype: DTypeLike, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.identity(n, dtype)
    elif backend == "cupy":
        import cupy

        return cupy.identity(n, dtype)
    elif backend == "torch":
        import torch

        return torch.eye(n, dtype=dtype)


def arrayRandom(size: Sequence[int], backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.random.random(size)
    elif backend == "cupy":
        import cupy

        return cupy.random.random(size, cupy.float64)
    elif backend == "torch":
        import torch

        return torch.rand(size, dtype=torch.float64)


def arrayNormal(loc: float, scale: float, size: Sequence[int], backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.random.normal(loc, scale, size)
    elif backend == "cupy":
        import cupy

        return cupy.random.normal(loc, scale, size, cupy.float64)
    elif backend == "torch":
        import torch

        return torch.normal(loc, scale, size, dtype=torch.float64)


def arrayRandomGetState(backend: BackendType):
    if backend == "numpy":
        return numpy.random.get_state()
    elif backend == "cupy":
        import cupy

        return cupy.random.get_random_state()
    elif backend == "torch":
        import torch

        return torch.random.get_rng_state()


def arrayRandomSetState(state, backend: BackendType):
    if backend == "numpy":
        numpy.random.set_state(state)
    elif backend == "cupy":
        import cupy

        cupy.random.set_random_state(state)
    elif backend == "torch":
        import torch

        torch.random.set_rng_state(state)


def arrayRandomSeed(seed: int, backend: BackendType):
    if backend == "numpy":
        numpy.random.seed(seed)
    elif backend == "cupy":
        import cupy

        cupy.random.seed(seed)
    elif backend == "torch":
        import torch

        torch.manual_seed(seed)
