from typing import Callable, Literal, Sequence

import numpy
from numpy.typing import NDArray, DTypeLike

BackendType = Literal["numpy", "cupy", "dpnp", "torch"]
BackendTargetType = Literal["cpu", "cuda", "hip", "sycl"]

dpnp_device = None
dpnp_sycl_queue = None


def setDPNPDevice(device: str):
    import dpctl.tensor

    global dpnp_device, dpnp_sycl_queue
    dpnp_device = dpctl.tensor.Device.create_device(device)
    dpnp_sycl_queue = dpnp_device.sycl_queue


def backendDeviceAPI(backend: BackendType, backend_target: BackendTargetType):
    if backend == "numpy":
        if backend_target == "cpu":
            getDeviceCount: Callable[[], int] = lambda: 0x7FFFFFFF
            setDevice: Callable[[int], None] = lambda device: None
        else:
            raise ValueError(f"Array API backend {backend} does not support target {backend_target}")
    elif backend == "cupy":
        import cupy

        if backend_target == "cuda":
            assert not cupy.cuda.runtime.is_hip
            getDeviceCount: Callable[[], int] = lambda: cupy.cuda.runtime.getDeviceCount()
            setDevice: Callable[[int], None] = lambda device: cupy.cuda.Device(device).use()
        elif backend_target == "hip":
            assert cupy.cuda.runtime.is_hip
            getDeviceCount: Callable[[], int] = lambda: cupy.cuda.runtime.getDeviceCount()
            setDevice: Callable[[int], None] = lambda device: cupy.cuda.Device(device).use()
        else:
            raise ValueError(f"Array API backend {backend} does not support target {backend_target}")
    elif backend == "dpnp":
        import dpctl

        if backend_target == "cpu":
            getDeviceCount: Callable[[], int] = lambda: 0x7FFFFFFF
            setDevice: Callable[[int], None] = lambda device: setDPNPDevice("opencl:cpu")
        elif backend_target == "cuda":
            getDeviceCount: Callable[[], int] = lambda: dpctl.get_num_devices("cuda", "gpu")
            setDevice: Callable[[int], None] = lambda device: setDPNPDevice(f"cuda:gpu{device}")
        elif backend_target == "hip":
            getDeviceCount: Callable[[], int] = lambda: dpctl.get_num_devices("hip", "gpu")
            setDevice: Callable[[int], None] = lambda device: setDPNPDevice(f"hip:gpu:{device}")
        elif backend_target == "sycl":
            getDeviceCount: Callable[[], int] = lambda: dpctl.get_num_devices("level_zero", "gpu")
            setDevice: Callable[[int], None] = lambda device: setDPNPDevice(f"level_zero:gpu:{device}")
        else:
            raise ValueError(f"Array API backend {backend} does not support target {backend_target}")
    elif backend == "torch":
        import torch
        import torch.version

        if backend_target == "cpu":
            getDeviceCount: Callable[[], int] = lambda: 0x7FFFFFFF
            setDevice: Callable[[int], None] = lambda device: torch.set_default_device("cpu")
        elif backend_target == "cuda":
            assert torch.version.cuda is not None
            getDeviceCount: Callable[[], int] = lambda: torch.cuda.device_count()
            setDevice: Callable[[int], None] = lambda device: torch.set_default_device(f"cuda:{device}")
        elif backend_target == "hip":
            assert torch.version.hip is not None
            getDeviceCount: Callable[[], int] = lambda: torch.cuda.device_count()
            setDevice: Callable[[int], None] = lambda device: torch.set_default_device(f"cuda:{device}")
        elif backend_target == "sycl":
            assert torch.version.xpu is not None
            getDeviceCount: Callable[[], int] = lambda: torch.xpu.device_count()
            setDevice: Callable[[int], None] = lambda device: torch.set_default_device(f"xpu:{device}")
        else:
            raise ValueError(f"Array API backend {backend} does not support target {backend_target}")

    return getDeviceCount, setDevice


def arrayDType(dtype: DTypeLike, backend: BackendType) -> DTypeLike:
    if backend == "numpy":
        return numpy.dtype(dtype).type
    elif backend == "cupy":
        return numpy.dtype(dtype).type
    elif backend == "dpnp":
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
    elif backend == "dpnp":
        return numpy.ascontiguousarray(data.asnumpy())  # TODO: https://github.com/IntelPython/dpnp/issues/2568
    elif backend == "torch":
        return data.cpu().numpy()


def arrayHostCopy(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return data.copy()
    elif backend == "cupy":
        return data.get()
    elif backend == "dpnp":
        return numpy.ascontiguousarray(data.asnumpy())  # TODO: https://github.com/IntelPython/dpnp/issues/2568
    elif backend == "torch":
        return data.cpu().numpy()


def arrayDevice(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.asarray(data)
    elif backend == "cupy":
        import cupy

        return cupy.asarray(data)
    elif backend == "dpnp":
        import dpnp

        return dpnp.asarray(data, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.as_tensor(data)


def arrayCopy(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return data.copy()
    elif backend == "cupy":
        return data.copy()
    elif backend == "dpnp":
        return data.copy(sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        return data.clone()


def arrayIsContiguous(data, backend: BackendType) -> bool:
    if backend == "numpy":
        return data.flags.c_contiguous
    elif backend == "cupy":
        return data.flags.c_contiguous
    elif backend == "dpnp":
        return data.flags.c_contiguous
    elif backend == "torch":
        return data.is_contiguous()


def arrayAsContiguous(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.ascontiguousarray(data)
    elif backend == "cupy":
        import cupy

        return cupy.ascontiguousarray(data)
    elif backend == "dpnp":
        import dpnp

        return dpnp.ascontiguousarray(data, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        return data.contiguous()


def arrayLinalgNorm(data, backend: BackendType) -> float:
    if backend == "numpy":
        return numpy.linalg.norm(data).item()
    elif backend == "cupy":
        import cupy

        return cupy.linalg.norm(data).item()
    elif backend == "dpnp":
        import dpnp.linalg

        return dpnp.linalg.norm(data).item()
    elif backend == "torch":
        import torch

        return torch.linalg.norm(data).item()


def arrayZeros(shape: Sequence[int], dtype: DTypeLike, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.zeros(shape, dtype)
    elif backend == "cupy":
        import cupy

        return cupy.zeros(shape, dtype)
    elif backend == "dpnp":
        import dpnp

        return dpnp.zeros(shape, dtype=dtype, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.zeros(shape, dtype=dtype)


def arrayEmpty(shape: Sequence[int], dtype: DTypeLike, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.empty(shape, dtype)
    elif backend == "cupy":
        import cupy

        return cupy.empty(shape, dtype)
    elif backend == "dpnp":
        import dpnp

        return dpnp.empty(shape, dtype=dtype, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.empty(shape, dtype=dtype)


def arrayExp(data, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.exp(data)
    elif backend == "cupy":
        import cupy

        return cupy.exp(data)
    elif backend == "dpnp":
        import dpnp

        return dpnp.exp(data)
    elif backend == "torch":
        import torch

        return torch.exp(data)


def arrayIdentity(n: int, dtype: DTypeLike, backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.identity(n, dtype)
    elif backend == "cupy":
        import cupy

        return cupy.identity(n, dtype)
    elif backend == "dpnp":
        import dpnp

        return dpnp.identity(n, dtype, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.eye(n, dtype=dtype)


def arrayRandomRandom(size: Sequence[int], backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.random.random(size)
    elif backend == "cupy":
        import cupy

        return cupy.random.random(size, cupy.float64)
    elif backend == "dpnp":
        import dpnp.random

        return dpnp.random.random(size, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.rand(size, dtype=torch.float64)


def arrayRandomRandomComplex(size: Sequence[int], backend: BackendType) -> NDArray:
    if backend == "numpy":
        size = tuple(size[:-1]) + (2 * size[-1],)
        return numpy.random.random(size).view(numpy.complex128)
    elif backend == "cupy":
        import cupy

        size = tuple(size[:-1]) + (2 * size[-1],)
        return cupy.random.random(size, cupy.float64).view(cupy.complex128)
    elif backend == "dpnp":
        import dpnp.random

        size = tuple(size[:-1]) + (2 * size[-1],)
        return dpnp.random.random(size, sycl_queue=dpnp_sycl_queue).view(dpnp.complex128)
        # return dpnp.random.random(size, sycl_queue=dpnp_sycl_queue) + 1j * dpnp.random.random(
        #     size, sycl_queue=dpnp_sycl_queue
        # )  # ? dpnp<0.19.0
    elif backend == "torch":
        import torch

        size = tuple(size[:-1]) + (2 * size[-1],)
        return torch.rand(size, dtype=torch.float64).view(torch.complex128)


def arrayRandomNormal(loc: float, scale: float, size: Sequence[int], backend: BackendType) -> NDArray:
    if backend == "numpy":
        return numpy.random.normal(loc, scale, size)
    elif backend == "cupy":
        import cupy

        return cupy.random.normal(loc, scale, size, cupy.float64)
    elif backend == "dpnp":
        import dpnp.random

        return dpnp.random.normal(loc, scale, size, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.normal(loc, scale, size, dtype=torch.float64)


def arrayRandomNormalComplex(loc: float, scale: float, size: Sequence[int], backend: BackendType) -> NDArray:
    if backend == "numpy":
        size = tuple(size[:-1]) + (2 * size[-1],)
        return numpy.random.normal(loc, scale, size).view(numpy.complex128)
    elif backend == "cupy":
        import cupy

        size = tuple(size[:-1]) + (2 * size[-1],)
        return cupy.random.normal(loc, scale, size, cupy.float64).view(cupy.complex128)
    elif backend == "dpnp":
        import dpnp.random

        size = tuple(size[:-1]) + (2 * size[-1],)
        return dpnp.random.normal(loc, scale, size, sycl_queue=dpnp_sycl_queue).view(dpnp.complex128)
        # return dpnp.random.normal(loc, scale, size, sycl_queue=dpnp_sycl_queue) + 1j * dpnp.random.normal(
        #     loc, scale, size, sycl_queue=dpnp_sycl_queue
        # )  # ? dpnp<0.19.0
    elif backend == "torch":
        import torch

        size = tuple(size[:-1]) + (2 * size[-1],)
        return torch.normal(loc, scale, size, dtype=torch.float64).view(torch.complex128)


def arrayRandomGetState(backend: BackendType):
    if backend == "numpy":
        return numpy.random.get_state()
    elif backend == "cupy":
        import cupy

        return cupy.random.get_random_state()
    elif backend == "dpnp":
        import dpnp.random.dpnp_iface_random

        return dpnp.random.dpnp_iface_random._get_random_state(sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        return torch.random.get_rng_state()


def arrayRandomSetState(state, backend: BackendType):
    if backend == "numpy":
        numpy.random.set_state(state)
    elif backend == "cupy":
        import cupy

        cupy.random.set_random_state(state)
    elif backend == "dpnp":
        import dpnp
        import dpnp.random.dpnp_iface_random

        sycl_queue = dpnp.get_normalized_queue_device(sycl_queue=dpnp_sycl_queue)
        dpnp.random.dpnp_iface_random._dpnp_random_states[sycl_queue] = state
    elif backend == "torch":
        import torch

        torch.random.set_rng_state(state)


def arrayRandomSeed(seed: int, backend: BackendType):
    if backend == "numpy":
        numpy.random.seed(seed)
    elif backend == "cupy":
        import cupy

        cupy.random.seed(seed)
    elif backend == "dpnp":
        import dpnp.random

        dpnp.random.seed(seed, sycl_queue=dpnp_sycl_queue)
    elif backend == "torch":
        import torch

        torch.manual_seed(seed)
