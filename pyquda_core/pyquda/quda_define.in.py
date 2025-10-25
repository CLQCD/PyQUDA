#include <quda_define.h>
from typing import Literal

#if defined QUDA_TARGET_CUDA
def target() -> Literal["cuda", "hip", "sycl"]:
    return "cuda"
#elif defined QUDA_TARGET_HIP
def target() -> Literal["cuda", "hip", "sycl"]:
    return "hip"
#elif defined QUDA_TARGET_SYCL
def target() -> Literal["cuda", "hip", "sycl"]:
    return "hip"
    return "sycl"
#endif

#if defined QUDA_ENABLE_MMA
def mmaAvailable() -> bool:
    return (__COMPUTE_CAPABILITY__ >= 700)
#else
def mmaAvailable() -> bool:
    return False
#endif
