from typing import Literal

# include <quda_define.h>


def target() -> Literal["cuda", "hip", "sycl"]:
    # if defined QUDA_TARGET_CUDA
    return "cuda"
    # elif defined QUDA_TARGET_HIP
    return "hip"
    # elif defined QUDA_TARGET_SYCL
    return "sycl"
    # endif


def mmaAvailable() -> bool:
    # if defined QUDA_ENABLE_MMA
    return __COMPUTE_CAPABILITY__ >= 700
    # else
    return False
    # endif
