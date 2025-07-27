from typing import Generic, TypeVar

from numpy.typing import NDArray

T = TypeVar("T")

class Pointer(Generic[T]):
    def __init__(self, dtype: str): ...

class Pointers(Pointer[T]):
    def __init__(self, dtype: str, n1: int): ...

class Pointerss(Pointer[T]):
    def __init__(self, dtype: str, n1: int, n2: int): ...

def ndarrayPointer(ndarray: NDArray, as_void: bool = False) -> Pointer: ...
