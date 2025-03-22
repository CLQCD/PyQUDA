from typing import Generic, TypeVar

_DataType = TypeVar("_DataType")

class Pointer(Generic[_DataType]):
    def __init__(self, dtype: str): ...

class Pointers(Pointer[_DataType]):
    def __init__(self, dtype: str, n1: int): ...

class Pointerss(Pointer[_DataType]):
    def __init__(self, dtype: str, n1: int, n2: int): ...

def ndarrayPointer(ndarray, as_void: bool = False) -> Pointer: ...
