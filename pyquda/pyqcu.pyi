from pointer import Pointer

class QcuParam:
    def __init__(self) -> None: ...

    lattice_size: int

def dslashQcu(fermion_out: Pointer, fermion_in: Pointer, gauge: Pointer, param: QcuParam, parity: int) -> None: ...
