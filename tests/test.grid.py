from typing import List, Tuple


def composition4(n):
    """
    Writing n as the sum of 4 natural numbers
    """
    addend: List[Tuple[int, int, int, int]] = []
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            for k in range(j + 1, n + 3):
                x, y, z, t = i, j - i - 1, k - j - 1, n + 3 - k - 1
                addend.append((x, y, z, t))
    return addend


def factorization4(k: int):
    """
    Writing k as the product of 4 positive numbers
    """
    prime_factor: List[List[Tuple[int, int, int, int]]] = []
    for p in range(2, int(k**0.5) + 1):
        n = 0
        while k % p == 0:
            n += 1
            k //= p
        if n != 0:
            prime_factor.append([(p**x, p**y, p**z, p**t) for x, y, z, t in composition4(n)])
    if k != 1:
        prime_factor.append([(k**x, k**y, k**z, k**t) for x, y, z, t in composition4(1)])
    return prime_factor


def allGrid(factor: List[List[Tuple[int, int, int, int]]], idx: int, sublatt_size: List[int], grid_size: List[int]):
    if idx == 0:
        factor = factorization4(factor)
    if idx == len(factor):
        yield grid_size
    else:
        Lx, Ly, Lz, Lt = sublatt_size
        Gx, Gy, Gz, Gt = grid_size
        for x, y, z, t in factor[idx]:
            if Lx % x == 0 and Ly % y == 0 and Lz % z == 0 and Lt % x == 0:
                yield from allGrid(
                    factor, idx + 1, [Lx // x, Ly // y, Lz // z, Lt // t], [Gx * x, Gy * y, Gz * z, Gt * t]
                )


def getDefaultGrid(mpi_size: int, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    latt_vol = Lx * Ly * Lz * Lt
    latt_surf = [latt_vol // latt_size[dir] for dir in range(4)]
    min_comm, min_grid = latt_vol, []
    assert latt_vol % mpi_size == 0
    for grid_size in allGrid(mpi_size, 0, latt_size, [1, 1, 1, 1]):
        comm = [latt_surf[dir] * grid_size[dir] for dir in range(4) if grid_size[dir] > 1]
        if sum(comm) < min_comm:
            min_comm, min_grid = sum(comm), [grid_size]
        elif sum(comm) == min_comm:
            min_grid.append(grid_size)
    return min_grid


print(getDefaultGrid(32, [48, 48, 48, 144]))
