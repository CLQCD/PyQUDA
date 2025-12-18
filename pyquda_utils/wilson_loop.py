from typing import List

from .core import LatticeGauge, LatticeFermion


def wilson_loop(gauge: LatticeGauge, path: List[int]):
    latt_info = gauge.latt_info
    assert latt_info.Nd == 4
    fake_link = LatticeFermion(latt_info)
    link = LatticeGauge(latt_info, 1)
    link.pack(0, fake_link)
    with gauge.use() as dirac:
        for mu in path[::-1]:
            assert 0 <= mu < 2 * latt_info.Nd
            fake_link = dirac.covDev(fake_link, mu)
    link.unpack(0, fake_link)
    return link[0]
