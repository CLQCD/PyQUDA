from typing import List

from .core import LatticeGauge, LatticeFermion


def wilson_loop(gauge: LatticeGauge, path: List[int]):
    assert gauge.latt_info.Nd == 4
    fake_link = LatticeFermion(gauge.latt_info)
    link = LatticeGauge(gauge.latt_info, 1)
    link.pack(0, fake_link)
    gauge.gauge_dirac.loadGauge(gauge)
    for mu in path[::-1]:
        assert 0 <= mu < 2 * gauge.latt_info.Nd
        fake_link = gauge.gauge_dirac.covDev(fake_link, mu)
    gauge.gauge_dirac.freeGauge()
    link.unpack(0, fake_link)
    return link[0]
