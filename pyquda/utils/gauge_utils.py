from ..field import LatticeInfo, LatticeGauge


def gaussGauge(latt_info: LatticeInfo, seed: int):
    from ..pyquda import loadGaugeQuda, saveGaugeQuda, gaussGaugeQuda
    from ..core import getDslash

    gauge = LatticeGauge(latt_info, None)

    dslash = getDslash(latt_info.size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge.data_ptrs, dslash.gauge_param)
    dslash.gauge_param.use_resident_gauge = 1
    gaussGaugeQuda(seed, 1.0)
    saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)

    return gauge
