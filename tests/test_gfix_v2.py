from check_pyquda import weak_field, data

from pyquda_utils import core, io
from pyquda import quda

core.init(resource_path=".cache/gfix")


gauge = io.readQIOGauge(weak_field)
rot = core.LatticeRotation(gauge.latt_info)

fix_param = quda.QudaGaugeFixParam()
fix_param.tol = 2e-15
fix_param.maxiter = 1000
fix_param.dir_ignore = 4
fix_param.omega = 1.3
fix_param.reunit_interval = 1
fix_param.verbose_interval = 100
fix_param.compute_theta = True
fix_param.use_theta = False

with gauge.use() as dirac:
    dirac.gauge_param.use_resident_gauge = 1
    dirac.gauge_param.make_resident_gauge = 0
    dirac.gauge_param.return_result_gauge = 1
    quda.performGaugeFixQuda(rot.data_ptrs, gauge.data_ptrs, gauge.gauge_dirac.gauge_param, fix_param)

    gauge.gauge_dirac.gauge_param.use_resident_gauge = 1
    gauge.gauge_dirac.gauge_param.make_resident_gauge = 0
    gauge.gauge_dirac.gauge_param.return_result_gauge = 1
    quda.performGaugeRotateQuda(rot.data_ptrs, gauge.data_ptrs, gauge.gauge_dirac.gauge_param)

land_gauge = io.readQIOGauge(data("coul_cfg.lime"))
print((land_gauge - gauge).norm2() ** 0.5)
