from check_pyquda import weak_field

from pyquda import init
from pyquda_utils import io

init([1, 1, 1, 1], [4, 4, 4, 8], 1, 1.0, resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge.toDevice()
gauge_shift = gauge.shift([4, 5, 6, 7])

print(gauge.data[0, 0, 0, 0, 0, 0])
print(gauge_shift.data[0, 1, 0, 0, 0, 0])
print(gauge.data[1, 0, 0, 0, 0, 0])
print(gauge_shift.data[1, 1, 0, 0, 1, 0])
print(gauge.data[2, 0, 0, 0, 0, 0])
print(gauge_shift.data[2, 1, 0, 1, 0, 0])
print(gauge.data[3, 0, 0, 0, 0, 0])
print(gauge_shift.data[3, 1, 1, 0, 0, 0])
