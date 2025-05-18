from check_pyquda import weak_field

from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

core.init([1, 1, 1, 1], [4, 4, 4, 8], 1, 1.0, resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge.toDevice()
gauge_shift = gauge.shift([-X, -Y, -Z, -T])

print(gauge[X].data[0, 0, 0, 0, 0])
print(gauge_shift[X].data[1, 0, 0, 0, 0])
print(gauge[Y].data[0, 0, 0, 0, 0])
print(gauge_shift[Y].data[1, 0, 0, 1, 0])
print(gauge[Z].data[0, 0, 0, 0, 0])
print(gauge_shift[Z].data[1, 0, 1, 0, 0])
print(gauge[T].data[0, 0, 0, 0, 0])
print(gauge_shift[T].data[1, 1, 0, 0, 0])
