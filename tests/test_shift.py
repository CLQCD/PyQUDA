from check_pyquda import weak_field

from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

core.init([1, 1, 1, 2], [4, 4, 4, 8], 1, 1.0, resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge.toDevice()
gauge_shift = gauge.shift([-1, -1, -1, -1], [X, Y, Z, T])

print(gauge[X].data[0, 0, 0, 0, 0])
print(gauge_shift[X].data[1, 0, 0, 0, 0])
print(gauge[Y].data[0, 0, 0, 0, 0])
print(gauge_shift[Y].data[1, 0, 0, 1, 0])
print(gauge[Z].data[0, 0, 0, 0, 0])
print(gauge_shift[Z].data[1, 0, 1, 0, 0])
print(gauge[T].data[0, 0, 0, 0, 0])
print(gauge_shift[T].data[1, 1, 0, 0, 0])

gauge_x = gauge[X]
gauge_x_shift_1_1 = gauge_x.shift(1, X).shift(1, X)
gauge_x_shift_2 = gauge_x.shift(2, X)
print((gauge_x_shift_1_1 - gauge_x_shift_2).norm2() ** 0.5)

gauge_t = gauge[T]
gauge_t_shift_1_2 = gauge_x.shift(1, T).shift(2, T)
gauge_t_shift_2_1 = gauge_x.shift(2, T).shift(1, T)
gauge_t_shift_3 = gauge_x.shift(3, T)
print((gauge_t_shift_1_2 - gauge_t_shift_3).norm2() ** 0.5)
print((gauge_t_shift_2_1 - gauge_t_shift_3).norm2() ** 0.5)
