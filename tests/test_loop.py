from check_pyquda import weak_field

from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T
from pyquda_utils.wilson_loop import wilson_loop

core.init(None, [4, 4, 4, 8], resource_path=".cache")
gauge = io.readQIOGauge(weak_field)
gauge.toDevice()

loop = gauge.loop([[[X, Y, -X, -Y]], [[Y, Z, -Y, -Z]], [[Z, X, -Z, -X]], [[T, -T, T, -T]]], [1.0])
loop_x = wilson_loop(gauge, [X, Y, -X, -Y])
loop_y = wilson_loop(gauge, [Y, Z, -Y, -Z])
loop_z = wilson_loop(gauge, [Z, X, -Z, -X])

print((loop[X] - loop_x).norm2() ** 0.5)
print((loop[Y] - loop_y).norm2() ** 0.5)
print((loop[Z] - loop_z).norm2() ** 0.5)
