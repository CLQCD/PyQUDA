import os
import numpy as np

from check_pyquda import test_dir

from pyquda import init
from pyquda.utils import io

init(resource_path=".cache")

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))
gauge.fixingOVR(4, 1000, 1, 1.0, 2e-15, 1, 1)

land_gauge = io.readQIOGauge("coul_cfg.lime")
print(np.linalg.norm(land_gauge.data - gauge.data))
