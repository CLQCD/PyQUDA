# flake8: noqa

from typing import Optional, Sequence
from pyquda_comm import initGrid, initDevice


def init(grid_size: Optional[Sequence[int]], latt_size: Optional[Sequence[int]] = None):
    initGrid("default", grid_size, latt_size, False)
    initDevice("numpy")


from .io_general import (
    IOGeneral,
    read as readIOGeneral,
    write as writeIOGeneral,
)
from .lime import (
    Lime,
)
from .npy import (
    readGauge as readNPYGauge,
    writeGauge as writeNPYGauge,
    readPropagator as readNPYPropagator,
    writePropagator as writeNPYPropagator,
)
from .chroma import (
    readQIOGauge as readChromaQIOGauge,
    readQIOPropagator as readChromaQIOPropagator,
)
from .ildg import (
    readGauge as readILDGGauge,
    readBinGauge as readILDGBinGauge,
)
from .milc import (
    readGauge as readMILCGauge,
    writeGauge as writeMILCGauge,
    readQIOPropagator as readMILCQIOPropagator,
)
from .kyu import (
    readGauge as readKYUGauge,
    writeGauge as writeKYUGauge,
    readPropagator as readKYUPropagator,
    writePropagator as writeKYUPropagator,
)
from .xqcd import (
    readPropagator as readXQCDPropagator,
    writePropagator as writeXQCDPropagator,
    readPropagatorFast as readXQCDPropagatorFast,
    writePropagatorFast as writeXQCDPropagatorFast,
)
from .nersc import (
    readGauge as readNERSCGauge,
    writeGauge as writeNERSCGauge,
)
from .openqcd import (
    readGauge as readOpenQCDGauge,
    writeGauge as writeOpenQCDGauge,
)
