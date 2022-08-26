import numpy as np


def colorvec_cb2(data, t_srce: int = None):
    if t_srce is not None:
        t = t_srce
        Ne, Lz, Ly, Lx, Nc = data.shape
        data_cb2 = np.zeros((Ne, 2, Lz, Ly, Lx // 2, Nc), "<c16")
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) // 2
                if eo == 0:
                    data_cb2[:, 0, z, y] = data[:, z, y, 0::2]
                    data_cb2[:, 1, z, y] = data[:, z, y, 1::2]
                else:
                    data_cb2[:, 0, z, y] = data[:, z, y, 1::2]
                    data_cb2[:, 1, z, y] = data[:, z, y, 0::2]
    else:
        raise NotImplementedError("Havn't implemented t_srce=None yet")
    return data_cb2
