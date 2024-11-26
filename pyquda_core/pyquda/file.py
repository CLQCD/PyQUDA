from time import perf_counter
from typing import List, Sequence, Tuple, Union

import numpy
from mpi4py import MPI
import h5py


class _LatticeInfo:
    def __init__(self, latt_size: List[int], grid_size: List[int]) -> None:
        self._checkLattice(latt_size, grid_size)
        self._setLattice(latt_size, grid_size)

    def _checkLattice(self, latt_size: List[int], grid_size: List[int]):
        assert len(latt_size) == len(grid_size), "lattice size and grid size must have the same dimension"
        for GL, G in zip(latt_size, grid_size):
            if not (GL % G == 0):
                raise ValueError("lattice size must be divisible by gird size")

    def _setLattice(self, latt_size: List[int], grid_size: List[int]):
        if MPI.COMM_WORLD.Get_size() != int(numpy.prod(grid_size)):
            raise ValueError(f"The MPI size {MPI.COMM_WORLD.Get_size()} does not match the grid size {grid_size}")
        sublatt_size = [GL // G for GL, G in zip(latt_size, grid_size)]
        sublatt_slice = []
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        for G, L in zip(grid_size[::-1], sublatt_size[::-1]):
            g = mpi_rank % G
            mpi_rank //= G
            sublatt_slice.append(slice(g * L, (g + 1) * L))

        self.global_size = latt_size
        self.global_volume = int(numpy.prod(latt_size))
        self.size = sublatt_size
        self.volume = int(numpy.prod(sublatt_size))
        self.slice = tuple(sublatt_slice)


# CRC32LUT = numpy.empty((4, 256), dtype="<u4")
# # fmt: off
# CRC32LUT[0] = numpy.array(
#     [
#         0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
#         0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
#         0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
#         0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
#         0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
#         0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
#         0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
#         0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
#         0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
#         0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
#         0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
#         0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
#         0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
#         0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
#         0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
#         0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
#         0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
#         0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
#         0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
#         0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
#         0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
#         0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
#         0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
#         0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
#         0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
#         0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
#         0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
#         0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
#         0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
#         0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
#         0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
#         0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
#     ],
#     "<u4"
# )
# # fmt: on
# for i in range(256):
#     CRC32LUT[1, i] = (CRC32LUT[0, i] >> 8) ^ CRC32LUT[0, CRC32LUT[0, i] & 0xFF]
#     CRC32LUT[2, i] = (CRC32LUT[1, i] >> 8) ^ CRC32LUT[0, CRC32LUT[1, i] & 0xFF]
#     CRC32LUT[3, i] = (CRC32LUT[2, i] >> 8) ^ CRC32LUT[0, CRC32LUT[2, i] & 0xFF]
#     # CRC32LUT[4, i] = (CRC32LUT[3, i] >> 8) ^ CRC32LUT[0, CRC32LUT[3, i] & 0xFF]
#     # CRC32LUT[5, i] = (CRC32LUT[4, i] >> 8) ^ CRC32LUT[0, CRC32LUT[4, i] & 0xFF]
#     # CRC32LUT[6, i] = (CRC32LUT[5, i] >> 8) ^ CRC32LUT[0, CRC32LUT[5, i] & 0xFF]
#     # CRC32LUT[7, i] = (CRC32LUT[6, i] >> 8) ^ CRC32LUT[0, CRC32LUT[6, i] & 0xFF]


def checksum(latt_info, data: numpy.ndarray) -> Tuple[int, int]:
    import zlib

    work = numpy.empty((latt_info.volume), "<u4")
    for i in range(latt_info.volume):
        work[i] = zlib.crc32(data[i])
    # work = numpy.full_like(data[:, 0], 0xFFFFFFFF)
    # for i in range(data.shape[1]):
    #     work ^= data[:, i]
    #     work_view = work.view("|u1").reshape(-1, 4)
    #     work = (
    #         CRC32LUT[0].take(work_view[:, 0], mode="wrap")
    #         ^ CRC32LUT[1].take(work_view[:, 1], mode="wrap")
    #         ^ CRC32LUT[2].take(work_view[:, 2], mode="wrap")
    #         ^ CRC32LUT[3].take(work_view[:, 3], mode="wrap")
    #     )
    # work ^= 0xFFFFFFFF
    # work = numpy.bitwise_xor.reduce(data, 1)
    rank = (
        numpy.arange(latt_info.global_volume, dtype="<u8")
        .reshape(*latt_info.global_size[::-1])[latt_info.slice]
        .reshape(-1)
    )
    rank29 = (rank % 29).astype("<u4")
    rank31 = (rank % 31).astype("<u4")
    sum29 = MPI.COMM_WORLD.allreduce(numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)).item(), MPI.BXOR)
    sum31 = MPI.COMM_WORLD.allreduce(numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)).item(), MPI.BXOR)
    return sum29, sum31


def _spin_color_dtype(name: str, shape: Sequence[int], use_fp32: bool = True) -> Tuple[int, int]:
    float_nbytes = 4 if use_fp32 else 8
    Ns, Nc, dtype = 4, 3, f"<c{2 * float_nbytes}"
    if name.endswith("Int"):
        () = shape
        dtype = "<i4"
    elif name.endswith("Real"):
        () = shape
        dtype = f"<f{float_nbytes}"
    elif name.endswith("Complex"):
        () = shape
    elif name.endswith("SpinColorVector"):
        Ns, Nc = shape
    elif name.endswith("SpinColorMatrix"):
        Ns, Ns_, Nc, Nc_ = shape
        assert Ns == Ns_ and Nc == Nc_
    elif name.endswith("ColorVector"):
        (Nc,) = shape
    elif name.endswith("ColorMatrix"):
        Nc, Nc_ = shape
        assert Nc == Nc_
    else:
        raise ValueError(f"Invalid field type: {name}")
    return Ns, Nc, dtype


def _field_info(
    group: str,
    label: Union[int, str, Sequence[str], Sequence[str]],
    field: numpy.ndarray,
    grid_size: Sequence[int],
    use_fp32: bool,
):
    if isinstance(label, (int, str)):
        keys = str(label)
        sublatt_size = field.shape[0 : len(grid_size)][::-1]
        field_shape = field.shape[len(grid_size) :]
    elif isinstance(label, (list, tuple, range)):
        assert len(label) == field.shape[0]
        keys = [str(key) for key in label]
        sublatt_size = field.shape[1 : 1 + len(grid_size)][::-1]
        field_shape = field.shape[1 + len(grid_size) :]
    else:
        raise TypeError(f"Invalid label {label} for field type {group}")
    latt_size = [G * L for G, L in zip(grid_size, sublatt_size)]
    Ns, Nc, field_dtype = _spin_color_dtype(group, field_shape, use_fp32)
    return keys, latt_size, Ns, Nc, field_shape, field_dtype


class File(h5py.File):
    def __init__(self, name, mode="r", **kwds):
        """Create a new file object with the mpio driver.

        See the h5py user guide for a detailed explanation of the options.

        name
            Name of the file on disk, or file-like object.
        mode
            r        Readonly, file must exist (default)
            r+       Read/write, file must exist
            w        Create file, truncate if exists
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        """
        super().__init__(name, mode, driver="mpio", comm=MPI.COMM_WORLD, **kwds)

    @classmethod
    def _load(cls, latt_info: _LatticeInfo, dataset: h5py.Dataset, check: bool = True):
        data: numpy.ndarray = dataset[latt_info.slice]
        if check:
            sum29, sum31 = checksum(latt_info, data.reshape(latt_info.volume, -1).view("<u4"))
            assert dataset.attrs["sum29"] == f"0x{sum29:08x}", f"{dataset.attrs['sum29']} != 0x{sum29:08x}"
            assert dataset.attrs["sum31"] == f"0x{sum31:08x}", f"{dataset.attrs['sum31']} != 0x{sum31:08x}"
        return data

    def load(
        self,
        group: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        grid_size: Sequence[int],
        *,
        check: bool = True,
    ):
        s = perf_counter()
        gbytes = 0
        g = self[group]
        # assert g.attrs["Lattice"] == " ".join([str(L) for L in latt_size])
        # assert g.attrs["Spin"] == str(Ns)
        # assert g.attrs["Color"] == str(Nc)
        latt_size = [int(GL) for GL in g.attrs["Lattice"].split()]
        Ns = int(g.attrs["Spin"])
        Nc = int(g.attrs["Color"])
        if isinstance(label, (int, str)):
            keys = str(label)
        elif isinstance(label, (list, tuple, range)):
            assert len(label) <= len(g)
            keys = [str(key) for key in label]

        for key in g.keys():
            field_dtype = g[key].dtype.str.replace("<c8", "<c16").replace("<f4", "<f8")
            break
        latt_info = _LatticeInfo(latt_size, grid_size)
        if isinstance(keys, str):
            key = keys
            value = self._load(latt_info, g[key], check).astype(field_dtype)
            gbytes += g[key].nbytes / 1024**3
        else:
            value = []
            for key in keys:
                value.append(self._load(latt_info, g[key], check).astype(field_dtype))
                gbytes += g[key].nbytes / 1024**3
        secs = perf_counter() - s
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Loaded {group} from {self.filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return latt_size, Ns, Nc, value

    @classmethod
    def _save(cls, latt_info: _LatticeInfo, dataset: h5py.Dataset, data: numpy.ndarray, check: bool = True):
        dataset[latt_info.slice] = data
        if check:
            sum29, sum31 = checksum(latt_info, data.reshape(latt_info.volume, -1).view("<u4"))
            dataset.attrs["sum29"] = f"0x{sum29:08x}"
            dataset.attrs["sum31"] = f"0x{sum31:08x}"

    def save(
        self,
        group: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        field: numpy.ndarray,
        grid_size: Sequence[int],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        s = perf_counter()
        gbytes = 0
        keys, latt_size, Ns, Nc, field_shape, field_dtype = _field_info(group, label, field, grid_size, use_fp32)
        g = self.create_group(group)
        g.attrs["Annotation"] = annotation
        g.attrs["Lattice"] = " ".join([str(GL) for GL in latt_size])
        g.attrs["Spin"] = str(Ns)
        g.attrs["Color"] = str(Nc)

        latt_info = _LatticeInfo(latt_size, grid_size)
        if isinstance(keys, str):
            key = keys[0]
            g.create_dataset(key, (*latt_size[::-1], *field_shape), field_dtype)
            self._save(latt_info, g[key], field.astype(field_dtype), check)
            gbytes += g[key].nbytes / 1024**3
        else:
            for index, key in enumerate(keys):
                g.create_dataset(key, (*latt_size[::-1], *field_shape), field_dtype)
                self._save(latt_info, g[key], field[index].astype(field_dtype), check)
                gbytes += g[key].nbytes / 1024**3
        secs = perf_counter() - s
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Saved {group} to {self.filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def append(
        self,
        group: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        field: numpy.ndarray,
        grid: Sequence[int],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        s = perf_counter()
        gbytes = 0
        keys, latt_size, Ns, Nc, field_shape, field_dtype = _field_info(group, label, field, grid, use_fp32)
        g = self.create_group(group)
        g.attrs["Annotation"] = annotation
        g.attrs["Lattice"] = " ".join([str(GL) for GL in latt_size])
        g.attrs["Spin"] = str(Ns)
        g.attrs["Color"] = str(Nc)

        latt_info = _LatticeInfo(latt_size, grid)
        if isinstance(keys, str):
            key = keys
            g.create_dataset(key, (*latt_size[::-1], *field_shape), field_dtype)
            self._save(latt_info, g[key], field.astype(field_dtype), check)
            gbytes += g[key].nbytes / 1024**3
        else:
            for index, key in enumerate(keys):
                g.create_dataset(key, (*latt_size[::-1], *field_shape), field_dtype)
                self._save(latt_info, g[key], field[index].astype(field_dtype), check)
                gbytes += g[key].nbytes / 1024**3
        secs = perf_counter() - s
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Append {group} to {self.filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def update(
        self,
        group: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        field: numpy.ndarray,
        grid: Sequence[int],
        *,
        annotation: str = "",
        check: bool = True,
    ):
        s = perf_counter()
        gbytes = 0
        keys, latt_size, Ns, Nc, field_shape, field_dtype = _field_info(group, label, field, grid, False)
        g = self[group]
        if annotation != "":
            g.attrs["Annotation"] = annotation
        assert g.attrs["Lattice"] == " ".join([str(L) for L in latt_size])
        assert g.attrs["Spin"] == str(Ns)
        assert g.attrs["Color"] == str(Nc)

        for key in g.keys():
            field_dtype = g[key].dtype.str
            break
        latt_info = _LatticeInfo(latt_size, grid)
        if isinstance(keys, str):
            key = keys
            self._save(latt_info, g[key], field.astype(field_dtype), check)
            gbytes += g[key].nbytes / 1024**3
        else:
            for index, key in enumerate(keys):
                if key not in g:
                    g.create_dataset(key, (*latt_size[::-1], *field_shape), field_dtype)
                self._save(latt_info, g[key], field[index].astype(field_dtype), check)
                gbytes += g[key].nbytes / 1024**3
        secs = perf_counter() - s
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Updated {group} to {self.filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
