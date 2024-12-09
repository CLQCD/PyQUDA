import io
from os import path
import struct
from typing import List, NamedTuple


class LimeRecord(NamedTuple):
    name: str
    offset: int
    length: int


class Lime:
    def __init__(self, filename: str):
        self.filename = path.expanduser(path.expandvars(filename))
        self._records: List[LimeRecord] = []
        with open(self.filename, "rb") as f:
            buffer = f.read(8)
            while buffer != b"" and buffer != b"\x0A":
                assert buffer.startswith(b"\x45\x67\x89\xAB\x00\x01")
                length = struct.unpack(">Q", f.read(8))[0]
                name = f.read(128).strip(b"\x00").decode("utf-8")
                self._records.append(LimeRecord(name, f.tell(), length))
                f.seek((length + 7) // 8 * 8, io.SEEK_CUR)
                buffer = f.read(8)

    def keys(self):
        return [record.name for record in self._records]

    def records(self, key: str):
        return [record for record in self._records if record.name == key]

    def record(self, key: str, index: int = 0):
        return [record for record in self._records if record.name == key][index]

    def read(self, key: str, index: int = 0):
        record = self.record(key, index)
        with open(self.filename, "rb") as f:
            f.seek(record.offset)
            buffer = f.read(record.length)
        return buffer
