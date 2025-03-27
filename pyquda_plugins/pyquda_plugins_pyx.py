import os
from typing import Dict, List, Tuple

_C_TO_PYTHON: Dict[str, str] = {
    "void": "None",
    "int": "int",
    "size_t": "int",
    "double": "float",
    "double complex": "complex",
    "bool": "bool",
}

_C_TO_NUMPY: Dict[str, str] = {
    "int": "int32",
    "double": "float64",
    "double complex": "complex128",
}


class Plugin:
    def __init__(self, lib: str, header: str):
        self.lib = lib
        self.lib_pxd = f'cdef extern from "{header}":\n'
        self.pylib_pyx = (
            "from libcpp cimport bool\n"
            "from numpy cimport ndarray\n"
            "from pyquda.pointer cimport Pointer, Pointers, Pointerss, _NDArray\n"
            f"cimport {lib}\n"
            "\n"
            "\n"
        )
        self.pylib_pyi = (
            "from numpy import int32, float64, complex128\n"
            "from numpy.typing import NDArray\n"
            "from pyquda.pointer import Pointer, Pointers, Pointerss\n"
            "\n"
            "\n"
        )

    @classmethod
    def parseArguments(cls, args: str):
        args = [arg.strip() for arg in args.replace("*", " * ").split(",")]
        for i, arg in enumerate(args):
            t, n = " ".join(arg.split(" ")[:-1]), arg.split(" ")[-1]
            t_ = t.replace("  ", " ").replace("* ", "*")
            while t != t_:
                t, t_ = t_, t.replace("  ", " ").replace("* ", "*")
            args[i] = (t, n)
        return args

    def function(self, c_func: str, func: str, args: List[Tuple[str, str]], ret_type: str = "void"):
        if isinstance(args, str):
            args = self.parseArguments(args)
        args_type = [arg[0] for arg in args]
        args_name = [arg[1] for arg in args]
        arg = [f"{t}{n}" if t.endswith("*") else f"{t} {n}" for t, n in zip(args_type, args_name)]
        self.lib_pxd += f"    {ret_type} {c_func}({', '.join(arg)})\n"
        args_type_in = []
        args_name_in = []
        args_type_stub = []
        for t, n in args:
            if t == "const char *":
                args_type_in.append(t)
                args_name_in.append(n)
                args_type_stub.append("bytes")
            elif t.endswith("***") and t != "void ***":
                args_type_in.append(f"ndarray[{t[:-4]}, ndim=3]")
                args_name_in.append(f"<{t}>_{n}.ptrss")
                args_type_stub.append(f"NDArray[{_C_TO_NUMPY[t[:-4]]}]")
            elif t == "void ***":
                args_type_in.append("Pointerss")
                args_name_in.append(f"{n}.ptrss")
                args_type_stub.append("Pointerss")
            elif t.endswith("**") and t != "void **":
                args_type_in.append(f"ndarray[{t[:-3]}, ndim=2]")
                args_name_in.append(f"<{t}>_{n}.ptrs")
                args_type_stub.append(f"NDArray[{_C_TO_NUMPY[t[:-3]]}]")
            elif t == "void **":
                args_type_in.append("Pointers")
                args_name_in.append(f"{n}.ptrs")
                args_type_stub.append("Pointers")
            elif t.endswith("*") and t != "void *":
                args_type_in.append(f"ndarray[{t[:-2]}, ndim=1]")
                args_name_in.append(f"<{t}>_{n}.ptr")
                args_type_stub.append(f"NDArray[{_C_TO_NUMPY[t[:-2]]}]")
            elif t == "void *":
                args_type_in.append("Pointer")
                args_name_in.append(f"{n}.ptr")
                args_type_stub.append("Pointer")
            else:
                args_type_in.append(t)
                args_name_in.append(n)
                args_type_stub.append(_C_TO_PYTHON[t])
        arg = [f"{t} {n}" for t, n in zip(args_type_in, args_name)]
        self.pylib_pyx += f"def {func}({', '.join(arg)}):\n"
        for t, n in zip(args_type_in, args_name):
            if t.startswith("ndarray"):
                self.pylib_pyx += f"    _{n} = _NDArray({n})\n"
        if ret_type != "void":
            self.pylib_pyx += f"    return {self.lib}.{c_func}({', '.join(args_name_in)})\n"
        else:
            self.pylib_pyx += f"    {self.lib}.{c_func}({', '.join(args_name_in)})\n"
        self.pylib_pyx += "\n"
        arg = [f"{n}: {t}" for t, n in zip(args_type_stub, args_name)]
        self.pylib_pyi += f"def {func}({', '.join(arg)}) -> {_C_TO_PYTHON[ret_type]}:\n" "    ...\n" "\n"

    def write(self, pyquda_plugins_root: str):
        with open(os.path.join(pyquda_plugins_root, f"{self.lib}.pxd"), "w") as f:
            f.write(self.lib_pxd)
        with open(os.path.join(pyquda_plugins_root, f"py{self.lib}.pyx"), "w") as f:
            f.write(self.pylib_pyx)
        with open(os.path.join(pyquda_plugins_root, f"py{self.lib}.pyi"), "w") as f:
            f.write(self.pylib_pyi)
