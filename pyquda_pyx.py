import os
import sys
from typing import Dict, List, NamedTuple


class Meta(NamedTuple):
    name: str
    type: str
    ptr: str
    array: List[int]

    def __repr__(self) -> str:
        return f"{self.type} {self.ptr} {self.name} {self.array}"


normal = """
    @property
    def %name%(self):
        return self.param.%name%

    @%name%.setter
    def %name%(self, value):
        self.param.%name% = value
"""

param = """
    @property
    def %name%(self):
        param = %type%()
        param.from_ptr(self.param.%name%)
        return param

    @%name%.setter
    def %name%(self, value):
        self.set_%name%(value)

    cdef set_%name%(self, QudaInvertParam value):
        self.param.%name% = &value.param
"""

multigrid = """
    @property
    def %name%(self):
        value = []
        for i in range(self.n_level):
            value.append(self.param.%name%[i])
        return value

    @%name%.setter
    def %name%(self, value):
        for i in range(self.n_level):
            self.param.%name%[i] = value[i]
"""

multigrid_param = """
    @property
    def %name%(self):
        params = []
        for i in range(self.param.n_level):
            param = %type%()
            param.from_ptr(self.param.%name%[i])
            params.append(param)
        return params

    @%name%.setter
    def %name%(self, value):
        for i in range(self.param.n_level):
            self.set_%name%(value[i], i)

    cdef set_%name%(self, %type% value, int i):
        self.param.%name%[i] = &value.param
"""

void_ptr = """
    @property
    def %name%(self):
        ptr = Pointer("void")
        ptr.set_ptr(self.param.%name%)
        return ptr

    @%name%.setter
    def %name%(self, value):
        self.set_%name%(value)

    cdef set_%name%(self, Pointer value):
        assert value.dtype == "void"
        self.param.%name% = value.ptr
"""

ptr = """
    @property
    def %name%(self):
        ptr = Pointer("%type%")
        ptr.set_ptr(self.param.%name%)
        return ptr

    @%name%.setter
    def %name%(self, value):
        self.set_%name%(value)

    cdef set_%name%(self, Pointer value):
        assert value.dtype == "%type%"
        self.param.%name% = <%type% *>value.ptr
"""

ptrptr = """
    @property
    def %name%(self):
        ptr = Pointers("%type%", self.param.num_paths)
        ptr.set_ptrs(<void **>self.param.%name%)
        return ptr

    @%name%.setter
    def %name%(self, value):
        self.set_%name%(value)

    cdef set_%name%(self, Pointers value):
        assert value.dtype == "%type%"
        self.param.%name% = <%type% **>value.ptrs
"""


def build_pyquda_pyx(pyquda_root, quda_path):
    print(f"Building pyquda wrapper from {os.path.join(quda_path, 'include', 'quda.h')}")
    pycparser_root = os.path.join(pyquda_root, "pycparser")
    sys.path.insert(1, pycparser_root)
    from pycparser import parse_file, c_ast

    meta: Dict[str, List[Meta]] = {}
    ast = parse_file(
        os.path.join(quda_path, "include", "quda.h"),
        use_cpp=True,
        cpp_path="cc",
        cpp_args=[
            "-E",
            Rf"-I{os.path.join(pycparser_root, 'utils', 'fake_libc_include')}",
            Rf"-I{os.path.join(quda_path, 'include')}",
        ],
    )
    for node in ast:
        if node.name.startswith("Quda") and node.name.endswith("Param"):
            meta[node.name] = []
            # print(node.name)
            for decl in node.type.type.decls:
                n, t = decl.name, decl.type
                tt = t.type
                if type(t) is c_ast.TypeDecl:
                    meta[node.name].append(Meta(n, " ".join(tt.names), "", []))
                    # print(" ".join(t.type.names), n)
                elif type(t) is c_ast.ArrayDecl:
                    ttt = tt.type
                    if type(tt) is c_ast.TypeDecl:
                        meta[node.name].append(Meta(n, " ".join(ttt.names), "", [t.dim.value]))
                        # print(" ".join(t.type.type.names), n, f"[{t.dim.value}]")
                    elif type(tt) is c_ast.PtrDecl:
                        meta[node.name].append(Meta(n, " ".join(ttt.type.names), "*", [t.dim.value]))
                        # print(" ".join(t.type.type.type.names), "*", n, f"[{t.dim.value}]")
                    elif type(tt) is c_ast.ArrayDecl:
                        meta[node.name].append(Meta(n, " ".join(ttt.type.names), "", [t.dim.value, tt.dim.value]))
                        # print(" ".join(t.type.type.type.names), n, f"[{t.dim.value}][{t.type.dim.value}]")
                    else:
                        raise ValueError(f"Unexpected node {node}")
                elif type(t) is c_ast.PtrDecl:
                    ttt = tt.type
                    if type(tt) is c_ast.TypeDecl:
                        meta[node.name].append(Meta(n, " ".join(ttt.names), "*", ""))
                        # print(" ".join(t.type.type.names), "*", n)
                    elif type(tt) is c_ast.PtrDecl:
                        meta[node.name].append(Meta(n, " ".join(ttt.type.names), "**", ""))
                        # print(" ".join(t.type.type.type.names), "**", n)
                    else:
                        raise ValueError(f"Unexpected node {node}")
                else:
                    raise ValueError(f"Unexpected node {node}")

    with open(os.path.join(pyquda_root, "pyquda", "src", "quda.in.pxd"), "r") as f:
        quda_pxd = f.read()
    with open(os.path.join(pyquda_root, "pyquda", "src", "pyquda.in.pyx"), "r") as f:
        pyquda_pyx = f.read()
    with open(os.path.join(pyquda_root, "pyquda", "pyquda.in.pyi"), "r") as f:
        pyquda_pyi = f.read()

    for key, val in meta.items():
        pxd = ""
        pyx = ""
        pyi = ""
        for item in val:
            item_array = "" if len(item.array) == 0 else f"[{']['.join(item.array)}]"
            pxd += f"        {item.type} {item.ptr}{item.name}{item_array}\n"
            if len(item.array) > 0 and item.type.startswith("Quda") and item.type.endswith("Param"):
                pyx += multigrid_param.replace("%name%", item.name).replace("%type%", item.type)
                pyi += f"    {item.name}: List[{item.type}, {item.array[0]}]\n"
            elif item.type.startswith("Quda") and item.type.endswith("Param"):
                pyx += param.replace("%name%", item.name).replace("%type%", item.type)
                pyi += f"    {item.name}: {item.type}\n"
            elif len(item.array) == 1:
                pyx += normal.replace("%name%", item.name)
                if item.type == "char":
                    pyi += f"    {item.name}: bytes[{item.array[0]}]\n"
                else:
                    pyi += f"    {item.name}: List[{item.type}, {item.array[0]}]\n"
            elif len(item.array) == 2:
                pyx += multigrid.replace("%name%", item.name)
                if item.type == "char":
                    pyi += f"    {item.name}: List[bytes[{item.array[1]}], {item.array[0]}]\n"
                else:
                    pyi += f"    {item.name}: List[List[{item.type}, {item.array[1]}], {item.array[0]}]\n"
            elif item.ptr == "*" and item.type == "void":
                pyx += void_ptr.replace("%name%", item.name)
                pyi += f"    {item.name}: Pointer\n"
            elif item.ptr == "*":
                pyx += ptr.replace("%name%", item.name).replace("%type%", item.type)
                pyi += f"    {item.name}: Pointer\n"
            elif item.ptr == "**":
                pyx += ptrptr.replace("%name%", item.name).replace("%type%", item.type)
                pyi += f"    {item.name}: Pointers\n"
            else:
                pyx += normal.replace("%name%", item.name)
                pyi += f"    {item.name}: {item.type}\n"
        quda_pxd = quda_pxd.replace(
            f"        pass\n\n##%%!! {key}\n",
            pxd.replace("double _Complex", "double_complex"),
        )
        pyquda_pyx = pyquda_pyx.replace(
            f"\n##%%!! {key}\n",
            pyx.replace("double _Complex", "double_complex"),
        )
        pyquda_pyi = pyquda_pyi.replace(
            f"##%%!! {key}\n",
            pyi.replace("double _Complex", "double_complex").replace("unsigned int", "int"),
        )

    with open(os.path.join(pyquda_root, "pyquda", "src", "quda.pxd"), "w") as f:
        f.write(quda_pxd)
    with open(os.path.join(pyquda_root, "pyquda", "src", "pyquda.pyx"), "w") as f:
        f.write(pyquda_pyx)
    # with open(os.path.join(pyquda_root, "pyquda", "pyquda.pyi"), "w") as f:
    #     f.write(pyquda_pyi)

    os.remove(os.path.join(pyquda_root, "yacctab.py"))
    os.remove(os.path.join(pyquda_root, "lextab.py"))
