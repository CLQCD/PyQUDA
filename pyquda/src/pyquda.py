import os
from typing import Dict, List, NamedTuple
from pycparser import __file__, parse_file, c_ast


class Meta(NamedTuple):
    name: str
    type: str
    ptr: str
    array: str

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

meta: Dict[str, List[Meta]] = {}
ast = parse_file(
    "../quda/include/quda.h",
    use_cpp=True,
    cpp_path="cc",
    cpp_args=["-E", Rf"-I{os.path.dirname(__file__)}/utils/fake_libc_include", R"-I../quda/include"],
)
for node in ast:
    if node.name.startswith("Quda") and node.name.endswith("Param"):
        meta[node.name] = []
        # print(node.name)
        for decl in node.type.type.decls:
            n, t = decl.name, decl.type
            tt = t.type
            if type(t) is c_ast.TypeDecl:
                meta[node.name].append(Meta(n, " ".join(tt.names), "", ""))
                # print(" ".join(t.type.names), n)
            elif type(t) is c_ast.ArrayDecl:
                ttt = tt.type
                if type(tt) is c_ast.TypeDecl:
                    meta[node.name].append(Meta(n, " ".join(ttt.names), "", f"[{t.dim.value}]"))
                    # print(" ".join(t.type.type.names), n, f"[{t.dim.value}]")
                elif type(tt) is c_ast.PtrDecl:
                    meta[node.name].append(Meta(n, " ".join(ttt.type.names), "*", f"[{t.dim.value}]"))
                    # print(" ".join(t.type.type.type.names), "*", n, f"[{t.dim.value}]")
                elif type(tt) is c_ast.ArrayDecl:
                    meta[node.name].append(Meta(n, " ".join(ttt.type.names), "", f"[{t.dim.value}][{tt.dim.value}]"))
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

with open("pyquda.in.pyx", "r") as f:
    content = f.read()

for key, val in meta.items():
    pyx = ""
    for item in val:
        if "][" in item.array:
            pyx += multigrid.replace("%name%", item.name)
        elif item.type.startswith("Quda") and item.type.endswith("Param") and item.array != "":
            pyx += multigrid_param.replace("%name%", item.name).replace("%type%", item.type)
        elif item.type.startswith("Quda") and item.type.endswith("Param"):
            pyx += param.replace("%name%", item.name).replace("%type%", item.type)
        elif item.ptr == "*" and item.type == "void":
            pyx += void_ptr.replace("%name%", item.name)
        elif item.ptr == "*":
            pyx += ptr.replace("%name%", item.name).replace("%type%", item.type)
        elif item.ptr == "**":
            pyx += ptrptr.replace("%name%", item.name).replace("%type%", item.type)
        else:
            pyx += normal.replace("%name%", item.name)
    content = content.replace(
        f"\n##%%!! {key}\n",
        pyx.replace("double _Complex", "double_complex").replace("double_complex *", "double complex *"),
    )

with open("pyquda.pyx", "w") as f:
    f.write(content)
