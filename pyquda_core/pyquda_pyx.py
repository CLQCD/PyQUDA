import os
import subprocess
import sys
from typing import Dict, List, NamedTuple, Union


class QudaEnumMeta(NamedTuple):
    name: str
    value: Union[int, str]

    def __repr__(self) -> str:
        return f"{self.name} = {self.value}"


class QudaParamsMeta(NamedTuple):
    name: str
    type: str
    ptr: str
    array: List[int]

    def __repr__(self) -> str:
        self_array = "" if len(self.array) == 0 else f"[{']['.join(self.array)}]"
        return f"{self.type} {self.ptr}{self.name}{self_array}"


normal = """
    @property
    def %name%(self):
        return self.param.%name%

    @%name%.setter
    def %name%(self, value):
        self.param.%name% = value
"""

cstring = """
    @property
    def %name%(self):
        return self.param.%name%

    @%name%.setter
    def %name%(self, const char value[]):
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

    cdef set_%name%(self, %type% value):
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
            if self.param.%name%[i] != NULL:
                param = %type%()
                param.from_ptr(self.param.%name%[i])
                params.append(param)
            else:
                params.append(None)
        return params

    @%name%.setter
    def %name%(self, value):
        for i in range(self.param.n_level):
            if value[i] is not None:
                self.set_%name%(value[i], i)
            else:
                self.param.%name%[i] = NULL

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
    def %name%(self, Pointer value):
        assert value.dtype == "void"
        self.param.%name% = value.ptr
"""

ptr = """
    @property
    def %name%(self):
        ptr = Pointer("%type%")
        ptr.set_ptr(<void *>self.param.%name%)
        return ptr

    @%name%.setter
    def %name%(self, Pointer value):
        assert value.dtype == "%type%"
        self.param.%name% = <%type% *>value.ptr
"""

ptrptr = """
    @property
    def %name%(self):
        ptrs = Pointers("%type%", 0)
        ptrs.set_ptrs(<void **>self.param.%name%)
        return ptrs

    @%name%.setter
    def %name%(self, Pointers value):
        assert value.dtype == "%type%"
        self.param.%name% = <%type% **>value.ptrs
"""


def build_pyquda_pyx(pyquda_root, quda_path):
    fake_libc_include = os.path.join(pyquda_root, "pycparser", "utils", "fake_libc_include")
    quda_include = os.path.join(quda_path, "include")
    assert os.path.exists(fake_libc_include), f"{fake_libc_include} not found"
    print(f"Building pyquda wrapper from {os.path.join(quda_include, 'quda.h')}")
    sys.path.insert(1, os.path.join(pyquda_root, "pycparser"))
    try:
        from pycparser import parse_file, c_ast
    except ImportError or ModuleNotFoundError:
        from pycparser.pycparser import parse_file, c_ast  # This is for the language server
    sys.path.remove(os.path.join(pyquda_root, "pycparser"))

    def evaluate(node):
        if node is None:
            return
        elif type(node) is c_ast.UnaryOp:
            if node.op == "+":
                return evaluate(node.expr)
            elif node.op == "-":
                return -evaluate(node.expr)
        elif type(node) is c_ast.BinaryOp:
            if node.op == "+":
                return evaluate(node.left) + evaluate(node.right)
            elif node.op == "-":
                return evaluate(node.left) - evaluate(node.right)
        elif type(node) is c_ast.Constant:
            return int(node.value, 0)
        elif type(node) is c_ast.ID:
            return node.name
        else:
            raise ValueError(f"Unknown node {node}")

    subprocess.run(
        [
            "cpp",
            Rf"-I{quda_include}",
            "-o",
            os.path.join(pyquda_root, 'pyquda', 'quda_define.py'),
            os.path.join(pyquda_root, 'pyquda', 'quda_define.in.py'),
        ],
        check=True,
    )

    quda_enum_meta: Dict[str, List[QudaParamsMeta]] = {}
    quda_params_meta: Dict[str, List[QudaParamsMeta]] = {}
    ast = parse_file(
        os.path.join(quda_include, "quda.h"),
        use_cpp=True,
        cpp_path=os.environ["CC"] if "CC" in os.environ else "cc",
        cpp_args=["-E", Rf"-I{fake_libc_include}", Rf"-I{quda_include}"],
    )
    for node in ast:
        if not hasattr(node, "name") or node.name is None:
            continue
        if node.name.startswith("Quda") and type(node.type.type) is c_ast.Enum:
            quda_enum_meta[node.name] = []
            current_value = -1
            for item in node.type.type.values:
                item_value = evaluate(item.value)
                if item_value is not None:
                    current_value = item_value
                else:
                    current_value += 1
                if "INVALID" in item.name:
                    quda_enum_meta[node.name].append(QudaEnumMeta(item.name, "QUDA_INVALID_ENUM"))
                else:
                    quda_enum_meta[node.name].append(QudaEnumMeta(item.name, current_value))
        if node.name.startswith("Quda") and node.name.endswith("Param"):
            quda_params_meta[node.name] = []
            # print(node.name)
            for decl in node.type.type.decls:
                n, t = decl.name, decl.type
                if type(t) is c_ast.Union:
                    n, t = t.decls[0].name, t.decls[0].type
                tt = t.type
                if type(t) is c_ast.TypeDecl:
                    quda_params_meta[node.name].append(QudaParamsMeta(n, " ".join(tt.names), "", []))
                    # print(" ".join(t.type.names), n)
                elif type(t) is c_ast.ArrayDecl:
                    ttt = tt.type
                    if type(tt) is c_ast.TypeDecl:
                        quda_params_meta[node.name].append(QudaParamsMeta(n, " ".join(ttt.names), "", [t.dim.value]))
                        # print(" ".join(t.type.type.names), n, f"[{t.dim.value}]")
                    elif type(tt) is c_ast.PtrDecl:
                        quda_params_meta[node.name].append(
                            QudaParamsMeta(n, " ".join(ttt.type.names), "*", [t.dim.value])
                        )
                        # print(" ".join(t.type.type.type.names), "*", n, f"[{t.dim.value}]")
                    elif type(tt) is c_ast.ArrayDecl:
                        quda_params_meta[node.name].append(
                            QudaParamsMeta(n, " ".join(ttt.type.names), "", [t.dim.value, tt.dim.value])
                        )
                        # print(" ".join(t.type.type.type.names), n, f"[{t.dim.value}][{t.type.dim.value}]")
                    else:
                        raise ValueError(f"Unexpected node {node}")
                elif type(t) is c_ast.PtrDecl:
                    ttt = tt.type
                    if type(tt) is c_ast.TypeDecl:
                        quda_params_meta[node.name].append(QudaParamsMeta(n, " ".join(ttt.names), "*", ""))
                        # print(" ".join(t.type.type.names), "*", n)
                    elif type(tt) is c_ast.PtrDecl:
                        quda_params_meta[node.name].append(QudaParamsMeta(n, " ".join(ttt.type.names), "**", ""))
                        # print(" ".join(t.type.type.type.names), "**", n)
                    else:
                        raise ValueError(f"Unexpected node {node}")
                else:
                    raise ValueError(f"Unexpected node {node}")
        if isinstance(node.type, c_ast.FuncDecl):
            continue
            if node.name in ["set_dim", "pack_ghost"]:
                continue
            p = []
            if node.type.args is not None:
                for arg in node.type.args.params:
                    n, t = arg.name, arg.type
                    tt = t.type
                    if type(t) is c_ast.TypeDecl:
                        p.append((" ".join(t.quals + tt.names + [""]), n))
                    elif type(t) is c_ast.PtrDecl:
                        ttt = tt.type
                        if type(tt) is c_ast.TypeDecl:
                            p.append(
                                (" ".join(tt.quals + ttt.names + ["*"]), " ".join(t.quals + [n]))
                            )  # const type * const
                        elif type(tt) is c_ast.PtrDecl:
                            tttt = ttt.type
                            if type(ttt) is c_ast.TypeDecl:
                                p.append((" ".join(ttt.quals + tttt.names + ["**"]), n))
                            elif type(ttt) is c_ast.PtrDecl:
                                ttttt = tttt.type
                                if type(tttt) is c_ast.TypeDecl:
                                    p.append((" ".join(tttt.quals + ttttt.names + ["***"]), n))
                                else:
                                    raise ValueError(f"Unexpected node {node}")
                            else:
                                raise ValueError(f"Unexpected node {node}")
                        else:
                            raise ValueError(f"Unexpected node {node}")
                    elif type(t) is c_ast.ArrayDecl:
                        ttt = tt.type
                        if t.dim is not None:
                            if type(tt) is c_ast.TypeDecl:
                                p.append((" ".join(tt.quals + ttt.names + [""]), f"{n}[{t.dim.value}]"))
                            elif type(tt) is c_ast.PtrDecl:
                                tttt = ttt.type
                                if type(ttt) is c_ast.TypeDecl:
                                    p.append((" ".join(ttt.quals + tttt.names + ["*"]), f"{n}[{t.dim.value}]"))
                                else:
                                    raise ValueError(f"Unexpected node {node}")
                            else:
                                raise ValueError(f"Unexpected node {node}")
                        else:
                            if type(tt) is c_ast.TypeDecl:
                                p.append((" ".join(tt.quals + ttt.names + [""]), f"{n}[]"))
                            elif type(tt) is c_ast.PtrDecl:
                                tttt = ttt.type
                                if type(ttt) is c_ast.TypeDecl:
                                    p.append((" ".join(ttt.quals + tttt.names + ["*"]), f"{n}[]"))
                                else:
                                    raise ValueError(f"Unexpected node {node}")
                    else:
                        raise ValueError(f"Unexpected node {node}")
            ntt = node.type.type
            p = [""] if p == [("void", None)] else [f"{_t}{_a}" for _t, _a in p]
            if type(ntt) is c_ast.TypeDecl:
                print(" ".join(ntt.quals + ntt.type.names + [""]) + f"{node.name}(" + ", ".join(p) + ")")
            elif type(ntt) is c_ast.PtrDecl:
                nttt = ntt.type
                print(" ".join(nttt.quals + nttt.type.names + ["*"]) + f"{node.name}(" + ", ".join(p) + ")")
            else:
                raise ValueError(f"Unexpected node {node}")

    with open(os.path.join(quda_include, "quda_constants.h"), "r") as f:
        quda_constants_h = f.read()
    with open(os.path.join(quda_include, "enum_quda.h"), "r") as f:
        enum_quda_h = f.read()
    with open(os.path.join(pyquda_root, "pyquda", "enum_quda.in.py"), "r") as f:
        enum_quda_py = f.read()
    with open(os.path.join(pyquda_root, "pyquda", "src", "quda.in.pxd"), "r") as f:
        quda_pxd = f.read()
    with open(os.path.join(pyquda_root, "pyquda", "src", "pyquda.in.pyx"), "r") as f:
        pyquda_pyx = f.read()
    # with open(os.path.join(pyquda_root, "pyquda", "pyquda.in.pyi"), "r") as f:
    #     pyquda_pyi = f.read()

    quda_constants_py = ""
    quda_constants_pxd = 'cdef extern from "quda_constants.h":\n    cdef enum:'
    start = quda_constants_h.find("#define")
    while start >= 0:
        stop = quda_constants_h.find("\n", start)
        name_value = quda_constants_h[start + len("#define") : stop].strip().split(" ")
        name, value = name_value[0], " ".join(name_value[1:]).strip()
        idx_comment = quda_constants_h.rfind("@brief", 0, start)
        idx_endline = quda_constants_h.find("*/", idx_comment, start)
        if idx_comment < idx_endline:
            comment = "\n".join(
                [
                    line.strip("* ")
                    for line in quda_constants_h[idx_comment + len("@brief") : idx_endline].strip().split("\n")
                ]
            )
            comment = f'"""\n{comment}\n"""\n'
        else:
            comment = ""
        quda_constants_py += f"{name} = {value}\n{comment}\n"
        quda_constants_pxd += f"\n        {name}\n"
        start = quda_constants_h.find("#define", stop)
    enum_quda_py = enum_quda_py.replace("# quda_constants.py\n", quda_constants_py)

    enum_quda_pxd = 'cdef extern from "enum_quda.h":'
    for key, val in quda_enum_meta.items():
        enum_quda_pxd += f"\n    ctypedef enum {key}:\n        pass\n"

    idx_key = 0
    for key, val in quda_enum_meta.items():
        enum_quda_py_block = ""
        for item in val:
            comment = ""
            idx = enum_quda_h.find(item.name)
            idx_comment = enum_quda_h.find("//", idx)
            idx_endline = enum_quda_h.find("\n", idx)
            if idx_comment != -1 and idx_comment < idx_endline:
                comment = f'\n    """{enum_quda_h[idx_comment + 2 : idx_endline].strip()}"""'
            enum_quda_py_block += f"    {item}{comment}\n"
        if enum_quda_py.find(key) != -1:
            idx_key = enum_quda_py.find(key)
            enum_quda_py = enum_quda_py[:idx_key] + enum_quda_py[idx_key:].replace("    pass\n", enum_quda_py_block, 1)
        else:
            if enum_quda_py.find("class", idx_key) != -1:
                idx_key = enum_quda_py.find("class", idx_key)
                enum_quda_py = (
                    enum_quda_py[:idx_key] + f"class {key}(IntEnum):\n{enum_quda_py_block}\n\n" + enum_quda_py[idx_key:]
                )
                idx_key += len("class")
            else:
                enum_quda_py += f"\n\nclass {key}(IntEnum):\n{enum_quda_py_block}"

    for key, val in quda_params_meta.items():
        quda_pxd_block = ""
        pyquda_pyx_block = ""
        pyquda_pyi_block = ""
        for item in val:
            quda_pxd_block += f"        {item}\n"
            if len(item.array) > 0 and item.type.startswith("Quda") and item.type.endswith("Param"):
                pyquda_pyx_block += multigrid_param.replace("%name%", item.name).replace("%type%", item.type)
                pyquda_pyi_block += f"    {item.name}: List[{item.type}, {item.array[0]}]\n"
            elif item.type.startswith("Quda") and item.type.endswith("Param"):
                pyquda_pyx_block += param.replace("%name%", item.name).replace("%type%", item.type)
                pyquda_pyi_block += f"    {item.name}: {item.type}\n"
            elif len(item.array) == 1:
                if item.type == "char":
                    pyquda_pyx_block += cstring.replace("%name%", item.name)
                    pyquda_pyi_block += f"    {item.name}: bytes[{item.array[0]}]\n"
                else:
                    pyquda_pyx_block += normal.replace("%name%", item.name)
                    pyquda_pyi_block += f"    {item.name}: List[{item.type}, {item.array[0]}]\n"
            elif len(item.array) == 2:
                pyquda_pyx_block += multigrid.replace("%name%", item.name)
                if item.type == "char":
                    pyquda_pyi_block += f"    {item.name}: List[bytes[{item.array[1]}], {item.array[0]}]\n"
                else:
                    pyquda_pyi_block += f"    {item.name}: List[List[{item.type}, {item.array[1]}], {item.array[0]}]\n"
            elif item.ptr == "*" and item.type == "void":
                pyquda_pyx_block += void_ptr.replace("%name%", item.name)
                pyquda_pyi_block += f"    {item.name}: Pointer\n"
            elif item.ptr == "*":
                pyquda_pyx_block += ptr.replace("%name%", item.name).replace("%type%", item.type)
                pyquda_pyi_block += f"    {item.name}: Pointer\n"
            elif item.ptr == "**":
                pyquda_pyx_block += ptrptr.replace("%name%", item.name).replace("%type%", item.type)
                pyquda_pyi_block += f"    {item.name}: Pointers\n"
            else:
                pyquda_pyx_block += normal.replace("%name%", item.name)
                pyquda_pyi_block += f"    {item.name}: {item.type}\n"
        idx = quda_pxd.find(key)
        quda_pxd = quda_pxd[:idx] + quda_pxd[idx:].replace(
            "        pass\n", quda_pxd_block.replace("double _Complex", "double_complex"), 1
        )
        pyquda_pyx = pyquda_pyx.replace(
            f"\n##%%!! {key}\n",
            pyquda_pyx_block.replace("double _Complex", "double_complex"),
        )
        # pyquda_pyi = pyquda_pyi.replace(
        #     f"##%%!! {key}\n",
        #     pyi.replace("double _Complex", "double_complex").replace("unsigned int", "int"),
        # )

    with open(os.path.join(pyquda_root, "pyquda", "src", "quda_constants.pxd"), "w") as f:
        f.write(quda_constants_pxd)
    with open(os.path.join(pyquda_root, "pyquda", "src", "enum_quda.pxd"), "w") as f:
        f.write(enum_quda_pxd)
    with open(os.path.join(pyquda_root, "pyquda", "src", "quda.pxd"), "w") as f:
        f.write(quda_pxd)
    with open(os.path.join(pyquda_root, "pyquda", "src", "pyquda.pyx"), "w") as f:
        f.write(pyquda_pyx)
    with open(os.path.join(pyquda_root, "pyquda", "enum_quda.py"), "w") as f:
        f.write(enum_quda_py)
    # with open(os.path.join(pyquda_root, "pyquda", "pyquda.pyi"), "w") as f:
    #     f.write(pyquda_pyi)

    if os.path.exists(os.path.join(pyquda_root, "yacctab.py")):
        os.remove(os.path.join(pyquda_root, "yacctab.py"))
    if os.path.exists(os.path.join(pyquda_root, "lextab.py")):
        os.remove(os.path.join(pyquda_root, "lextab.py"))
