import os
from typing import Dict, List, NamedTuple, Tuple

_C_TO_PYTHON: Dict[str, str] = {
    "void *": "Pointer",
    "void": "None",
    "short": "int",
    "unsigned short": "int",
    "int": "int",
    "unsigned int": "int",
    "long": "int",
    "unsigned long": "int",
    "long long": "int",
    "unsigned long long": "int",
    "float": "float",
    "double": "float",
    "long double": "float",
    "float _Complex": "complex",
    "double _Complex": "complex",
    "long double _Complex": "complex",
    "bool": "bool",
}
_C_TO_NUMPY: Dict[str, str] = {
    "int": "int32",
    "double": "float64",
    "double _Complex": "complex128",
}


class FunctionMeta(NamedTuple):
    ret: Tuple[str, str]
    name: str
    args: List[Tuple[str, str, str, str, str]]

    def __repr__(self):
        ret_t, ret_p = self.ret
        ret_str = " ".join([ret_t] + [ret_p])
        args_str = []
        for q, t, p, n, a in self.args:
            q = [] if q == "" else [q]
            args_str.append(f'{" ".join(q + [t] + [p])}{n}{a}')
        return f"{ret_str}{self.name}({', '.join(args_str)})".replace("_Complex", "complex")

    def pyi(self):
        ret_t, ret_p = self.ret
        ret_p = [] if ret_p == "" else [ret_p]
        ret_str = " ".join([ret_t] + ret_p)
        args_str = []
        for q, t, p, n, a in self.args:
            if a != "":
                args_str.append(f"{n}: List[{_C_TO_PYTHON[t]}, {a[1:-1]}]")
            elif p == "*":
                if q == "const" and t == "char":
                    args_str.append(f"{n}: bytes")
                elif t == "void":
                    args_str.append(f"{n}: NDArray")
                else:
                    args_str.append(f"{n}: NDArray[{_C_TO_NUMPY[t]}]")
            elif p == "**":
                if t == "void":
                    args_str.append(f"{n}: NDArray")
                else:
                    args_str.append(f"{n}: NDArray[{_C_TO_NUMPY[t]}]")
            elif p == "***":
                if t == "void":
                    args_str.append(f"{n}: NDArray")
                else:
                    args_str.append(f"{n}: NDArray[{_C_TO_NUMPY[t]}]")
            else:
                args_str.append(f"{n}: {_C_TO_PYTHON[t]}")
        return f"def {self.name}({', '.join(args_str)}) -> {_C_TO_PYTHON[ret_str]}: ...\n"

    def pyx(self, lib: str):
        result = ""
        # ret_t, ret_p = self.ret
        # if ret_t != "void" or ret_p == "":
        #     ret_str = "return "
        args_str = []
        args_in_str = []
        for q, t, p, n, a in self.args:
            q = [] if q == "" else [q]
            if a != "":
                args_str.append(f'{" ".join(q + [t] + [p])}{n}{a}')
                args_in_str.append(n)
            elif p == "*":
                if q == ["const"] and t == "char":
                    args_in_str.append(n)
                    args_str.append(f'{" ".join(q + [t] + [p])}{n}')
                elif t == "void":
                    args_in_str.append(f"<{' '.join(q + [t])} {p}>_{n}.ptr")
                    args_str.append(f"{n}")
                    result += f"    _{n} = _NDArray({n}, 1)\n"
                else:
                    args_in_str.append(f"<{' '.join(q + [t])} {p}>_{n}.ptr")
                    t = f"ndarray[{t}, ndim=1]"
                    args_str.append(f'{" ".join([t])} {n}')
                    result += f"    _{n} = _NDArray({n})\n"
            elif p == "**":
                if t == "void":
                    args_in_str.append(f"<{' '.join(q + [t])} {p}>_{n}.ptrs")
                    args_str.append(f"{n}")
                    result += f"    _{n} = _NDArray({n}, 2)\n"
                else:
                    args_in_str.append(f"<{' '.join(q + [t])} {p}>_{n}.ptrs")
                    t = f"ndarray[{t}, ndim=2]"
                    args_str.append(f'{" ".join([t])} {n}')
                    result += f"    _{n} = _NDArray({n})\n"
            elif p == "***":
                if t == "void":
                    args_in_str.append(f"<{' '.join(q + [t])} {p}>_{n}.ptrss")
                    args_str.append(f"{n}")
                    result += f"    _{n} = _NDArray({n}, 3)\n"
                else:
                    args_in_str.append(f"<{' '.join(q + [t])} {p}>_{n}.ptrss")
                    t = f"ndarray[{t}, ndim=3]"
                    args_str.append(f'{" ".join([t])} {n}')
                    result += f"    _{n} = _NDArray({n})\n"
            else:
                args_in_str.append(n)
                args_str.append(f'{" ".join(q + [t] + [p])}{n}')
        result = f"def {self.name}({', '.join(args_str)}):\n" + result
        result += f"    {lib}.{self.name}({', '.join(args_in_str)})\n"
        return result.replace("_Complex", "complex")


def parseHeader(header, include_path):
    print(f"Building wrapper from {os.path.join(include_path, header)}")
    try:
        from pycparser import parse_file, c_ast
    except ImportError or ModuleNotFoundError:
        from pyquda_core.pycparser.pycparser import parse_file, c_ast  # This is for the language server

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
        else:
            raise ValueError(f"Unknown node {node}")

    ast = parse_file(
        os.path.join(include_path, header), use_cpp=True, cpp_path="cc", cpp_args=["-E", Rf"-I{include_path}"]
    )
    funcs: List[FunctionMeta] = []
    enums: Dict[str, List[Tuple[str, int]]] = {}
    for node in ast:
        if isinstance(node.type, c_ast.FuncDecl):
            a = []
            if node.type.args is not None:
                for arg in node.type.args.params:
                    n, t = arg.name, arg.type
                    tt = t.type
                    if type(t) is c_ast.TypeDecl:
                        a.append((" ".join(t.quals), " ".join(tt.names), "", n, ""))
                    elif type(t) is c_ast.PtrDecl:
                        ttt = tt.type
                        if type(tt) is c_ast.TypeDecl:
                            a.append((" ".join(tt.quals), " ".join(ttt.names), "*", n, ""))
                        elif type(tt) is c_ast.PtrDecl:
                            tttt = ttt.type
                            if type(ttt) is c_ast.TypeDecl:
                                a.append((" ".join(ttt.quals), " ".join(tttt.names), "**", n, ""))
                            elif type(ttt) is c_ast.PtrDecl:
                                ttttt = tttt.type
                                if type(tttt) is c_ast.TypeDecl:
                                    a.append((" ".join(tttt.quals), " ".join(ttttt.names), "***", n, ""))
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
                                a.append((" ".join(tt.quals), " ".join(ttt.names), "", n, f"[{t.dim.value}]"))
                            else:
                                raise ValueError(f"Unexpected node {node}")
                        else:
                            if type(tt) is c_ast.TypeDecl:
                                a.append((" ".join(tt.quals), " ".join(ttt.names), "*", n, ""))
                            else:
                                raise ValueError(f"Unexpected node {node}")
                    else:
                        raise ValueError(f"Unexpected node {node}")
            ntt = node.type.type
            a = [] if a == [("", "void", "", None, "")] else a
            if type(ntt) is c_ast.TypeDecl:
                funcs.append(FunctionMeta((" ".join(ntt.type.names), ""), node.name, a))
            elif type(ntt) is c_ast.PtrDecl:
                nttt = ntt.type
                funcs.append(FunctionMeta((" ".join(nttt.type.names), "*"), node.name, a))
            else:
                raise ValueError(f"Unexpected node {node}")
        elif type(node.type.type) is c_ast.Enum:
            print(node.name)
            current_value = -1
            enums[node.name] = []
            for item in node.type.type.values:
                item_value = evaluate(item.value)
                if item_value is not None:
                    current_value = item_value
                else:
                    current_value += 1
                enums[node.name].append((item.name, current_value))
    return funcs, enums


def build_plugin_pyx(plugins_root, lib: str, header: str, include_path: str):
    lib_pxd = f'cdef extern from "{header}":\n'
    pylib_pyx = (
        "from enum import IntEnum\n"
        "from libcpp cimport bool\n"
        "from numpy cimport ndarray\n"
        "from pyquda_comm.pointer cimport Pointer, _NDArray\n"
        f"cimport {lib}\n"
        "\n"
    )
    pylib_pyi = (
        "from enum import IntEnum\n"
        "from numpy import int32, float64, complex128\n"
        "from numpy.typing import NDArray\n"
        "\n"
    )
    c_funcs, c_enums = parseHeader(header, include_path)
    for c_enum in c_enums.keys():
        lib_pxd += f"    ctypedef enum {c_enum}:\n        pass\n\n"
        pylib_pyx += f"class {c_enum}(IntEnum):\n"
        pylib_pyi += f"class {c_enum}(IntEnum):\n"
        for key, value in c_enums[c_enum]:
            pylib_pyx += f"    {key} = {value}\n"
            pylib_pyi += f"    {key} = {value}\n"
        pylib_pyx += "\n"
        pylib_pyi += "\n"
        _C_TO_PYTHON.update({c_enum: c_enum})
    for c_func in c_funcs:
        lib_pxd += f"    {c_func}\n"
        pylib_pyx += "\n" + c_func.pyx(lib)
        pylib_pyi += c_func.pyi()
    for c_enum in c_enums.keys():
        pylib_pyx = pylib_pyx.replace(f"{c_enum} ", f"{lib}.{c_enum} ")

    os.makedirs(os.path.join(plugins_root, f"py{lib}", "src"), exist_ok=True)
    with open(os.path.join(plugins_root, f"py{lib}", "src", f"{lib}.pxd"), "w") as f:
        f.write(lib_pxd)
    with open(os.path.join(plugins_root, f"py{lib}", "src", f"_py{lib}.pyx"), "w") as f:
        f.write(pylib_pyx)
    with open(os.path.join(plugins_root, f"py{lib}", f"_py{lib}.pyi"), "w") as f:
        f.write(pylib_pyi)

    if os.path.exists(os.path.join(plugins_root, "yacctab.py")):
        os.remove(os.path.join(plugins_root, "yacctab.py"))
    if os.path.exists(os.path.join(plugins_root, "lextab.py")):
        os.remove(os.path.join(plugins_root, "lextab.py"))
