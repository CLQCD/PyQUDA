import os
import sys
from setuptools import Extension, setup
from pyquda_pyx import build_pyquda_pyx

if "egg_info" in sys.argv or "dist_info" in sys.argv or "sdist" in sys.argv:
    setup()
elif "QUDA_PATH" in os.environ:
    quda_path = os.path.realpath(os.environ["QUDA_PATH"])
    build_pyquda_pyx(os.path.dirname(__file__), quda_path)
    if os.path.exists(os.path.join(quda_path, "lib", "libquda.so")) or os.path.exists(
        os.path.join(quda_path, "lib", "quda.dll")
    ):
        _STATIC = False
    elif os.path.exists(os.path.join(quda_path, "lib", "libquda.a")) or os.path.exists(
        os.path.join(quda_path, "lib", "quda.lib")
    ):
        _STATIC = True
    else:
        raise FileNotFoundError(f"Cannot find libquda.so or libquda.a in {os.path.join(quda_path, 'lib')}")

    from Cython.Build import cythonize
    import numpy

    extensions = cythonize(
        [
            Extension(
                name="pyquda_comm.pointer",
                sources=["pyquda_comm/src/pointer.pyx"],
                language="c",
            ),
            Extension(
                name="pyquda.pyquda",
                sources=["pyquda/src/pyquda.pyx"],
                include_dirs=[os.path.join(quda_path, "include"), numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                library_dirs=[os.path.join(quda_path, "lib")],
                libraries=["quda"],
                extra_link_args=[f"-Wl,-rpath={os.path.join(quda_path, 'lib')}"] if not _STATIC else None,
                language="c",
            ),
            Extension(
                name="pyquda.malloc_pyquda",
                sources=["pyquda/src/malloc_pyquda.pyx"],
                include_dirs=[os.path.join(quda_path, "include")],
                library_dirs=[os.path.join(quda_path, "lib")],
                libraries=["quda"],
                extra_link_args=[f"-Wl,-rpath={os.path.join(quda_path, 'lib')}"] if not _STATIC else None,
                language="c++",
            ),
        ],
        language_level="3",
    )

    setup(ext_modules=extensions)
else:
    raise EnvironmentError("QUDA_PATH environment is needed to link against libquda")
