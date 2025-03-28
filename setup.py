import os
import sys
from setuptools import Extension, setup


def dynamic(lib, header, include_path, library_path):
    from plugin_pyx import Plugin

    plugin = Plugin(lib, header, include_path)
    plugin.write(os.path.join(os.path.dirname(__file__), "pyquda_plugins", f"py{lib}"))
    return Extension(
        name=f"pyquda_plugins.py{lib}.py{lib}",
        sources=[f"pyquda_plugins/py{lib}/src/py{lib}.pyx"],
        include_dirs=[include_path, numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        library_dirs=[library_path],
        libraries=[lib],
        extra_link_args=[f"-Wl,-rpath={library_path}"],
        language="c",
    )


if "egg_info" in sys.argv or "dist_info" in sys.argv or "sdist" in sys.argv:
    describe = os.popen("git describe --tags", "r").read().strip()
    if describe != "":
        if "-" in describe:
            tag, post, _ = describe.split("-")
        else:
            tag, post = describe, 0
        with open(os.path.join(os.path.dirname(__file__), "pyquda_utils", "_version.py"), "w") as f:
            f.write(f'__version__ = "{tag[1:]}.dev{post}"\n')
    setup()
elif "PYQUDA_PLUGINS" in os.environ:
    from Cython.Build import cythonize
    import numpy

    libs = os.environ["PYQUDA_PLUGINS"].split(",")
    for i, lib in enumerate(libs):
        lib = lib.upper()
        libs[i] = dynamic(
            lib.lower(),
            os.environ[f"{lib}_HEADER"],
            os.environ[f"{lib}_INCLUDE_PATH"],
            os.environ[f"{lib}_LIBRARY_PATH"],
        )
    setup(ext_modules=cythonize(libs, language_level="3"))
else:
    setup()
