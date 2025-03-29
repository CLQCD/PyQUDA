import os
import sys
from setuptools import setup


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
