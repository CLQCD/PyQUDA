import os
import sys
from setuptools import setup

if "egg_info" in sys.argv or "sdist" in sys.argv:
    describe = os.popen("git describe --tags", "r").read()
    if describe != "":
        tag, post, hash = describe.strip().split("-")
        with open(os.path.join(os.path.dirname(__file__), "pyquda_utils", "_version.py"), "w") as f:
            f.write(f'__version__ = "{tag[1:]}.dev{post}"\n')

setup()
