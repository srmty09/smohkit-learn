from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

ext_modules = [
    Pybind11Extension(
        "ENGINE",  # This will be your module name
        ["ENGINE.cpp"],  # Your C++ file name
        cxx_std=17,
        include_dirs=[pybind11.get_cmake_dir() + "/../../../include"],
    ),
]

setup(
    name="ENGINE",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
