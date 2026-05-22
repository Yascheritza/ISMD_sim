import os
import sys

from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.errors import DistutilsPlatformError
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize


def check_compiler_type():
    try:
        compiler = new_compiler()
        return compiler.compiler_type
    except DistutilsPlatformError:
        sys.exit("Error: No valid C++ compiler found.")


compiler_type = check_compiler_type()
if compiler_type == "msvc":
    # Use /O2 for maximum optimization on MSVC
    compile_args = ["/std:c++20", "/O2"]
    link_args = []  # MSVC typically does not need optimization flags for linking
else:
    # For GCC/Clang, add -O3 and -march=native for maximum optimization
    compile_args = ["-std=c++20", "-O3", "-march=native"]
    link_args = compile_args


extensions = [
    Extension(
        name="simulation.SpatialBirthDeath",
        sources=[
            "simulation/SpatialBirthDeathWrapper.pyx",
            "src/SpatialBirthDeath.cpp"
        ],
        language="c++",
        include_dirs=[os.path.abspath("include")],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
]

setup(
    name="spatial_sim",
    version="1.0",
    description="Spatial birth-death point process simulator",
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={"language_level": "3"}),
    packages=find_packages(),
)
