"""Build the deterministic SENT Cython extension shipped with ELDA."""

from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup


extensions = [
    Extension(
        "elda.datamodules.data.sent_utils",
        ["elda/datamodules/data/sent_utils.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
)
