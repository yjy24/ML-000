import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="Cython Update State",
    ext_modules=cythonize("target_encoding.pyx"),
    include_dirs=[np.get_include()]
)
