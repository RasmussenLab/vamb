# python build_vambtools_c.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize

setup(name='vambtools_c', ext_modules=cythonize("vambtools_c.pyx"))