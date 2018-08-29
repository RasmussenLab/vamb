# python build_vambtools.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize

setup(name='_vambtools', ext_modules=cythonize("_vambtools.pyx"))
