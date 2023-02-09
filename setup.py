from setuptools import Extension, setup

setup(ext_modules=[Extension(name="vamb._vambtools", sources=["src/_vambtools.pyx"])])
