# setup.py

from distutils.core import setup
from Cython.Build import cythonize

setup(
	ext_modules=cythonize(
		"cyclic_multi_factor_pow_seq_cython.pyx", annotate=True,
	),
	include_dirs=['/usr/include/python3.13'],
)
