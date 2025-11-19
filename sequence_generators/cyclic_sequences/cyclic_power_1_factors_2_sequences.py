#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import itertools
import os
import pdb
import re
import sqlite3
import sys
import traceback

import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

# own import
# execute before: python3.13 setup_cyclic_multi_factor_pow_seq.py build_ext --inplace
import cyclic_multi_factor_pow_seq_cython

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def get_f(modulo, t_fac):
	# TODO: try other variants e.g. for modulo 4
	"""
		new =
			x0 * t00 + (x0+1) * t01 + (x0+2) * t02 + (x0+3) * t03 +
			x1 * t10 + (x1+1) * t11 + (x1+2) * t12 + (x1+3) * t13 +
			t_const
	"""

	def f_pow_8(t_x):
		return (
			t_x[0]**8 * t_x[1]**8 * t_fac[0] +

			t_x[0]**7 * t_x[1]**8 * t_fac[1] +
			t_x[0]**8 * t_x[1]**7 * t_fac[2] +

			t_x[0]**6 * t_x[1]**8 * t_fac[3] +
			t_x[0]**8 * t_x[1]**6 * t_fac[4] +

			t_x[0]**5 * t_x[1]**8 * t_fac[5] +
			t_x[0]**8 * t_x[1]**5 * t_fac[6] +

			t_x[0]**4 * t_x[1]**8 * t_fac[7] +
			t_x[0]**8 * t_x[1]**4 * t_fac[8] +

			t_x[0]**3 * t_x[1]**8 * t_fac[9] +
			t_x[0]**8 * t_x[1]**3 * t_fac[10] +

			t_x[0]**2 * t_x[1]**8 * t_fac[11] +
			t_x[0]**8 * t_x[1]**2 * t_fac[12] +

			t_x[0]**1 * t_x[1]**8 * t_fac[13] +
			t_x[0]**8 * t_x[1]**1 * t_fac[14] +

			t_x[0]**0 * t_x[1]**8 * t_fac[15] +
			t_x[0]**8 * t_x[1]**0 * t_fac[16] +


			t_x[0]**7 * t_x[1]**7 * t_fac[17] +

			t_x[0]**6 * t_x[1]**7 * t_fac[18] +
			t_x[0]**7 * t_x[1]**6 * t_fac[19] +

			t_x[0]**5 * t_x[1]**7 * t_fac[20] +
			t_x[0]**7 * t_x[1]**5 * t_fac[21] +

			t_x[0]**4 * t_x[1]**7 * t_fac[22] +
			t_x[0]**7 * t_x[1]**4 * t_fac[23] +

			t_x[0]**3 * t_x[1]**7 * t_fac[24] +
			t_x[0]**7 * t_x[1]**3 * t_fac[25] +

			t_x[0]**2 * t_x[1]**7 * t_fac[26] +
			t_x[0]**7 * t_x[1]**2 * t_fac[27] +

			t_x[0]**1 * t_x[1]**7 * t_fac[28] +
			t_x[0]**7 * t_x[1]**1 * t_fac[29] +

			t_x[0]**0 * t_x[1]**7 * t_fac[30] +
			t_x[0]**7 * t_x[1]**0 * t_fac[31] +


			t_x[0]**6 * t_x[1]**6 * t_fac[32] +

			t_x[0]**5 * t_x[1]**6 * t_fac[33] +
			t_x[0]**6 * t_x[1]**5 * t_fac[34] +

			t_x[0]**4 * t_x[1]**6 * t_fac[35] +
			t_x[0]**6 * t_x[1]**4 * t_fac[36] +

			t_x[0]**3 * t_x[1]**6 * t_fac[37] +
			t_x[0]**6 * t_x[1]**3 * t_fac[38] +

			t_x[0]**2 * t_x[1]**6 * t_fac[39] +
			t_x[0]**6 * t_x[1]**2 * t_fac[40] +

			t_x[0]**1 * t_x[1]**6 * t_fac[41] +
			t_x[0]**6 * t_x[1]**1 * t_fac[42] +

			t_x[0]**0 * t_x[1]**6 * t_fac[43] +
			t_x[0]**6 * t_x[1]**0 * t_fac[44] +


			t_x[0]**5 * t_x[1]**5 * t_fac[45] +

			t_x[0]**4 * t_x[1]**5 * t_fac[46] +
			t_x[0]**5 * t_x[1]**4 * t_fac[47] +

			t_x[0]**3 * t_x[1]**5 * t_fac[48] +
			t_x[0]**5 * t_x[1]**3 * t_fac[49] +

			t_x[0]**2 * t_x[1]**5 * t_fac[50] +
			t_x[0]**5 * t_x[1]**2 * t_fac[51] +

			t_x[0]**1 * t_x[1]**5 * t_fac[52] +
			t_x[0]**5 * t_x[1]**1 * t_fac[53] +

			t_x[0]**0 * t_x[1]**5 * t_fac[54] +
			t_x[0]**5 * t_x[1]**0 * t_fac[55] +


			t_x[0]**4 * t_x[1]**4 * t_fac[56] +

			t_x[0]**3 * t_x[1]**4 * t_fac[57] +
			t_x[0]**4 * t_x[1]**3 * t_fac[58] +

			t_x[0]**2 * t_x[1]**4 * t_fac[59] +
			t_x[0]**4 * t_x[1]**2 * t_fac[60] +

			t_x[0]**1 * t_x[1]**4 * t_fac[61] +
			t_x[0]**4 * t_x[1]**1 * t_fac[62] +

			t_x[0]**0 * t_x[1]**4 * t_fac[63] +
			t_x[0]**4 * t_x[1]**0 * t_fac[64] +


			t_x[0]**3 * t_x[1]**3 * t_fac[65] +

			t_x[0]**2 * t_x[1]**3 * t_fac[66] +
			t_x[0]**3 * t_x[1]**2 * t_fac[67] +

			t_x[0]**1 * t_x[1]**3 * t_fac[68] +
			t_x[0]**3 * t_x[1]**1 * t_fac[69] +

			t_x[0]**0 * t_x[1]**3 * t_fac[70] +
			t_x[0]**3 * t_x[1]**0 * t_fac[71] +

			
			t_x[0]**2 * t_x[1]**2 * t_fac[72] +

			t_x[0]**1 * t_x[1]**2 * t_fac[73] +
			t_x[0]**2 * t_x[1]**1 * t_fac[74] +

			t_x[0]**0 * t_x[1]**2 * t_fac[75] +
			t_x[0]**2 * t_x[1]**0 * t_fac[76] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[77] +

			t_x[0]**0 * t_x[1]**1 * t_fac[78] +
			t_x[0]**1 * t_x[1]**0 * t_fac[79] +


			t_x[0]**0 * t_x[1]**0 * t_fac[80]
		) % modulo

	def f_pow_7(t_x):
		return (
			t_x[0]**7 * t_x[1]**7 * t_fac[0] +

			t_x[0]**6 * t_x[1]**7 * t_fac[1] +
			t_x[0]**7 * t_x[1]**6 * t_fac[2] +

			t_x[0]**5 * t_x[1]**7 * t_fac[3] +
			t_x[0]**7 * t_x[1]**5 * t_fac[4] +

			t_x[0]**4 * t_x[1]**7 * t_fac[5] +
			t_x[0]**7 * t_x[1]**4 * t_fac[6] +

			t_x[0]**3 * t_x[1]**7 * t_fac[7] +
			t_x[0]**7 * t_x[1]**3 * t_fac[8] +

			t_x[0]**2 * t_x[1]**7 * t_fac[9] +
			t_x[0]**7 * t_x[1]**2 * t_fac[10] +

			t_x[0]**1 * t_x[1]**7 * t_fac[11] +
			t_x[0]**7 * t_x[1]**1 * t_fac[12] +

			t_x[0]**0 * t_x[1]**7 * t_fac[13] +
			t_x[0]**7 * t_x[1]**0 * t_fac[14] +


			t_x[0]**6 * t_x[1]**6 * t_fac[15] +

			t_x[0]**5 * t_x[1]**6 * t_fac[16] +
			t_x[0]**6 * t_x[1]**5 * t_fac[17] +

			t_x[0]**4 * t_x[1]**6 * t_fac[18] +
			t_x[0]**6 * t_x[1]**4 * t_fac[19] +

			t_x[0]**3 * t_x[1]**6 * t_fac[20] +
			t_x[0]**6 * t_x[1]**3 * t_fac[21] +

			t_x[0]**2 * t_x[1]**6 * t_fac[22] +
			t_x[0]**6 * t_x[1]**2 * t_fac[23] +

			t_x[0]**1 * t_x[1]**6 * t_fac[24] +
			t_x[0]**6 * t_x[1]**1 * t_fac[25] +

			t_x[0]**0 * t_x[1]**6 * t_fac[26] +
			t_x[0]**6 * t_x[1]**0 * t_fac[27] +


			t_x[0]**5 * t_x[1]**5 * t_fac[28] +

			t_x[0]**4 * t_x[1]**5 * t_fac[29] +
			t_x[0]**5 * t_x[1]**4 * t_fac[30] +

			t_x[0]**3 * t_x[1]**5 * t_fac[31] +
			t_x[0]**5 * t_x[1]**3 * t_fac[32] +

			t_x[0]**2 * t_x[1]**5 * t_fac[33] +
			t_x[0]**5 * t_x[1]**2 * t_fac[34] +

			t_x[0]**1 * t_x[1]**5 * t_fac[35] +
			t_x[0]**5 * t_x[1]**1 * t_fac[36] +

			t_x[0]**0 * t_x[1]**5 * t_fac[37] +
			t_x[0]**5 * t_x[1]**0 * t_fac[38] +


			t_x[0]**4 * t_x[1]**4 * t_fac[39] +

			t_x[0]**3 * t_x[1]**4 * t_fac[40] +
			t_x[0]**4 * t_x[1]**3 * t_fac[41] +

			t_x[0]**2 * t_x[1]**4 * t_fac[42] +
			t_x[0]**4 * t_x[1]**2 * t_fac[43] +

			t_x[0]**1 * t_x[1]**4 * t_fac[44] +
			t_x[0]**4 * t_x[1]**1 * t_fac[45] +

			t_x[0]**0 * t_x[1]**4 * t_fac[46] +
			t_x[0]**4 * t_x[1]**0 * t_fac[47] +


			t_x[0]**3 * t_x[1]**3 * t_fac[48] +

			t_x[0]**2 * t_x[1]**3 * t_fac[49] +
			t_x[0]**3 * t_x[1]**2 * t_fac[50] +

			t_x[0]**1 * t_x[1]**3 * t_fac[51] +
			t_x[0]**3 * t_x[1]**1 * t_fac[52] +

			t_x[0]**0 * t_x[1]**3 * t_fac[53] +
			t_x[0]**3 * t_x[1]**0 * t_fac[54] +

			
			t_x[0]**2 * t_x[1]**2 * t_fac[55] +

			t_x[0]**1 * t_x[1]**2 * t_fac[56] +
			t_x[0]**2 * t_x[1]**1 * t_fac[57] +

			t_x[0]**0 * t_x[1]**2 * t_fac[58] +
			t_x[0]**2 * t_x[1]**0 * t_fac[59] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[60] +

			t_x[0]**0 * t_x[1]**1 * t_fac[61] +
			t_x[0]**1 * t_x[1]**0 * t_fac[62] +


			t_x[0]**0 * t_x[1]**0 * t_fac[63]
		) % modulo

	def f_pow_6(t_x):
		return (
			t_x[0]**6 * t_x[1]**6 * t_fac[0] +

			t_x[0]**5 * t_x[1]**6 * t_fac[1] +
			t_x[0]**6 * t_x[1]**5 * t_fac[2] +

			t_x[0]**4 * t_x[1]**6 * t_fac[3] +
			t_x[0]**6 * t_x[1]**4 * t_fac[4] +

			t_x[0]**3 * t_x[1]**6 * t_fac[5] +
			t_x[0]**6 * t_x[1]**3 * t_fac[6] +

			t_x[0]**2 * t_x[1]**6 * t_fac[7] +
			t_x[0]**6 * t_x[1]**2 * t_fac[8] +

			t_x[0]**1 * t_x[1]**6 * t_fac[9] +
			t_x[0]**6 * t_x[1]**1 * t_fac[10] +

			t_x[0]**0 * t_x[1]**6 * t_fac[11] +
			t_x[0]**6 * t_x[1]**0 * t_fac[12] +


			t_x[0]**5 * t_x[1]**5 * t_fac[13] +

			t_x[0]**4 * t_x[1]**5 * t_fac[14] +
			t_x[0]**5 * t_x[1]**4 * t_fac[15] +

			t_x[0]**3 * t_x[1]**5 * t_fac[16] +
			t_x[0]**5 * t_x[1]**3 * t_fac[17] +

			t_x[0]**2 * t_x[1]**5 * t_fac[18] +
			t_x[0]**5 * t_x[1]**2 * t_fac[19] +

			t_x[0]**1 * t_x[1]**5 * t_fac[20] +
			t_x[0]**5 * t_x[1]**1 * t_fac[21] +

			t_x[0]**0 * t_x[1]**5 * t_fac[22] +
			t_x[0]**5 * t_x[1]**0 * t_fac[23] +


			t_x[0]**4 * t_x[1]**4 * t_fac[24] +

			t_x[0]**3 * t_x[1]**4 * t_fac[25] +
			t_x[0]**4 * t_x[1]**3 * t_fac[26] +

			t_x[0]**2 * t_x[1]**4 * t_fac[27] +
			t_x[0]**4 * t_x[1]**2 * t_fac[28] +

			t_x[0]**1 * t_x[1]**4 * t_fac[29] +
			t_x[0]**4 * t_x[1]**1 * t_fac[30] +

			t_x[0]**0 * t_x[1]**4 * t_fac[31] +
			t_x[0]**4 * t_x[1]**0 * t_fac[32] +


			t_x[0]**3 * t_x[1]**3 * t_fac[33] +

			t_x[0]**2 * t_x[1]**3 * t_fac[34] +
			t_x[0]**3 * t_x[1]**2 * t_fac[35] +

			t_x[0]**1 * t_x[1]**3 * t_fac[36] +
			t_x[0]**3 * t_x[1]**1 * t_fac[37] +

			t_x[0]**0 * t_x[1]**3 * t_fac[38] +
			t_x[0]**3 * t_x[1]**0 * t_fac[39] +

			
			t_x[0]**2 * t_x[1]**2 * t_fac[40] +

			t_x[0]**1 * t_x[1]**2 * t_fac[41] +
			t_x[0]**2 * t_x[1]**1 * t_fac[42] +

			t_x[0]**0 * t_x[1]**2 * t_fac[43] +
			t_x[0]**2 * t_x[1]**0 * t_fac[44] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[45] +

			t_x[0]**0 * t_x[1]**1 * t_fac[46] +
			t_x[0]**1 * t_x[1]**0 * t_fac[47] +


			t_x[0]**0 * t_x[1]**0 * t_fac[48]
		) % modulo

	def f_pow_5(t_x):
		return (
			t_x[0]**5 * t_x[1]**5 * t_fac[0] +

			t_x[0]**4 * t_x[1]**5 * t_fac[1] +
			t_x[0]**5 * t_x[1]**4 * t_fac[2] +

			t_x[0]**3 * t_x[1]**5 * t_fac[3] +
			t_x[0]**5 * t_x[1]**3 * t_fac[4] +

			t_x[0]**2 * t_x[1]**5 * t_fac[5] +
			t_x[0]**5 * t_x[1]**2 * t_fac[6] +

			t_x[0]**1 * t_x[1]**5 * t_fac[7] +
			t_x[0]**5 * t_x[1]**1 * t_fac[8] +

			t_x[0]**0 * t_x[1]**5 * t_fac[9] +
			t_x[0]**5 * t_x[1]**0 * t_fac[10] +


			t_x[0]**4 * t_x[1]**4 * t_fac[11] +

			t_x[0]**3 * t_x[1]**4 * t_fac[12] +
			t_x[0]**4 * t_x[1]**3 * t_fac[13] +

			t_x[0]**2 * t_x[1]**4 * t_fac[14] +
			t_x[0]**4 * t_x[1]**2 * t_fac[15] +

			t_x[0]**1 * t_x[1]**4 * t_fac[16] +
			t_x[0]**4 * t_x[1]**1 * t_fac[17] +

			t_x[0]**0 * t_x[1]**4 * t_fac[18] +
			t_x[0]**4 * t_x[1]**0 * t_fac[19] +


			t_x[0]**3 * t_x[1]**3 * t_fac[20] +

			t_x[0]**2 * t_x[1]**3 * t_fac[21] +
			t_x[0]**3 * t_x[1]**2 * t_fac[22] +

			t_x[0]**1 * t_x[1]**3 * t_fac[23] +
			t_x[0]**3 * t_x[1]**1 * t_fac[24] +

			t_x[0]**0 * t_x[1]**3 * t_fac[25] +
			t_x[0]**3 * t_x[1]**0 * t_fac[26] +

			
			t_x[0]**2 * t_x[1]**2 * t_fac[27] +

			t_x[0]**1 * t_x[1]**2 * t_fac[28] +
			t_x[0]**2 * t_x[1]**1 * t_fac[29] +

			t_x[0]**0 * t_x[1]**2 * t_fac[30] +
			t_x[0]**2 * t_x[1]**0 * t_fac[31] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[32] +

			t_x[0]**0 * t_x[1]**1 * t_fac[33] +
			t_x[0]**1 * t_x[1]**0 * t_fac[34] +


			t_x[0]**0 * t_x[1]**0 * t_fac[35]
		) % modulo

	def f_pow_4(t_x):
		return (
			t_x[0]**4 * t_x[1]**4 * t_fac[0] +

			t_x[0]**3 * t_x[1]**4 * t_fac[1] +
			t_x[0]**4 * t_x[1]**3 * t_fac[2] +

			t_x[0]**2 * t_x[1]**4 * t_fac[3] +
			t_x[0]**4 * t_x[1]**2 * t_fac[4] +

			t_x[0]**1 * t_x[1]**4 * t_fac[5] +
			t_x[0]**4 * t_x[1]**1 * t_fac[6] +

			t_x[0]**0 * t_x[1]**4 * t_fac[7] +
			t_x[0]**4 * t_x[1]**0 * t_fac[8] +


			t_x[0]**3 * t_x[1]**3 * t_fac[9] +

			t_x[0]**2 * t_x[1]**3 * t_fac[10] +
			t_x[0]**3 * t_x[1]**2 * t_fac[11] +

			t_x[0]**1 * t_x[1]**3 * t_fac[12] +
			t_x[0]**3 * t_x[1]**1 * t_fac[13] +

			t_x[0]**0 * t_x[1]**3 * t_fac[14] +
			t_x[0]**3 * t_x[1]**0 * t_fac[15] +

			
			t_x[0]**2 * t_x[1]**2 * t_fac[16] +

			t_x[0]**1 * t_x[1]**2 * t_fac[17] +
			t_x[0]**2 * t_x[1]**1 * t_fac[18] +

			t_x[0]**0 * t_x[1]**2 * t_fac[19] +
			t_x[0]**2 * t_x[1]**0 * t_fac[20] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[21] +

			t_x[0]**0 * t_x[1]**1 * t_fac[22] +
			t_x[0]**1 * t_x[1]**0 * t_fac[23] +


			t_x[0]**0 * t_x[1]**0 * t_fac[24]
		) % modulo

	def f_pow_3(t_x):
		return (
			t_x[0]**3 * t_x[1]**3 * t_fac[0] +

			t_x[0]**2 * t_x[1]**3 * t_fac[1] +
			t_x[0]**3 * t_x[1]**2 * t_fac[2] +

			t_x[0]**1 * t_x[1]**3 * t_fac[3] +
			t_x[0]**3 * t_x[1]**1 * t_fac[4] +

			t_x[0]**0 * t_x[1]**3 * t_fac[5] +
			t_x[0]**3 * t_x[1]**0 * t_fac[6] +

			
			t_x[0]**2 * t_x[1]**2 * t_fac[7] +

			t_x[0]**1 * t_x[1]**2 * t_fac[8] +
			t_x[0]**2 * t_x[1]**1 * t_fac[9] +

			t_x[0]**0 * t_x[1]**2 * t_fac[10] +
			t_x[0]**2 * t_x[1]**0 * t_fac[11] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[12] +

			t_x[0]**0 * t_x[1]**1 * t_fac[13] +
			t_x[0]**1 * t_x[1]**0 * t_fac[14] +


			t_x[0]**0 * t_x[1]**0 * t_fac[15]
		) % modulo

	def f_pow_2(t_x):
		return (
			t_x[0]**2 * t_x[1]**2 * t_fac[0] +

			t_x[0]**1 * t_x[1]**2 * t_fac[1] +
			t_x[0]**2 * t_x[1]**1 * t_fac[2] +

			t_x[0]**0 * t_x[1]**2 * t_fac[3] +
			t_x[0]**2 * t_x[1]**0 * t_fac[4] +
			

			t_x[0]**1 * t_x[1]**1 * t_fac[5] +

			t_x[0]**0 * t_x[1]**1 * t_fac[6] +
			t_x[0]**1 * t_x[1]**0 * t_fac[7] +


			t_x[0]**0 * t_x[1]**0 * t_fac[8]
		) % modulo

	def f_pow_1(t_x):
		return (
			t_x[0]**1 * t_x[1]**1 * t_fac[0] +

			t_x[0]**0 * t_x[1]**1 * t_fac[1] +
			t_x[0]**1 * t_x[1]**0 * t_fac[2] +


			t_x[0]**0 * t_x[1]**0 * t_fac[3]
		) % modulo

	f = None

	match len(t_fac):
		case 4: f = f_pow_1
		case 9: f = f_pow_2
		case 16: f = f_pow_3
		case 25: f = f_pow_4
		case 36: f = f_pow_5
		case 49: f = f_pow_6
		case 64: f = f_pow_7
		case 81: f = f_pow_8
		case _: f = None

	assert not f is None

	return f

def proc_func(modulo, l_t_fac, factor_len, ):
	print(f'proc doing now modulo {modulo}')
	d_cycle_len_to_d_t_fac_to_s_l_cycle = {}
	found_max_cycle_len = 0
	arr_factors = np.zeros((factor_len, ), dtype=np.int64)
	arr_values = np.zeros((modulo*modulo*2, ), dtype=np.int64)
	for t_fac in l_t_fac:
		l_l_t_x_cycle = []
		# print(f"modulo: {modulo}, t_fac: {t_fac}")

		# do it with the faster one! compiled with cython
		if factor_len == 9 and False:
		# if factor_len == 9 and True:
			arr_factors[:] = t_fac
			arr_values[:] = 0
			ret_val = cyclic_multi_factor_pow_seq_cython.calc_2_value_sequence_pow_2(modulo, arr_factors, arr_values)
			
			if ret_val == 1:
				l_l_t_x_cycle.append([tuple(l) for l in arr_values.reshape((-1, 2)).tolist()])
			
			continue

		f = get_f(modulo=modulo, t_fac=t_fac)
		# create all possible starting positions
		# s_all = set(itertools.product(*[list(range(0, modulo)) for _ in range(0, 2)]))

		# start position: starting from all values zero
		t_x = (0, 0)
		l_t_x = [t_x]
		s_t_x = set(l_t_x)
		x1, x2 = t_x
		

		x1_next = x2
		x2_next = f(t_x=(x1, x2))
		t_x_next = (x1_next, x2_next)
		while not t_x_next in s_t_x:
			t_x = t_x_next
			l_t_x.append(t_x)
			s_t_x.add(t_x)

			x1, x2 = t_x
			x1_next = x2
			x2_next = f(t_x=(x1, x2))
			t_x_next = (x1_next, x2_next)

		# end position should be the starting one, where all values are zero
		if t_x_next != (0, 0):
			continue

		if len(l_t_x) != modulo**2:
			continue

		l_l_t_x_cycle.append(l_t_x)

		# # find all possible cycles
		# while s_all:
		# 	t_x = s_all.pop()
		# 	l_t_x = [t_x]

		# 	x1, x2 = t_x
		# 	x1_next = x2
		# 	x2_next = f(t_x=(x1, x2))
		# 	t_x_next = (x1_next, x2_next)
		# 	while t_x_next in s_all:
		# 		t_x = t_x_next
		# 		s_all.remove(t_x)
		# 		l_t_x.append(t_x)

		# 		x1, x2 = t_x
		# 		x1_next = x2
		# 		x2_next = f(t_x=(x1, x2))
		# 		t_x_next = (x1_next, x2_next)

		# 	if t_x_next not in l_t_x:
		# 		break

		# 	i_l_t_x = l_t_x.index(t_x_next)
		# 	l_t_x_cycle = l_t_x[i_l_t_x:]
		# 	t_x_min = min(l_t_x_cycle)
		# 	i_t_x_min = l_t_x_cycle.index(t_x_min)
		# 	l_t_x_cycle_min = l_t_x_cycle[i_t_x_min:] + l_t_x_cycle[:i_t_x_min]

		# 	# check if the cycle is closed
		# 	t_x_last = l_t_x_cycle_min[-1]
		# 	x1_last, x2_last = t_x_last
			
		# 	x1_last_first = x2_last
		# 	x2_last_first = f(t_x=(x1_last, x2_last))
		# 	t_x_last_first = (x1_last_first, x2_last_first)

		# 	if l_t_x_cycle_min[0] != t_x_last_first:
		# 		continue

		# 	l_l_t_x_cycle.append(l_t_x_cycle_min)

		# choose only the sequences, which starts with (0, 0)
		l_l_t_x_cycle = [l_t_x_cycle for l_t_x_cycle in l_l_t_x_cycle if l_t_x_cycle[0] == (0, 0)]
		if len(l_l_t_x_cycle) == 0:
			continue

		# leave only the longest cycles
		l_len = [len(l_t_x_cycle) for l_t_x_cycle in l_l_t_x_cycle]
		max_cycle_len = max(l_len)

		if found_max_cycle_len < max_cycle_len:
			found_max_cycle_len = max_cycle_len
			del d_cycle_len_to_d_t_fac_to_s_l_cycle
			d_cycle_len_to_d_t_fac_to_s_l_cycle = {}

		for l_t_x_cycle in l_l_t_x_cycle:
			cycle_len = len(l_t_x_cycle)
			if cycle_len == found_max_cycle_len:
				if cycle_len not in d_cycle_len_to_d_t_fac_to_s_l_cycle:
					d_cycle_len_to_d_t_fac_to_s_l_cycle[cycle_len] = {}
				d_t_fac_to_s_l_cycle = d_cycle_len_to_d_t_fac_to_s_l_cycle[cycle_len]

				if t_fac not in d_t_fac_to_s_l_cycle:
					d_t_fac_to_s_l_cycle[t_fac] = set()
				s_l_cycle = d_t_fac_to_s_l_cycle[t_fac]

				t_t_x_cycle = tuple(l_t_x_cycle)
				if not t_t_x_cycle in s_l_cycle:
					s_l_cycle.add(t_t_x_cycle)
	
	ret_val = {
		'modulo': modulo,
		'd_cycle_len_to_d_t_fac_to_s_l_cycle': d_cycle_len_to_d_t_fac_to_s_l_cycle,
	}

	return ret_val


def main(modulo, factor_len, amount_factor, jump_factor, additional_values=1):
	l_arguments = []

	if factor_len >= 25 and False:
		arr_fac_all = np.zeros((
			(factor_len-1)*(modulo-1)*(modulo-1) +
			(factor_len-1)*(factor_len-2)//2*(modulo-1)*(modulo-1)*(modulo-1), factor_len), dtype=np.int64)

		l_pos_fac_1 = list(itertools.combinations(list(range(0, factor_len-1)), 1))
		l_pos_fac_2 = list(itertools.combinations(list(range(0, factor_len-1)), 2))
		
		i_row_acc = 0
		for v1 in range(1, modulo):
			for v2 in range(1, modulo):
				for i_row, (i1, ) in enumerate(l_pos_fac_1, i_row_acc):
					arr_fac_all[i_row, i1] = v1
					arr_fac_all[i_row, factor_len-1] = v2
				i_row_acc += len(l_pos_fac_1)

		for v1 in range(1, modulo):
			for v2 in range(1, modulo):
				for v3 in range(1, modulo):
					for i_row, (i1, i2) in enumerate(l_pos_fac_2, i_row_acc):
						arr_fac_all[i_row, i1] = v1
						arr_fac_all[i_row, i2] = v2
						arr_fac_all[i_row, factor_len-1] = v3
					i_row_acc += len(l_pos_fac_2)

		for arr_idx_split in np.array_split(np.arange(0, arr_fac_all.shape[0]), 100):
			l_args_part = []
			arr_fac_split = arr_fac_all[arr_idx_split]
			for arr_fac_row in arr_fac_split:
				l_args_part.append(tuple(arr_fac_row.tolist()))
			l_arguments.append((modulo, l_args_part))

		assert arr_fac.shape[0] == i_row_acc
	elif factor_len >= 4 and False:
		arr_fac_all = np.zeros((
			(factor_len-1)*(modulo-1)*(modulo-1) +
			(factor_len-1)*(factor_len-2)//2*(modulo-1)*(modulo-1)*(modulo-1) +
			(factor_len-1)*(factor_len-2)*(factor_len-3)//2//3*(modulo-1)*(modulo-1)*(modulo-1)*(modulo-1), factor_len), dtype=np.int64)

		l_pos_fac_1 = list(itertools.combinations(list(range(0, factor_len-1)), 1))
		l_pos_fac_2 = list(itertools.combinations(list(range(0, factor_len-1)), 2))
		l_pos_fac_3 = list(itertools.combinations(list(range(0, factor_len-1)), 3))
		
		i_row_acc = 0
		for v1 in range(1, modulo):
			for v2 in range(1, modulo):
				for i_row, (i1, ) in enumerate(l_pos_fac_1, i_row_acc):
					arr_fac_all[i_row, i1] = v1
					arr_fac_all[i_row, factor_len-1] = v2
				i_row_acc += len(l_pos_fac_1)

		for v1 in range(1, modulo):
			for v2 in range(1, modulo):
				for v3 in range(1, modulo):
					for i_row, (i1, i2) in enumerate(l_pos_fac_2, i_row_acc):
						arr_fac_all[i_row, i1] = v1
						arr_fac_all[i_row, i2] = v2
						arr_fac_all[i_row, factor_len-1] = v3
					i_row_acc += len(l_pos_fac_2)

		for v1 in range(1, modulo):
			for v2 in range(1, modulo):
				for v3 in range(1, modulo):
					for v4 in range(1, modulo):
						for i_row, (i1, i2, i3) in enumerate(l_pos_fac_3, i_row_acc):
							arr_fac_all[i_row, i1] = v1
							arr_fac_all[i_row, i2] = v2
							arr_fac_all[i_row, i3] = v3
							arr_fac_all[i_row, factor_len-1] = v4
						i_row_acc += len(l_pos_fac_3)

		for arr_idx_split in np.array_split(np.arange(0, arr_fac_all.shape[0]), 100):
			l_args_part = []
			arr_fac_split = arr_fac_all[arr_idx_split]
			for arr_fac_row in arr_fac_split:
				l_args_part.append(tuple(arr_fac_row.tolist()))
			l_arguments.append((modulo, l_args_part))
		
		assert arr_fac_all.shape[0] == i_row_acc
	else:
		arr_fac = np.zeros((amount_factor, factor_len), dtype=np.int64)

		arr_fac[:, -1] = np.random.randint(1, modulo, (arr_fac.shape[0], ))

		arr_idx = np.random.rand(*(arr_fac.shape[0], arr_fac.shape[1]-1)).argsort(1)[:, :additional_values]
		arr_fac[np.repeat(np.arange(0, arr_fac.shape[0]), additional_values), arr_idx.reshape((-1, ))] = np.random.randint(1, modulo, (arr_fac.shape[0]*additional_values, ))

		for i in range(0, arr_fac.shape[0], jump_factor):
			l_args_part = []
			for j in range(0, jump_factor):
				l_args_part.append(tuple(arr_fac[i+j].tolist()))
			l_arguments.append((modulo, l_args_part, factor_len))

	mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count()-1)

	# only for testing the responsivness!
	mult_proc_mng.test_worker_threads_response()

	print('Define new Function!')
	mult_proc_mng.define_new_func('func_proc_func', proc_func)

	print('Do the jobs!!')
	l_ret = mult_proc_mng.do_new_jobs(
		['func_proc_func']*len(l_arguments),
		l_arguments,
	)
	print("len(l_ret): {}".format(len(l_ret)))

	# # testing the responsivness again!
	mult_proc_mng.test_worker_threads_response()
	del mult_proc_mng

	print(f"l_ret[0]:\n{l_ret[0]}")

	return l_ret


def do_table_name_exists(con, table_name):
	cur = con.cursor()
	cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
	return table_name in set([v[0] for v in cur])


if __name__ == '__main__':
	# TODO: query first and find out, what is the max found cycle length per modulo per factor_len
	# TODO: do not include other cycle lenghts except the found max one (best case is modulo**2)
	# TODO: create a cython function for the finding of new sequences
	print("Hello World!")
	d_modulo_to_d_t_cycle_to_s_t_fac = {}
	d_modulo_to_d_factor_len_to_d_t_cycle_to_s_t_fac = {}

	sys.exit(0)

	# param_modulo = 10
	# for param_modulo in [8, 9][::-1]*7:
	# for param_modulo in [9][::-1]:
	# for param_modulo in list(range(11, 21))[::-1]:
	# for param_modulo in list(range(2, 11))[::-1]:
	# for param_modulo in list(range(3, 5))[::-1]:
	# for param_modulo in list(range(5, 6))[::-1]:
	# for param_modulo in list(range(6, 7))[::-1]:
	# for param_modulo in list(range(7, 8))[::-1]:
	# for param_modulo in [11, 13, 17, 19, 23, 29][::-1]:
	# for param_modulo in [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17][::-1]:
	# for param_modulo in [4, 12][::-1]:
	# for param_modulo in [10][::-1]:
	# for param_modulo in [11][::-1]:
	for param_modulo in [5][::-1]:
	# for param_modulo in list(range(8, 9))[::-1]:
	# for param_modulo in list(range(9, 10))[::-1]:
	# for param_modulo in [8, 9, 13, 15, 16][::-1]*3:
	# for param_modulo in list(range(16, 31))[::-1]:
		# for param_factor_len in [4, 9, 16, 25, 36, 49, 64]:
		# for param_factor_len in [36, 49, 64, 81][::-1]:
		# for param_factor_len in [36, 49, 64][::-1]:
		# for param_factor_len in [81][::-1]:
		# for param_factor_len in [64][::-1]:
		# for param_factor_len in [36][::-1]:
		for param_factor_len in [25][::-1]:
		# for param_factor_len in [16][::-1]:
		# for param_factor_len in [9][::-1]:
		# for param_factor_len in [4][::-1]:
		# for param_factor_len in [4, 9, 16, 25, 36, 49][::-1]:
		# for param_factor_len in [9, 16, 25, 36, 49, 64][::-1]:
			# for param_additional_values in [1, 2, 3, 4, 5, 6, 7]:
			# for param_additional_values in [8, 9, 10, 11, 12, 13, 14]:
			# for param_additional_values in range(1, 9):
			# for param_additional_values in range(1, 16):
			# for param_additional_values in [5]:
			# for param_additional_values in range(1, 4):
			for param_additional_values in range(1, param_factor_len):
			# for param_additional_values in range(6, 13):
			# for param_additional_values in range(1, 16):
			# for param_additional_values in range(31, 64):
			# for param_additional_values in range(1, 31):
				print(f'param_modulo: {param_modulo}, param_factor_len: {param_factor_len}, param_additional_values: {param_additional_values}')
				l_ret = main(modulo=param_modulo, factor_len=param_factor_len, amount_factor=1_000_000, jump_factor=50_000, additional_values=param_additional_values)

				d_modulo_to_d_len_cycle_to_amount = {}
				l_modulo_full_cycle = []
				l_modulo_non_full_cycle = []

				for ret_val in l_ret:
					if ret_val is None:
						continue

					modulo = ret_val['modulo']

					if modulo not in d_modulo_to_d_len_cycle_to_amount:
						d_modulo_to_d_len_cycle_to_amount[modulo] = {}

					d_len_cycle_to_amount = d_modulo_to_d_len_cycle_to_amount[modulo]

					d_cycle_len_to_d_t_fac_to_s_l_cycle = ret_val['d_cycle_len_to_d_t_fac_to_s_l_cycle']

					for cycle_len, d_t_fac_to_s_l_cycle in d_cycle_len_to_d_t_fac_to_s_l_cycle.items():
						for t_fac, s_l_cycle in d_t_fac_to_s_l_cycle.items():
							for t_cycle in s_l_cycle:
								len_cycle = len(t_cycle)
								factor_len = len(t_fac)

								if len_cycle not in d_len_cycle_to_amount:
									d_len_cycle_to_amount[len_cycle] = 0

								d_len_cycle_to_amount[len_cycle] += 1

								if modulo not in d_modulo_to_d_t_cycle_to_s_t_fac:
									d_modulo_to_d_t_cycle_to_s_t_fac[modulo] = {}
								d_t_cycle_to_s_t_fac = d_modulo_to_d_t_cycle_to_s_t_fac[modulo]

								if t_cycle not in d_t_cycle_to_s_t_fac:
									d_t_cycle_to_s_t_fac[t_cycle] = set()
								s_t_fac = d_t_cycle_to_s_t_fac[t_cycle]

								if t_fac not in s_t_fac:
									s_t_fac.add(t_fac)


								if modulo not in d_modulo_to_d_factor_len_to_d_t_cycle_to_s_t_fac:
									d_modulo_to_d_factor_len_to_d_t_cycle_to_s_t_fac[modulo] = {}
								d_factor_len_to_d_t_cycle_to_s_t_fac = d_modulo_to_d_factor_len_to_d_t_cycle_to_s_t_fac[modulo]

								if factor_len not in d_factor_len_to_d_t_cycle_to_s_t_fac:
									d_factor_len_to_d_t_cycle_to_s_t_fac[factor_len] = {}
								d_t_cycle_to_s_t_fac = d_factor_len_to_d_t_cycle_to_s_t_fac[factor_len]

								if t_cycle not in d_t_cycle_to_s_t_fac:
									d_t_cycle_to_s_t_fac[t_cycle] = set()
								s_t_fac = d_t_cycle_to_s_t_fac[t_cycle]

								if t_fac not in s_t_fac:
									s_t_fac.add(t_fac)

				for modulo, d_len_cycle_to_amount in sorted(d_modulo_to_d_len_cycle_to_amount.items()):
					if d_len_cycle_to_amount and modulo**2 == sorted(d_len_cycle_to_amount.items())[-1][0]:
						l_modulo_full_cycle.append(modulo)
					else:
						l_modulo_non_full_cycle.append(modulo)

				print(f'l_modulo_full_cycle: {l_modulo_full_cycle}')
				print(f'l_modulo_non_full_cycle: {l_modulo_non_full_cycle}')


	d_modulo_to_d_t_cycle_to_t_fac = {
		modulo: {
			t_cycle: sorted(s_t_fac)[0]
			for t_cycle, s_t_fac in d_t_cycle_to_s_t_fac.items()
		}
		for modulo, d_t_cycle_to_s_t_fac in d_modulo_to_d_t_cycle_to_s_t_fac.items()
	}

	d_modulo_to_d_factor_len_to_d_t_cycle_to_t_fac = {
		modulo: {
			factor_len: {
				t_cycle: sorted(s_t_fac)[0]
				for t_cycle, s_t_fac in d_t_cycle_to_s_t_fac.items()
			}
			for factor_len, d_t_cycle_to_s_t_fac in d_factor_len_to_d_t_cycle_to_s_t_fac.items()
		}
		for modulo, d_factor_len_to_d_t_cycle_to_s_t_fac in d_modulo_to_d_factor_len_to_d_t_cycle_to_s_t_fac.items()
	}

	# for modulo, d_len_cycle_to_amount in sorted(d_modulo_to_d_len_cycle_to_amount.items()):
	# 	print(f'modulo: {modulo}, d_len_cycle_to_amount: {sorted(d_len_cycle_to_amount.items())[::-1][0]}')

	for modulo, d_t_cycle_to_t_fac in d_modulo_to_d_t_cycle_to_t_fac.items():
		u, c = np.unique([len(t_cycle) for t_cycle in d_t_cycle_to_t_fac.keys()], return_counts=True)
		l_t_u_c = sorted([(val_u, val_c) for val_u, val_c in zip(u.tolist(), c.tolist())], reverse=True)
		print(f'modulo: {modulo}, l_t_u_c: {l_t_u_c}')

	### create a sqlite db and write the data into it
	file_path_db_sqlite3 = os.path.join(PATH_ROOT_DIR, 'cyclic_2_factor_03.sqlite3')
	con = sqlite3.connect(file_path_db_sqlite3)
	cur = con.cursor()

	query_create_tbl_cyclic_2_factor_sequence = """
CREATE TABLE IF NOT EXISTS cyclic_2_factor_sequence (
	"id" INTEGER PRIMARY KEY AUTOINCREMENT,
	"modulo" INTEGER NOT NULL,
	"cycle_len" INTEGER NOT NULL,
	"factor_len" INTEGER NOT NULL,
	"t_cycle" TEXT NOT NULL,
	"t_factor" TEXT NOT NULL,
	"amount_nonzero_factors" INTEGER NOT NULL,
	"status" TEXT NOT NULL,
	"dt" TEXT NOT NULL,
	UNIQUE(modulo, factor_len, t_cycle)
);
"""

	cur.execute(query_create_tbl_cyclic_2_factor_sequence)
	con.commit()

	query_select_tbl_cyclic_2_factor_sequence = """
SELECT
	id, modulo, cycle_len, factor_len, t_cycle, t_factor
FROM
	cyclic_2_factor_sequence
;
"""

	query_select_tbl_cyclic_2_factor_sequence_complete = """
SELECT
	id, modulo, cycle_len, factor_len, t_cycle, t_factor, amount_nonzero_factors, status, dt
FROM
	cyclic_2_factor_sequence
;
"""

	res = cur.execute(query_select_tbl_cyclic_2_factor_sequence)
	l_res = list(res)

	d_res = {(modulo, cycle_len, factor_len, eval(t_cycle)): (id_, eval(t_factor)) for id_, modulo, cycle_len, factor_len, t_cycle, t_factor in l_res}

	# d_calc = {
	# 	(modulo, len(t_cycle), len(t_fac), t_cycle): t_fac
	# 	for modulo, d_t_cycle_to_t_fac in d_modulo_to_d_t_cycle_to_t_fac.items()
	# 	for t_cycle, t_fac in d_t_cycle_to_t_fac.items()
	# }
	d_calc = {
		(modulo, len(t_cycle), factor_len, t_cycle): t_fac
		for modulo, d_factor_len_to_d_t_cycle_to_t_fac in d_modulo_to_d_factor_len_to_d_t_cycle_to_t_fac.items()
		for factor_len, d_t_cycle_to_t_fac in d_factor_len_to_d_t_cycle_to_t_fac.items()
		for t_cycle, t_fac in d_t_cycle_to_t_fac.items()
	}

	s_res_key = set(d_res.keys())
	s_calc_key = set(d_calc.keys())

	s_key_found = s_calc_key & s_res_key
	s_key_new = s_calc_key - s_res_key

	print('--------')
	print(f'len(s_res_key): {len(s_res_key)}')
	print(f'len(s_calc_key): {len(s_calc_key)}')
	print('--------')
	print(f'len(s_key_found): {len(s_key_found)}')
	print(f'len(s_key_new): {len(s_key_new)}')

	# find first the not found ones and add them to the db
	# next find the already included ones and find the next smallest t_fac
	# if the new t_fac is smaller than the already one, update the line

	sql_query_insert_template = """
INSERT INTO "cyclic_2_factor_sequence" ("modulo", "cycle_len", "factor_len", "t_cycle", "t_factor", "amount_nonzero_factors", "status", "dt") VALUES
	({{modulo}}, {{cycle_len}}, {{factor_len}}, \"{{t_cycle}}\", \"{{t_factor}}\",
		{{amount_nonzero_factors}}, \"{{status}}\", \"{{dt}}\");
"""

	sql_query_delete_template = """
DELETE FROM "cyclic_2_factor_sequence"
WHERE id = {{id}};
"""

	amount_updated_rows = 0
	for key_found in s_key_found:
		id_, t_factor_res = d_res[key_found]
		t_factor_calc = d_calc[key_found]
		if t_factor_calc < t_factor_res:
			amount_updated_rows += 1

			delete_statement = sql_query_delete_template.replace("{{id}}", str(id_))
			cur.execute(delete_statement)

			modulo = str(key_found[0])
			cycle_len = str(key_found[1])
			factor_len = str(key_found[2])
			t_cycle = str(key_found[3]).replace(' ', '')
			t_factor_orig = d_calc[key_found]
			t_factor = str(t_factor_orig).replace(' ', '')
			amount_nonzero_factors = str(sum([1 if factor != 0 else 0 for factor in t_factor_orig]))
			status = "update"
			dt = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H:%M:%S.%f')
			insert_statement = (
				sql_query_insert_template
				.replace("{{modulo}}", modulo)
				.replace("{{cycle_len}}", cycle_len)
				.replace("{{factor_len}}", factor_len)
				.replace("{{t_cycle}}", t_cycle)
				.replace("{{t_factor}}", t_factor)
				.replace("{{amount_nonzero_factors}}", amount_nonzero_factors)
				.replace("{{status}}", status)
				.replace("{{dt}}", dt)
			)
			cur.execute(insert_statement)
	con.commit()

	print(f'amount_updated_rows: {amount_updated_rows}')

	for key_new in s_key_new:
		modulo = str(key_new[0])
		cycle_len = str(key_new[1])
		factor_len = str(key_new[2])
		t_cycle = str(key_new[3]).replace(' ', '')
		t_factor_orig = d_calc[key_new]
		t_factor = str(t_factor_orig).replace(' ', '')
		amount_nonzero_factors = str(sum([1 if factor != 0 else 0 for factor in t_factor_orig]))
		status = "new"
		dt = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H:%M:%S.%f')
		insert_statement = (
			sql_query_insert_template
			.replace("{{modulo}}", modulo)
			.replace("{{cycle_len}}", cycle_len)
			.replace("{{factor_len}}", factor_len)
			.replace("{{t_cycle}}", t_cycle)
			.replace("{{t_factor}}", t_factor)
			.replace("{{amount_nonzero_factors}}", amount_nonzero_factors)
			.replace("{{status}}", status)
			.replace("{{dt}}", dt)
		)
		cur.execute(insert_statement)
	con.commit()

	res_2 = cur.execute(query_select_tbl_cyclic_2_factor_sequence_complete)
	l_res_2 = list(res_2)

	con.close()
