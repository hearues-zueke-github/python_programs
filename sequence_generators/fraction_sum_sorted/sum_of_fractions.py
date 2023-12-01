#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.11 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import itertools
import math
import os
import pdb
import re
import sys
import time
import traceback

import numpy as np # need installation from pip
import pandas as pd # need installation from pip
import multiprocessing as mp

import matplotlib.pyplot as plt # need installation from pip

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap # need installation from pip
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile # need installation from pip
from recordclass import RecordClass
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

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
load_module_dynamically(**dict(var_glob=var_glob, name='utils_math', path=os.path.join(os.path.join(PYTHON_PROGRAMS_DIR, "utils"), "utils_math.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
gen_primes = utils_math.gen_primes
prime_factorization = utils_math.prime_factorization

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	print("Hello World!")

	n = 5
	l_prime = list(gen_primes(n=n))
	s_prime = set(l_prime)

	d_factors = {1: 8} | {i: sorted(prime_factorization(n=i, l_prime=l_prime).keys())[0] if i not in s_prime else i for i in range(2, n + 1)}

	amount_comb = reduce(lambda a, b: a*b, d_factors.values(), 1)

	arr_1 = np.zeros((amount_comb, n), dtype=np.int64)

	l_num_factor_rev = sorted(d_factors.items())[::-1]
	
	num, factor = l_num_factor_rev[0]
	arr_1[:, num-1] = np.tile(np.arange(0, factor).astype(np.int64), amount_comb // factor)

	factor_mult_prev = factor
	for num, factor in l_num_factor_rev[1:]:
		print(f"num: {num}, factor: {factor}")
		arr_1[:, num - 1] = np.tile(np.repeat(np.arange(0, factor).astype(np.int64), factor_mult_prev), amount_comb // factor // factor_mult_prev)
		factor_mult_prev *= factor

	l_num = list(range(1, n + 1))
	lcm_of_l_num = math.lcm(*l_num)
	l_factor_mult = [lcm_of_l_num // num for num in l_num]

	arr_sum = np.sum(arr_1*l_factor_mult, 1)
