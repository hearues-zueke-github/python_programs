#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.11 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import itertools
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
from math import gcd
from memory_tempfile import MemoryTempfile # need installation from pip
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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	print("Hello World!")

	for v_start in range(1, 100):
		# v_start = 6
		l_v = []
		l_gcd = []
		v = v_start
		l_v.append(v)
		for n in range(2, 1000000):
			if n % 1000000 == 0:
				print(f"n: {n}")
			v_gcd = gcd(n, v)
			v = v + v_gcd

			l_gcd.append(v_gcd)
			l_v.append(v)

		arr_gcd = np.array(l_gcd)

		arr_idx = np.where(arr_gcd > 1)[0]
		arr_gcd_gt_1 = arr_gcd[arr_idx]

		arr_idx_n = arr_idx + 2

		arr_n_gcd = np.vstack((arr_idx_n, arr_gcd_gt_1)).T
		# print(f"arr_n_gcd:\n{arr_n_gcd}")

		print(f"v_start: {v_start}")
		print(f"arr_n_gcd.shape[0]: {arr_n_gcd.shape[0]}")
		if arr_n_gcd.shape[0] > 0:
			max_v_gcd = np.max(arr_n_gcd[:, 1])
			print(f"max_v_gcd: {max_v_gcd}")
		print("")
