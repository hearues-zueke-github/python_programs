#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.13 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import decimal
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
from memory_tempfile import MemoryTempfile # need installation from pip
# from recordclass import RecordClass # need installation from pip
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from tqdm import tqdm # need installation from pip
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
	# arr_amount_volume = np.zeros((2, 2), dtype=np.float64)

	decimal.getcontext().prec = 100

	v_aa = decimal.Decimal('1')
	v_ab = decimal.Decimal('0')

	v_ba = decimal.Decimal('0')
	v_bb = decimal.Decimal('1')

	v_cup = decimal.Decimal('0.1')

	l_v_aa = []
	l_v_ab = []
	l_v_ba = []
	l_v_bb = []
	l_v_a_sum = []
	l_v_b_sum = []

	l_v_aa.append(v_aa)
	l_v_ab.append(v_ab)
	l_v_ba.append(v_ba)
	l_v_bb.append(v_bb)
	l_v_a_sum.append(v_aa + v_ab)
	l_v_b_sum.append(v_ba + v_bb)

	print("i: 0")
	print(f"- v_aa: {v_aa:13.11f}, v_ab: {v_ab:13.11f}")
	print(f"- v_ba: {v_ba:13.11f}, v_bb: {v_bb:13.11f}")

	for i in range(1, 100 + 1):
		print("")

		# mixing cup A to cup B
		diff_v_aa = v_cup * v_aa / (v_aa + v_ab)
		diff_v_ab = v_cup * v_ab / (v_aa + v_ab)
		
		v_ba += diff_v_aa
		v_aa -= diff_v_aa
		
		v_bb += diff_v_ab
		v_ab -= diff_v_ab

		l_v_aa.append(v_aa)
		l_v_ab.append(v_ab)
		l_v_ba.append(v_ba)
		l_v_bb.append(v_bb)
		l_v_a_sum.append(v_aa + v_ab)
		l_v_b_sum.append(v_ba + v_bb)

		print(f"i: {i}, mixing A -> B")
		print(f"- v_aa: {v_aa:13.11f}, v_ab: {v_ab:13.11f}")
		print(f"- v_ba: {v_ba:13.11f}, v_bb: {v_bb:13.11f}")

		# mixing cup B to cup A
		diff_v_ba = v_cup * v_ba / (v_ba + v_bb)
		diff_v_bb = v_cup * v_bb / (v_ba + v_bb)
		
		v_aa += diff_v_ba
		v_ba -= diff_v_ba

		v_ab += diff_v_bb
		v_bb -= diff_v_bb

		l_v_aa.append(v_aa)
		l_v_ab.append(v_ab)
		l_v_ba.append(v_ba)
		l_v_bb.append(v_bb)
		l_v_a_sum.append(v_aa + v_ab)
		l_v_b_sum.append(v_ba + v_bb)

		print(f"i: {i}, mixing B -> A")
		print(f"- v_aa: {v_aa:13.11f}, v_ab: {v_ab:13.11f}")
		print(f"- v_ba: {v_ba:13.11f}, v_bb: {v_bb:13.11f}")

	arr_v_aa = np.array(l_v_aa)
	arr_v_ab = np.array(l_v_ab)
	arr_v_ba = np.array(l_v_ba)
	arr_v_bb = np.array(l_v_bb)
	arr_v_a_sum = np.array(l_v_a_sum)
	arr_v_b_sum = np.array(l_v_b_sum)

	arr_v = np.vstack((arr_v_aa, arr_v_ab, arr_v_ba, arr_v_bb)).T
