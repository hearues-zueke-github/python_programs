#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import time
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
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	m = 6

	for n_1 in range(1, 11):
		n_0 = 2
		# n_1 = 3
		n_mult = n_0*n_1
		# arr_comb = get_all_combinations_repeat(m=m, n=n_mult)
		arr_comb = np.random.randint(0, m, (50000, n_mult))

		arr_comb = arr_comb[arr_comb[:, 0] != 0]

		arr_comb_part = arr_comb[:]
		# arr_comb_part = arr_comb[:30000]

		only_full_cycles = True
		stop_by_finding_first_full_cycle = False

		len_cyclic = m**2+1

		time_1_1 = time.time()

		arr_factors = np.zeros((n_0, n_1), dtype=np.int64)
		arr_const_n_0 = np.arange(0, n_0).reshape((n_0, 1))
		arr_const_n_1 = np.arange(0, n_1).reshape((1, n_1))

		time_1_2 = time.time()

		d_cyclic_pow_2_all = {}
		for row in arr_comb_part:
			# print(f"row: {row}")
			arr_factors[:] = row.reshape((n_0, n_1))
						
			l_x = [0, 0]
			s_x = set([(0, 0)])
			x_i0 = 0
			x_i1 = 0

			for _ in range(0, m**2):
				x_i2 = np.sum((arr_factors * (x_i0**arr_const_n_0)) * (x_i1**arr_const_n_1)) % m
				
				t = (x_i1, x_i2)
				if t in s_x:
					break

				x_i0 = x_i1
				x_i1 = x_i2
				l_x.append(x_i1)
				s_x.add((x_i0, x_i1))

			x_i2 = np.sum((arr_factors * (x_i0**arr_const_n_0)) * (x_i1**arr_const_n_1)) % m
			x_i0 = x_i1
			x_i1 = x_i2

			if l_x[0] == x_i0 and l_x[1] == x_i1:
				len_l_x = len(l_x)-1
				is_full_cycle = len_l_x == m**2
				
				if only_full_cycles and not is_full_cycle:
					continue

				if len_l_x not in d_cyclic_pow_2_all:
					d_cyclic_pow_2_all[len_l_x] = {}
				d_cyclic_pow_2_all[len_l_x][tuple(row.tolist())] = tuple(l_x[:-1])

				if is_full_cycle and stop_by_finding_first_full_cycle:
					break
		
		time_1_3 = time.time()


		time_2_1 = time.time()

		arr_factors = np.zeros((n_0, n_1), dtype=np.int64)
		arr_const_n_0 = np.arange(0, n_0).reshape((n_0, 1))
		arr_const_n_1 = np.arange(0, n_1).reshape((1, n_1))

		d_0 = {i: (i**arr_const_n_0) % m for i in range(0, m)}
		d_1 = {i: (i**arr_const_n_1) % m for i in range(0, m)}
		
		time_2_2 = time.time()

		d_cyclic_pow_2_all_faster = {}
		for row in arr_comb_part:
			# print(f"row: {row}")
			arr_factors[:] = row.reshape((n_0, n_1))
						
			l_x = [0, 0]
			s_x = set([(0, 0)])
			x_i0 = 0
			x_i1 = 0

			for _ in range(0, m**2):
				x_i2 = np.sum((arr_factors * d_0[x_i0]) * (d_1[x_i1])) % m
				
				t = (x_i1, x_i2)
				if t in s_x:
					break

				x_i0 = x_i1
				x_i1 = x_i2
				l_x.append(x_i1)
				s_x.add((x_i0, x_i1))

			x_i2 = np.sum((arr_factors * (x_i0**arr_const_n_0)) * (x_i1**arr_const_n_1)) % m
			x_i0 = x_i1
			x_i1 = x_i2

			if l_x[0] == x_i0 and l_x[1] == x_i1:
				len_l_x = len(l_x)-1
				is_full_cycle = len_l_x == m**2
				
				if only_full_cycles and not is_full_cycle:
					continue

				if len_l_x not in d_cyclic_pow_2_all_faster:
					d_cyclic_pow_2_all_faster[len_l_x] = {}
				d_cyclic_pow_2_all_faster[len_l_x][tuple(row.tolist())] = tuple(l_x[:-1])

				if is_full_cycle and stop_by_finding_first_full_cycle:
					break

		time_2_3 = time.time()


		time_3_1 = time.time()

		arr_factors = np.zeros((n_0, n_1), dtype=np.int64)
		arr_const_n_0 = np.arange(0, n_0).reshape((n_0, 1))
		arr_const_n_1 = np.arange(0, n_1).reshape((1, n_1))

		d_0 = {i: (i**arr_const_n_0) % m for i in range(0, m)}
		d_1 = {i: (i**arr_const_n_1) % m for i in range(0, m)}
		d_0_1 = {(k_0, k_1): v_0.dot(v_1) % m for k_0, v_0 in d_0.items() for k_1, v_1 in d_1.items()}

		time_3_2 = time.time()

		d_cyclic_pow_2_all_faster2 = {}
		for row in arr_comb_part:
			# print(f"row: {row}")
			arr_factors[:] = row.reshape((n_0, n_1))
						
			l_x = [0, 0]
			s_x = set([(0, 0)])
			x_i0 = 0
			x_i1 = 0

			for _ in range(0, m**2):
				x_i2 = np.sum(arr_factors * d_0_1[(x_i0, x_i1)]) % m
				
				t = (x_i1, x_i2)
				if t in s_x:
					break

				x_i0 = x_i1
				x_i1 = x_i2
				l_x.append(x_i1)
				s_x.add((x_i0, x_i1))

			x_i2 = np.sum(arr_factors * d_0_1[(x_i0, x_i1)]) % m
			x_i0 = x_i1
			x_i1 = x_i2

			if l_x[0] == x_i0 and l_x[1] == x_i1:
				len_l_x = len(l_x)-1
				is_full_cycle = len_l_x == m**2
				
				if only_full_cycles and not is_full_cycle:
					continue

				if len_l_x not in d_cyclic_pow_2_all_faster2:
					d_cyclic_pow_2_all_faster2[len_l_x] = {}
				d_cyclic_pow_2_all_faster2[len_l_x][tuple(row.tolist())] = tuple(l_x[:-1])

				if is_full_cycle and stop_by_finding_first_full_cycle:
					break

		time_3_3 = time.time()


		time_4_1 = time.time()

		arr_factors = np.zeros((n_0*n_1, ), dtype=np.int64)
		arr_const_n_0 = np.arange(0, n_0).reshape((n_0, 1))
		arr_const_n_1 = np.arange(0, n_1).reshape((1, n_1))

		d_0 = {i: (i**arr_const_n_0) % m for i in range(0, m)}
		d_1 = {i: (i**arr_const_n_1) % m for i in range(0, m)}
		arr_0_1 = np.array([d_0[k_0].dot(d_1[k_1]).flatten() % m for k_0 in range(0, m) for k_1 in range(0, m)], dtype=np.int64)

		time_4_2 = time.time()

		d_cyclic_pow_2_all_faster3 = {}
		for row in arr_comb_part:
			arr_factors[:] = row
						
			l_x = [0, 0]
			s_x = set([(0, 0)])
			x_i0 = 0
			x_i1 = 0

			for _ in range(0, m**2):
				x_i2 = np.sum(arr_factors * arr_0_1[x_i0*m+x_i1]) % m
				
				t = (x_i1, x_i2)
				if t in s_x:
					break

				x_i0 = x_i1
				x_i1 = x_i2
				l_x.append(x_i1)
				s_x.add((x_i0, x_i1))

			x_i2 = np.sum(arr_factors * arr_0_1[x_i0*m+x_i1]) % m
			x_i0 = x_i1
			x_i1 = x_i2

			if l_x[0] == x_i0 and l_x[1] == x_i1:
				len_l_x = len(l_x)-1
				is_full_cycle = len_l_x == m**2
				
				if only_full_cycles and not is_full_cycle:
					continue

				if len_l_x not in d_cyclic_pow_2_all_faster3:
					d_cyclic_pow_2_all_faster3[len_l_x] = {}
				d_cyclic_pow_2_all_faster3[len_l_x][tuple(row.tolist())] = tuple(l_x[:-1])

				if is_full_cycle and stop_by_finding_first_full_cycle:
					break

		time_4_3 = time.time()


		assert d_cyclic_pow_2_all == d_cyclic_pow_2_all_faster
		assert d_cyclic_pow_2_all == d_cyclic_pow_2_all_faster2
		assert d_cyclic_pow_2_all == d_cyclic_pow_2_all_faster3

		time_diff_1_creation = time_1_2 - time_1_1
		time_diff_2_creation = time_2_2 - time_2_1
		time_diff_3_creation = time_3_2 - time_3_1
		time_diff_4_creation = time_4_2 - time_4_1

		time_diff_1_calculation = time_1_3 - time_1_2
		time_diff_2_calculation = time_2_3 - time_2_2
		time_diff_3_calculation = time_3_3 - time_3_2
		time_diff_4_calculation = time_4_3 - time_4_2

		print(f"m: {m}, n_0: {n_0}, n_1: {n_1}")
		print(f"- time_diff_1_creation: {time_diff_1_creation:.5f}s")
		print(f"- time_diff_2_creation: {time_diff_2_creation:.5f}s")
		print(f"- time_diff_3_creation: {time_diff_3_creation:.5f}s")
		print(f"- time_diff_4_creation: {time_diff_4_creation:.5f}s")
		print(f"- time_diff_1_calculation: {time_diff_1_calculation:.5f}s")
		print(f"- time_diff_2_calculation: {time_diff_2_calculation:.5f}s")
		print(f"- time_diff_3_calculation: {time_diff_3_calculation:.5f}s")
		print(f"- time_diff_4_calculation: {time_diff_4_calculation:.5f}s")
