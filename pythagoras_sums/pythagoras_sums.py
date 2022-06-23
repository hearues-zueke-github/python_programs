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
from itertools import combinations
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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def get_pythagorians_tuples(n_max: int, p: int, s: int) -> List[Any]:
	print(f"Doing now: n_max: {n_max}, p: {p}, s: {s}")

	d = {i**p: i for i in range(1, n_max * s)}
	l = []
	for t in combinations(list(range(1, n_max + 1)), s):
		n_sum = 0

		for n in t:
			n_sum += n**p

		if n_sum in d:
			l.append(t + (d[n_sum], ))

	return l


def get_pythagorians_tuples_better(n_max: int, p: int, s: int, l_p: List[int]) -> List[Any]:
	print(f"Doing now: n_max: {n_max}, p: {p}, s: {s}")

	d = {i**v_p: (i, v_p) for i in range(1, n_max * s) for v_p in l_p}
	l = []
	for t in combinations(list(range(1, n_max + 1)), s):
		n_sum = 0

		for n in t:
			n_sum += n**p

		if n_sum in d:
			l.append(tuple((v, p) for v in t) + (d[n_sum], ))

	return d, l


if __name__ == '__main__':
	# n_max = 300

	# d, l = get_pythagorians_tuples_better(n_max=n_max, p=3, s=4, l_p=[2, 3, 4, 5, 6, 7])

	# d_t_p = {}
	# for t in l:
	#	 t_s = t[:-1]
	#	 t_p = t[-1]

	#	 if t_p not in d_t_p:
	#		 d_t_p[t_p] = []

	#	 d_t_p[t_p].append(t_s)

	# l_t_p_len_1 = [(k, v) for k, v in d_t_p.items() if len(v) == 1]
	# l_t_p_sorted = sorted(l_t_p_len_1, key=lambda x: (x[0][1], x[0][0])) # sort first power then the base
	# pprint(l_t_p_sorted)
	
	# l_t_p_len = [(k, v) for k, v in d_t_p.items()]
	# d_len_to_l_t_p = {}

	# for t_p, l_t_s in l_t_p_len:
	#	 len_l_t_s = len(l_t_s)
	#	 if len_l_t_s not in d_len_to_l_t_p:
	#		 d_len_to_l_t_p[len_l_t_s] = []

	#	 d_len_to_l_t_p[len_l_t_s].append(t_p)

	# for l in d_len_to_l_t_p.values():
	#	 l.sort(key=lambda x: (x[1], x[0]))

	# pprint(d_len_to_l_t_p)


	# l_p_2_s_2 = get_pythagorians_tuples(n_max=n_max, p=2, s=2)
	# l_p_2_s_3 = get_pythagorians_tuples(n_max=n_max, p=2, s=3)

	# l_p_3_s_3 = get_pythagorians_tuples(n_max=n_max, p=3, s=3)
	# l_p_3_s_4 = get_pythagorians_tuples(n_max=n_max, p=3, s=4)

	# l_p_4_s_4 = get_pythagorians_tuples(n_max=n_max, p=4, s=4)
	# l_p_4_s_5 = get_pythagorians_tuples(n_max=n_max, p=4, s=5)

	# l_p_5_s_5 = get_pythagorians_tuples(n_max=n_max, p=5, s=5)
	# l_p_5_s_6 = get_pythagorians_tuples(n_max=n_max, p=5, s=6)

	# l_p_6_s_6 = get_pythagorians_tuples(n_max=n_max, p=6, s=6)
	# l_p_6_s_7 = get_pythagorians_tuples(n_max=n_max, p=6, s=7)
	# l_p_6_s_8 = get_pythagorians_tuples(n_max=n_max, p=6, s=8)


	# d_p_s = {}
	# for p in range(2, 13):
	#	 d_s = {}
	#	 for s in range(2, 3):
	#		 d_s[s] = get_pythagorians_tuples(n_max=n_max, p=p, s=s)
	#	 d_p_s[p] = d_s


	l = get_pythagorians_tuples(n_max=500, p=2, s=3)
	l.sort()
	# df = pd.DataFrame(data=l, columns=['a1', 'a2', 'b'])

	arr = np.array(l)
	l_unique_t = [arr[0]]
	print(f"arr_row: {arr[0]}")

	for arr_row in arr[1:]:
		is_unique = True
		for arr_row_other in l_unique_t:
			if np.all((arr_row % arr_row_other) == 0):
				is_unique = False

		if is_unique:
			print(f"arr_row: {arr_row}")
			l_unique_t.append(arr_row)

	# df = pd.DataFrame(data=l_unique_t, columns=['a', 'b', 'c'], dtype=object)
	df = pd.DataFrame(data=l_unique_t, columns=['a', 'b', 'c', 'd'], dtype=object)


	d_val_d_to_l_t = {}

	for _, row in df.iterrows():
		arr = row.values
		val_d = arr[-1]
		t = tuple(arr[:-1].tolist())

		if val_d not in d_val_d_to_l_t:
			d_val_d_to_l_t[val_d] = []

		d_val_d_to_l_t[val_d].append(t)

	for l in d_val_d_to_l_t.values():
		l.sort()

	l_val_d = sorted(d_val_d_to_l_t.keys())

	l_val_d_len_l_t = []
	amount_t_next_min = 0
	for val_d in l_val_d:
		l_t = d_val_d_to_l_t[val_d]
		len_l_t = len(l_t)

		if amount_t_next_min < len_l_t:
			l_val_d_len_l_t.append((val_d, len_l_t))
			amount_t_next_min = len_l_t
