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
from memory_tempfile import MemoryTempfile # need installation from pip
# from recordclass import RecordClass
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(PATH_ROOT_DIR, '../..')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_graph_theory', path=os.path.join(PYTHON_PROGRAMS_DIR, "graph_theory/utils_graph_theory.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat
get_cycles_of_1_directed_graph = utils_graph_theory.get_cycles_of_1_directed_graph

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	inp_arg = sys.argv[1]
	l_key_val_str = [kv.split('=') for kv in inp_arg.split(',')]
	d_key_to_val = {key: int(val) for key, val in l_key_val_str}
	
	max_modulo = d_key_to_val['max_modulo']
	n = d_key_to_val['n']

	print(f"n: {n}, max_modulo: {max_modulo}")
	
	assert n >= 1
	assert max_modulo >= 1

	d_modulo_to_l_t_len_cycle_len_l_cycle = {}
	d_modulo_to_s_cycle_all = {}
	d_modulo_to_l_t_factors_s_t_cycle_all = {}
	d_modulo_to_arr_n = {}
	d_modulo_to_d_len_cycle_to_s_unique_factors = {}

	# we are calculating the following cycle:
	# given are e.g. modulo = 5 and n = 2
	# were d = 2 and arr = (4, 3), the starting x = (0, 2)
	# now the next calc x = vec((sum(x * arr) * d) % modulo_big)
	# were
	# - * : is the elemntwise multiplication
	# - sum : is the sum of the vector
	# - % : is the modulo operator
	# - vec : is the conversion from number to vector again
	# and modulo_big is the number modulo**n, in this case 5**2 = 25
	# 
	# so for x = (0, 2) the calculated value would be
	# (sum((0, 2) * (4, 3)) + 2) % 25 =
	# (sum((0, 6)) + 2) % 25 =
	# (6 + 2) % 25 =
	# (8) % 25 =
	# 8
	# 
	# now to convert the number back to a vector we simply convert the number into the base mod modulo
	# so 3 in base 5 with 2 places would be (1, 3), were 1*5**1 + 3*5**0 = 8
	# 
	# for the factors d = 2 and arr = (4, 3) in modulo 5 we get the following sequence:
	# (0, 2), (1, 3), (3, 0), (2, 4), (4, 2), (4, 4), (1, 0), (1, 1), (1, 4), (3, 3), (4, 3)
	# the cycle repeats after 11 times

	for modulo in range(1, max_modulo + 1):
		s_cycle_all = set()
		l_t_factors_s_t_cycle_all = []
		d_len_cycle_to_s_unique_factors = {}
		
		# arr_1 = get_all_combinations_repeat(n=1, m=modulo**n).astype(np.int64)
		# arr_n = get_all_combinations_repeat(n=n, m=modulo).astype(np.int64)
		# arr_n_pow_2 = get_all_combinations_repeat(n=n, m=modulo**n).astype(np.int64)

		arr_1 = [i for i in range(0, modulo**n)]
		arr_n = [(i1, i2) for i1 in range(0, modulo) for i2 in range(0, modulo)]
		arr_n_pow_2 = [(i1, i2) for i1 in range(0, modulo**n) for i2 in range(0, modulo**n)]

		modulo_big = modulo**n

		d_t_to_i = {tuple(l): i for i, l in enumerate(arr_n, 0)}
		# d_t_to_i = {tuple(l): i for i, l in enumerate(arr_n.tolist(), 0)}
		d_i_to_t = {i: t for t, i in d_t_to_i.items()}

		for arr in arr_n_pow_2:
			for d in arr_1:
				arr_next_idx = [(sum(v1*v2 for v1, v2 in zip(v, arr)) + d) % modulo_big for v in arr_n]
				# arr_next_idx = (sum(arr_n * arr, axis=1) + d) % modulo_big

				l_edges_directed = [(idx, idx_next) for idx, idx_next in zip(range(0, modulo_big), arr_next_idx)]
				# l_edges_directed = [(idx, idx_next) for idx, idx_next in zip(range(0, modulo_big), arr_next_idx.tolist())]

				l_cycles = get_cycles_of_1_directed_graph(l_edges_directed=l_edges_directed)
				s_cycles = set([tuple(l) for l in l_cycles])
				s_t_cycles = set([tuple(d_i_to_t[i] for i in l) for l in l_cycles])
				
				s_cycle_all.update(s_cycles)
				l_t_factors_s_t_cycle_all.append((d, arr, s_t_cycles))
				# l_t_factors_s_t_cycle_all.append((int(d), tuple(arr.tolist()), s_t_cycles))

		for d, t_arr, s_t_cycles in l_t_factors_s_t_cycle_all:
			t_factors = (d, t_arr)

			for t_cycles in s_t_cycles:
				len_t_cycles = len(t_cycles)
				
				if len_t_cycles not in d_len_cycle_to_s_unique_factors:
					d_len_cycle_to_s_unique_factors[len_t_cycles] = set()

				s = d_len_cycle_to_s_unique_factors[len_t_cycles]
				if t_factors not in s:
					s.add(t_factors)

		d_len_cycle_to_l_cycle = {}
		for cycle in s_cycle_all:
			len_cycle = len(cycle)

			if len_cycle not in d_len_cycle_to_l_cycle:
				d_len_cycle_to_l_cycle[len_cycle] = []

			d_len_cycle_to_l_cycle[len_cycle].append(cycle)

		for l_cycle in d_len_cycle_to_l_cycle.values():
			l_cycle.sort()

		d_len_cycle_to_len_l_cycle = {}
		for len_cycle, l_cycle in d_len_cycle_to_l_cycle.items():
			d_len_cycle_to_len_l_cycle[len_cycle] = len(l_cycle)

		l_t_len_cycle_len_l_cycle = sorted(d_len_cycle_to_len_l_cycle.items())
		
		print(f"modulo: {modulo}")
		print(f"l_t_len_cycle_len_l_cycle: {l_t_len_cycle_len_l_cycle}")

		d_modulo_to_l_t_len_cycle_len_l_cycle[modulo] = l_t_len_cycle_len_l_cycle
		d_modulo_to_s_cycle_all[modulo] = s_cycle_all
		d_modulo_to_l_t_factors_s_t_cycle_all[modulo] = l_t_factors_s_t_cycle_all
		d_modulo_to_arr_n[modulo] = arr_n
		d_modulo_to_d_len_cycle_to_s_unique_factors[modulo] = d_len_cycle_to_s_unique_factors

	l_modulo_max_len_cycle = sorted([(modulo, l[-1][0]) for modulo, l in d_modulo_to_l_t_len_cycle_len_l_cycle.items()])
	l_modulo_max_len_cycle_amount = sorted([(modulo, l[-1][1]) for modulo, l in d_modulo_to_l_t_len_cycle_len_l_cycle.items()])
	l_modulo_max_amount_len_cycle = sorted([(modulo, max(v[1] for v in l)) for modulo, l in d_modulo_to_l_t_len_cycle_len_l_cycle.items()])
	l_modulo_len_len_cycle = sorted([(modulo, len(l)) for modulo, l in d_modulo_to_l_t_len_cycle_len_l_cycle.items()])

	l_modulo_l_len_cycle_amount_unique_factors = sorted([
		(modulo, sorted([
			(len_cycle, len(s_unique_factors)) for len_cycle, s_unique_factors in d.items()
		]))
		for modulo, d
		in d_modulo_to_d_len_cycle_to_s_unique_factors.items()
	])

	l_modulo_amount_max_len_cycle_factors = sorted([(modulo, l[-1][1]) for modulo, l in l_modulo_l_len_cycle_amount_unique_factors])
	l_modulo_amount_max_len_cycle = sorted([(modulo, l[-1][0]) for modulo, l in l_modulo_l_len_cycle_amount_unique_factors])

	l_max_len_cycle = [t[1] for t in l_modulo_max_len_cycle]
	l_max_len_cycle_amount = [t[1] for t in l_modulo_max_len_cycle_amount]
	l_max_amount_len_cycle = [t[1] for t in l_modulo_max_amount_len_cycle]
	l_len_len_cycle = [t[1] for t in l_modulo_len_len_cycle]
	l_amount_max_len_cycle_factors = [t[1] for t in l_modulo_amount_max_len_cycle_factors]
	l_amount_max_len_cycle = [t[1] for t in l_modulo_amount_max_len_cycle]

	print(f"n: {n}")
	print(f"- l_max_len_cycle: {l_max_len_cycle}")
	print(f"- l_max_len_cycle_amount: {l_max_len_cycle_amount}")
	print(f"- l_max_amount_len_cycle: {l_max_amount_len_cycle}")
	print(f"- l_len_len_cycle: {l_len_len_cycle}")
	print(f"- l_amount_max_len_cycle_factors: {l_amount_max_len_cycle_factors}")
	# print(f"- l_amount_max_len_cycle: {l_amount_max_len_cycle}")

	assert l_max_len_cycle == l_amount_max_len_cycle

	# l = [t for t in [t[:2]+([t_cycle for t_cycle in t[2] if len(t_cycle) >= 25], ) for t in d_modulo_to_l_t_factors_s_t_cycle_all[5]] if len(t[2]) > 0]
