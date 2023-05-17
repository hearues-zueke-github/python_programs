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
from recordclass import RecordClass
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
	d_modulo_to_l_t_len_cycle_len_l_cycle = {}
	d_modulo_to_s_cycle_all = {}
	for modulo in range(1, 7):
		s_cycle_all = set()

		arr = get_all_combinations_repeat(n=2, m=modulo).astype(np.int64)
		arr_2 = get_all_combinations_repeat(n=2**2, m=modulo).astype(np.int64)

		d_t_to_i = {tuple(l): i for i, l in enumerate(arr.tolist(), 0)}
		d_i_to_t = {i: t for t, i in d_t_to_i.items()}
		
		for d in arr:
			# print(f"d: {d}")
			for M in arr_2.reshape((-1, 2, 2)):
				arr_next = (np.dot(arr, M) + d) % modulo

				l_edges_directed = [(d_t_to_i[tuple(v1)], d_t_to_i[tuple(v2)]) for v1, v2 in zip(arr.tolist(), arr_next.tolist())]

				l_cycles = get_cycles_of_1_directed_graph(l_edges_directed=l_edges_directed)
				s_cycles = set([tuple(l) for l in l_cycles])
				
				s_cycle_all.update(s_cycles)

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
		print(f"l_t_len_cycle_len_l_cycle: {l_t_len_cycle_len_l_cycle}")

		d_modulo_to_l_t_len_cycle_len_l_cycle[modulo] = l_t_len_cycle_len_l_cycle
		d_modulo_to_s_cycle_all[modulo] = s_cycle_all

	# d_t_to_i = {}

	# d = np.array([2, 3], dtype=np.int64)
	# M = np.array([[1, 2, ], [3, 4], ], dtype=np.int64)

	# x = np.array([4, 0], dtype=np.int64)

	# for i, t in enumerate(s_t, 0):
	# 	d_t_to_i[t] = i

	# for t, i in d_t_to_i.items():
	# 	x = (np.dot(M, x) + d) % modulo


	# t = tuple(x.tolist())
	# d_2_t_to_i = {t: 0}

	# i = 1
	# while True:
	# 	x = (np.dot(M.T, x) + d) % modulo
	# 	t = tuple(x.tolist())

	# 	if t in d_2_t_to_i:
	# 		break

	# 	d_2_t_to_i[t] = i
	# 	i += 1

	# t_start = t
	# length_cycle = i - d_2_t_to_i[t_start]
	# print(f"t_start: {t_start}")
	# print(f"length_cycle: {length_cycle}")
	# print(f"d_2_t_to_i: {d_2_t_to_i}")
