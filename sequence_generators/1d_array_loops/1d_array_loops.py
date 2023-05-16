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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

class Cycle(RecordClass):
	t_cycle: tuple
	d_t_to_i: dict[tuple, int]
	len_cycle: int
	l_l_t_cycle: list[tuple]
	l_factors: list[tuple[int]]


def calc_max_cycle(n, modulo):
	max_cycle = modulo**n
	
	# create all possible combinations
	s_t_orig = set(itertools.product(*[list(range(0, modulo)) for _ in range(0, n)]))
	# print(f"n: {n}, modulo: {modulo}, len(s_t): {len(s_t)}")

	l_cycle = []
	for f1_1 in range(0, n):
		for f1_2 in range(1, modulo):
			l_factors = [(f1_1, f1_2)]

			len_l_factors = len(l_factors)

			s_t_rest = set(s_t_orig)
			s_t_in_cycles = set()
			l_l_t_cycle = []
			while s_t_rest:
				t_start = s_t_rest.pop()
				i_factor = 0

				arr = np.zeros((n, ), dtype=np.int64)
				arr[:] = t_start
				
				d_t_to_i = {t_start: 0}
				d_i_to_t = {0: t_start}
				
				# loop, until the cycle is found
				i = 1
				is_finish_ok = False
				while True:
					shift, mult = l_factors[i_factor]
					i_factor += 1
					if i_factor >= len_l_factors:
						i_factor = 0

					arr_shift = (np.roll(arr, shift) * mult) % modulo
					arr = (arr + arr_shift) % modulo

					t = tuple(arr.tolist())

					if t in d_t_to_i:
						is_finish_ok = True
						break

					if t in s_t_in_cycles:
						break

					if t not in s_t_rest:
						break

					s_t_rest.remove(t)

					d_t_to_i[t] = i
					d_i_to_t[i] = t

					i += 1
					assert i <= max_cycle + 1

				if is_finish_ok:
					len_cycle = i - d_t_to_i[t]
					l_t_cycle = []
					for j in range(i - len_cycle, i):
						t = d_i_to_t[j]
						s_t_in_cycles.add(t)
						l_t_cycle.append(t)
					l_l_t_cycle.append(l_t_cycle)

					l_cycle.append(Cycle(
						t_cycle=t,
						d_t_to_i=d_t_to_i,
						len_cycle=len_cycle,
						l_l_t_cycle=l_l_t_cycle,
						l_factors=l_factors,
					))

					print(f"n: {n}, modulo: {modulo}, t_start: {t_start}, f1_1: {f1_1}, f1_2: {f1_2}, len_cycle: {len_cycle}")

	if len(l_cycle) == 0:
		return 1, l_cycle
	
	return max(cycle.len_cycle for cycle in l_cycle), l_cycle


# this is a program for finding the max length of cycles for
# a 1d array with the length n in a modulo
# e.g. the arr is [1, 2, 4, 0] for n=4 and modulo=5
# the factors are f1=2 and f2=3
# f1 is a right rotational shifting, f2 is multiplying the arr and add the vector to the original one
# the next is: arr <- (arr + (arr >> shift) * mult) % modulo
# in this case arr >> shift == [1, 2, 4, 0] >> 2 == [4, 0, 1, 2]
# and mult is: (arr >> shift) * mult == [4, 0, 1, 2] * 3 == [12, 0, 3, 6]
# adding together the two arr is giving [13, 2, 7, 6]
# therefore the next arr would be [3, 2, 2, 1]

if __name__ == '__main__':
	n_max = 5
	modulo_max = 5

	arr_cycle = np.zeros((n_max, modulo_max), dtype=np.int64)

	for n in range(1, n_max + 1):
		for modulo in range(1, modulo_max + 1):
			max_cycle, l_cycle = calc_max_cycle(n=n, modulo=modulo)
			arr_cycle[n-1, modulo-1] = max_cycle

			print(f"n: {n}, modulo: {modulo}, max_cycle: {max_cycle}")
