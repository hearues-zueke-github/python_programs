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
from recordclass import RecordClass # need installation from pip
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from tqdm import tqdm
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


def calc_quaternion_mult(l_a, l_b):
	return [
		l_a[0]*l_b[0] - l_a[1]*l_b[1] - l_a[2]*l_b[2] - l_a[3]*l_b[3],
		l_a[0]*l_b[1] + l_a[1]*l_b[0] + l_a[2]*l_b[3] - l_a[3]*l_b[2],
		l_a[2]*l_b[0] + l_a[3]*l_b[1] + l_a[0]*l_b[2] - l_a[1]*l_b[3],
		l_a[0]*l_b[3] + l_a[1]*l_b[2] + l_a[3]*l_b[0] - l_a[2]*l_b[1],
	]


def test_calc_quaternion_mult():
	l_t_l_a_l_b_l_c_ref = [
		# TODO: add more tests later!
		([1, 2, 3, 4], [5, 6, 7, 8], [-60, 12, 30, 24]),
		([3, 5, -2, 7], [4, 6, -1, -9], [43, 63, 76, 8]),
	]

	correct_tests = 0
	for l_a, l_b, l_c_ref in l_t_l_a_l_b_l_c_ref:
		l_c = calc_quaternion_mult(l_a=l_a, l_b=l_b)
		if l_c == l_c_ref:
			correct_tests += 1

	amount_tests = len(l_t_l_a_l_b_l_c_ref)
	print(f'correct_tests: {correct_tests}/{amount_tests}')

	assert correct_tests == amount_tests


if __name__ == '__main__':
	test_calc_quaternion_mult()

	print('Hello World!')

	n_min = 0
	n_max = 6

	l_v = []
	for a1 in range(n_min, n_max+1):
		for a2 in range(n_min, n_max+1):
			for a3 in range(n_min, n_max+1):
				for a4 in range(n_min, n_max+1):
					l_v.append((a1, a2, a3, a4))

	# l_t = []
	# for t_a in l_v:
	#	for t_b in l_v:
	#		l_c = calc_quaternion_mult(l_a=t_a, l_b=t_b)
	#		if l_c[0] != 0 and l_c[1] == 0 and l_c[2] == 0 and l_c[3] == 0:
	#			t_c = tuple(l_c)
	#			print(f't_a: {t_a}, t_b: {t_b}, t_c: {t_c}')
	#			l_t.append((t_a, t_b, t_c))

	print('Hello World!')

	l_t_num_ignore = []
	d_t_num_to_l_t_pair = {}
	for t_a in l_v:
		print(f't_a: {t_a}')
		for t_b in l_v:
			l_c = calc_quaternion_mult(l_a=t_a, l_b=t_b)
			t_c = tuple(l_c)
			if any(c < 0 for c in t_c):
				l_t_num_ignore.append(t_c)
				continue
			if t_c not in d_t_num_to_l_t_pair:
				d_t_num_to_l_t_pair[t_c] = []
			d_t_num_to_l_t_pair[t_c].append((t_a, t_b))

	# l_t_num_to_l_t_pair = sorted(d_t_num_to_l_t_pair.items())

	d_amount_pair_to_l_t_num = {}
	for t_num, l_t_pair in d_t_num_to_l_t_pair.items():
		print(f't_num: {t_num}')
		len_l_t_pair = len(l_t_pair)
		if len_l_t_pair not in d_amount_pair_to_l_t_num:
			d_amount_pair_to_l_t_num[len_l_t_pair] = []
		d_amount_pair_to_l_t_num[len_l_t_pair].append(t_num)
