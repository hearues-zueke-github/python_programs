#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.13 -i

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

def recursive_sum_number_without_one(s_t_found, l_stack, n_max):
	val_first = l_stack[0]
	val_second = l_stack[1]
	for n_next in range(n_max, 1, -1):
		diff_first = val_first - n_next
		if diff_first >= val_second:
			l_stack_new = [diff_first] + l_stack[1:] + [n_next]
			s_t_found.add(tuple(l_stack_new))
			recursive_sum_number_without_one(s_t_found=s_t_found, l_stack=l_stack_new, n_max=n_next)

if __name__ == '__main__':
	l_amount = []
	l_amount_true = []
	for n in range(2, 35):
		print(f'n: {n}')

		s_t_found = set()
		s_t_found.add((n, ))
		n_max = n // 2
		for i in range(n_max, 1, -1):
			l_stack = [n-i, i]
			s_t_found.add(tuple(l_stack))
			recursive_sum_number_without_one(s_t_found=s_t_found, l_stack=l_stack, n_max=n_max)

		sorted_s_t_found = sorted(s_t_found)
		print(f'sorted_s_t_found: {sorted_s_t_found}')
		amount_sums = len(s_t_found)
		print(f'amount_sums: {amount_sums}')
		l_amount.append(amount_sums)

		sorted_s_t_found_true = sorted(set([tuple(sorted(t)) for t in s_t_found]))
		amount_sums_true = len(sorted_s_t_found_true)
		print(f'amount_sums_true: {amount_sums_true}')
		l_amount_true.append(amount_sums_true)

	print(f'l_amount: {l_amount}')
	print(f'l_amount_true: {l_amount_true}')
