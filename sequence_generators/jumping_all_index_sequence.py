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


if __name__ == '__main__':
	d_num_to_next = {}
	d_num_to_prev = {}

	n_max = 40000
	for i1, i2 in zip(range(1, n_max), range(2, n_max + 1)):
		d_num_to_next[i1] = i2
		d_num_to_prev[i2] = i1

	start_num = 3
	current_num = start_num
	l_sequence = [start_num]

	index = 1

	for i_iter in range(0, n_max):
		print(f'i_iter: {i_iter}, current_num: {current_num}')
		# first check, if going to the left side/ going back n steps
		# or in other terms if decreasing order is apply able n times

		temp_num = current_num
		is_found_decreasing = True
		for i in range(0, index):
			if not temp_num in d_num_to_prev:
				is_found_decreasing = False
				break

			temp_num = d_num_to_prev[temp_num]

		if is_found_decreasing:
			# remove the current_num
			if current_num in d_num_to_prev and current_num in d_num_to_next:
				prev_num = d_num_to_prev[current_num]
				next_num = d_num_to_next[current_num]

				d_num_to_prev[next_num] = prev_num
				d_num_to_next[prev_num] = next_num

				del d_num_to_prev[current_num]
				del d_num_to_next[current_num]
			elif not current_num in d_num_to_prev and current_num in d_num_to_next:
				next_num = d_num_to_next[current_num]
				del d_num_to_prev[next_num]
				del d_num_to_next[current_num]
			elif current_num in d_num_to_prev and not current_num in d_num_to_next:
				prev_num = d_num_to_prev[current_num]
				del d_num_to_next[prev_num]
				del d_num_to_prev[current_num]
			else:
				assert False

			current_num = temp_num
			l_sequence.append(current_num)
			index += 1

			continue


		temp_num = current_num
		is_found_increasing = True
		for i in range(0, index):
			if not temp_num in d_num_to_next:
				is_found_increasing = False
				break

			temp_num = d_num_to_next[temp_num]

		if is_found_increasing:
			# remove the current_num
			if current_num in d_num_to_prev and current_num in d_num_to_next:
				prev_num = d_num_to_prev[current_num]
				next_num = d_num_to_next[current_num]

				d_num_to_prev[next_num] = prev_num
				d_num_to_next[prev_num] = next_num

				del d_num_to_prev[current_num]
				del d_num_to_next[current_num]
			elif not current_num in d_num_to_prev and current_num in d_num_to_next:
				next_num = d_num_to_next[current_num]
				del d_num_to_prev[next_num]
				del d_num_to_next[current_num]
			elif current_num in d_num_to_prev and not current_num in d_num_to_next:
				prev_num = d_num_to_prev[current_num]
				del d_num_to_next[prev_num]
				del d_num_to_prev[current_num]
			else:
				assert False

			current_num = temp_num
			l_sequence.append(current_num)
			index += 1

			continue

		break

	print(f'l_sequence: {l_sequence}')

	# find the next smallest index, where all numbers from 1 to k are included
	arr_arg_sort = np.argsort(l_sequence)

	l_index_with_k_numbers = []
	first_number_index = np.where(arr_arg_sort == 0)[0][0]
	
	# check, how many numbers k are included from index 0 to inclusive first_number_index
	arr_part = arr_arg_sort[:first_number_index+1]

	d_modulo_amount = {
		2: np.unique(np.array(l_sequence) % 2, return_counts=True),
		3: np.unique(np.array(l_sequence) % 3, return_counts=True),
		4: np.unique(np.array(l_sequence) % 4, return_counts=True),
		5: np.unique(np.array(l_sequence) % 5, return_counts=True),
		6: np.unique(np.array(l_sequence) % 6, return_counts=True),
		7: np.unique(np.array(l_sequence) % 7, return_counts=True),
	}


