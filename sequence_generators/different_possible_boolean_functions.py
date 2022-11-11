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
get_permutation_table = different_combinations.get_permutation_table

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	print("Hello World!")

	amount_variables = 4

	arr_comb = get_permutation_table(n=amount_variables-1, is_same_pos=True, is_sorted=False)

	s_t_t_range = set()
	s_str_func = set()
	for row_idx_op in arr_comb:
		print(f"row_idx_op: {row_idx_op}")

		arr_bracket_left = np.zeros((amount_variables+1, ), dtype=np.int32)
		arr_bracket_right = np.zeros((amount_variables+1, ), dtype=np.int32)

		d_idx_var_to_t_range_bracket = {}
		d_idx_var_to_str_func = {}
		arr_bracket_acc = np.zeros((2, amount_variables+1), dtype=np.int32)
		l_t_range = []
		l_str_func_part = []

		for idx_op_nr, idx_op in enumerate(row_idx_op):
			# print(f"-- idx_op: {idx_op}")

			idx_var_left = idx_op
			idx_var_right = idx_op + 1
			
			if idx_var_left not in d_idx_var_to_t_range_bracket:
				t_range_bracket_left = (idx_var_left, idx_var_left + 1)
				# str_func_left = f"f{idx_var_left}'(x{idx_var_left}')"
				str_func_left = f"f{idx_var_left}(x{idx_var_left})"
			else:
				t_range_bracket_left = d_idx_var_to_t_range_bracket[idx_var_left]
				str_func_left = d_idx_var_to_str_func[idx_var_left]

			if idx_var_right not in d_idx_var_to_t_range_bracket:
				t_range_bracket_right = (idx_var_right, idx_var_right + 1)
				# str_func_right = f"f{idx_var_right}'(x{idx_var_right}')"
				str_func_right = f"f{idx_var_right}(x{idx_var_right})"
			else:
				t_range_bracket_right = d_idx_var_to_t_range_bracket[idx_var_right]
				str_func_right = d_idx_var_to_str_func[idx_var_right]

			# print(f"--- t_range_bracket_left: {t_range_bracket_left}")
			# print(f"--- t_range_bracket_right: {t_range_bracket_right}")

			t_range_bracket = (t_range_bracket_left[0], t_range_bracket_right[1])
			# str_func = f"fn{idx_op_nr}'({str_func_left} op{idx_op}' {str_func_right})"
			# str_func = f"fn{idx_op_nr}'(op{idx_op}'({str_func_left}, {str_func_right}))"
			str_func = f"fn{idx_op_nr}(op{idx_op}({str_func_left}, {str_func_right}))"

			# print(f"---- t_range_bracket: {t_range_bracket}") 

			l_t_range.append(t_range_bracket)
			l_str_func_part.append(str_func)

			next_idx_var_left = t_range_bracket[0]
			next_idx_var_right = t_range_bracket[1] - 1

			d_idx_var_to_t_range_bracket[next_idx_var_left] = t_range_bracket
			d_idx_var_to_t_range_bracket[next_idx_var_right] = t_range_bracket
			
			d_idx_var_to_str_func[next_idx_var_left] = str_func
			d_idx_var_to_str_func[next_idx_var_right] = str_func

			arr_bracket_acc[0, next_idx_var_left] += 1
			arr_bracket_acc[1, next_idx_var_right] += 1

		print(f"- arr_bracket_acc: {arr_bracket_acc.tolist()}")
		print(f"- l_t_range: {l_t_range}")
		print(f"- sorted(l_t_range): {sorted(l_t_range)}")

		t_t_range = tuple(sorted(l_t_range))
		if t_t_range not in s_t_t_range:
			s_t_t_range.add(t_t_range)

		if not str_func in s_str_func:
			s_str_func.add(str_func)

	print(f"s_t_t_range: {s_t_t_range}")
	print(f"len(s_t_t_range): {len(s_t_t_range)}")
	print(f"")
	print(f"s_str_func: {s_str_func}")
	print(f"len(s_str_func): {len(s_str_func)}")

	for str_func in s_str_func:
		

		break
