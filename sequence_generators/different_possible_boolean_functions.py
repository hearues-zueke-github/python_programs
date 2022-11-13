#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

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
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	print("Hello World!")

	# 2, 3, 4, 5
	# 6, 20, 68, 232


	amount_variables = 7

	arr_comb = get_permutation_table(n=amount_variables-1, is_same_pos=True, is_sorted=False)

	method = "infix"
	# method = "polish"

	d_method_to_templates = {
		"infix": {
			"str_func_left_template": "(f{idx_var_left};(x{idx_var_left}))",
			"str_func_right_template": "(f{idx_var_right};(x{idx_var_right}))",
			"str_func_template": "fn{idx_op_nr};({str_func_left} op{idx_op}; {str_func_right})",
		},
		"polish": {
			"str_func_left_template": "f{idx_var_left};x{idx_var_left}",
			"str_func_right_template": "f{idx_var_right};x{idx_var_right}",
			"str_func_template": "fn{idx_op_nr};op{idx_op};({str_func_left}, {str_func_right})",
		},
	}

	d_temp = d_method_to_templates[method]
	str_func_left_template = d_temp["str_func_left_template"]
	str_func_right_template = d_temp["str_func_right_template"]
	str_func_template = d_temp["str_func_template"]

	s_t_t_range = set()
	s_str_func = set()
	d_t_t_range_to_str_func = {}
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
			idx_var_left = idx_op
			idx_var_right = idx_op + 1
			
			if idx_var_left not in d_idx_var_to_t_range_bracket:
				t_range_bracket_left = (idx_var_left, idx_var_left + 1)
				str_func_left = str_func_left_template.format(
					idx_var_left=idx_var_left,
				)
			else:
				t_range_bracket_left = d_idx_var_to_t_range_bracket[idx_var_left]
				str_func_left = d_idx_var_to_str_func[idx_var_left]

			if idx_var_right not in d_idx_var_to_t_range_bracket:
				t_range_bracket_right = (idx_var_right, idx_var_right + 1)
				str_func_right = str_func_right_template.format(
					idx_var_right=idx_var_right,
				)
			else:
				t_range_bracket_right = d_idx_var_to_t_range_bracket[idx_var_right]
				str_func_right = d_idx_var_to_str_func[idx_var_right]

			t_range_bracket = (t_range_bracket_left[0], t_range_bracket_right[1])
			str_func = str_func_template.format(
				idx_op_nr=idx_op_nr,
				idx_op=idx_op,
				str_func_left=str_func_left,
				str_func_right=str_func_right,
			)

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
			d_t_t_range_to_str_func[t_t_range] = str_func

		# if not str_func in s_str_func:
		# 	s_str_func.add(str_func)

	print(f"s_t_t_range: {s_t_t_range}")
	print(f"len(s_t_t_range): {len(s_t_t_range)}")
	print(f"")
	print(f"d_t_t_range_to_str_func: {d_t_t_range_to_str_func}")
	print(f"len(d_t_t_range_to_str_func): {len(d_t_t_range_to_str_func)}")

	# l_t_t_range_to_str_func = d_t_t_range_to_str_func.items()

	l_x = [f'x{i}' for i in range(0, amount_variables)]
	
	l_op = ['|', '&']
	l_neg_op = ['', '~']
	l_neg_x = ['', '~']

	l_var_op = [f'op{i};' for i in range(0, amount_variables-1)]
	l_var_neg_op = [f'fn{i};' for i in range(0, amount_variables-1)]
	l_var_neg_x = [f'f{i};' for i in range(0, amount_variables)]

	d_t_var_str_func = {}
	for t_t_range, str_func in d_t_t_range_to_str_func.items():
	# for (t_t_range_1, str_func), (t_t_range_2, str_func_2) in itertools.combinations(d_t_t_range_to_str_func.items(), 2):
		print(f"sorted(t_t_range): {sorted(t_t_range)}")
		print(f"- str_func: {str_func}")

		l_d = []
		for t_var_op_op in itertools.product(*[tuple((var_op, op) for op in l_op) for var_op in l_var_op]):
			d_replace_with_1 = dict((re.escape(k), v) for k, v in t_var_op_op)
			pattern_1 = re.compile("|".join(d_replace_with_1.keys()))
			str_func_1 = pattern_1.sub(lambda m: d_replace_with_1[re.escape(m.group(0))], str_func)

			for t_var_neg_op_neg_op in itertools.product(*[tuple((var_neg, neg) for neg in l_neg_op) for var_neg in l_var_neg_op]):
				d_replace_with_2 = dict((re.escape(k), v) for k, v in t_var_neg_op_neg_op)
				pattern_2 = re.compile("|".join(d_replace_with_2.keys()))
				str_func_2 = pattern_2.sub(lambda m: d_replace_with_2[re.escape(m.group(0))], str_func_1)

				for t_var_neg_x_neg_x in itertools.product(*[tuple((var_x, x) for x in l_neg_x) for var_x in l_var_neg_x]):
					d_replace_with_3 = dict((re.escape(k), v) for k, v in t_var_neg_x_neg_x)
					pattern_3 = re.compile("|".join(d_replace_with_3.keys()))
					str_func_3 = pattern_3.sub(lambda m: d_replace_with_3[re.escape(m.group(0))], str_func_2)

					# l_d.append((tuple(sorted(t_var_op_op+t_var_neg_op_neg_op+t_var_neg_x_neg_x)), str_func_3))

		# d_t_str_func.extend(l_d)
					d_t_var_str_func[tuple(sorted(t_var_op_op+t_var_neg_op_neg_op+t_var_neg_x_neg_x))] = str_func_3

	l_s_t_str_func_similar = []

	arr_comb_var = get_permutation_table(n=amount_variables, is_same_pos=True, is_sorted=False)
	arr_all_values = np.zeros((arr_comb_var.shape[0], 2**amount_variables), dtype=np.bool_)
	
	while len(d_t_var_str_func) > 1:
		print(f"xd_t_var_str_func: {len(d_t_var_str_func)}")
		l_rest_t_var = list(d_t_var_str_func.keys())

		t_var_1 = l_rest_t_var[0]
		str_func_1 = d_t_var_str_func[t_var_1]

		d_glob_1 = {}
		d_loc_1 = {}
		arr_comb = get_all_combinations_repeat(m=2, n=amount_variables).T.astype(np.bool_)
		exec(f'def fn():\n\treturn {str_func_1}\n', d_glob_1, d_loc_1)

		for row_idx, comb_var in enumerate(arr_comb_var, 0):
			for i, var_i in enumerate(comb_var, 0):
				d_glob_1[f'x{i}'] = arr_comb[var_i]
			arr_all_values[row_idx] = d_loc_1['fn']()
		
		d_glob_2 = {}
		d_loc_2 = {}
		for i in range(0, amount_variables):
			d_glob_2[f'x{i}'] = arr_comb[i]
		
		s_t_var_to_remove = set([t_var_1])
		for t_var_2 in l_rest_t_var[1:]:
			str_func_2 = d_t_var_str_func[t_var_2]
			exec(f'def fn():\n\treturn {str_func_2}\n', d_glob_2, d_loc_2)
			arr_row = d_loc_2['fn']()

			if np.any(np.all(arr_row == arr_all_values, 1)):
				s_t_var_to_remove.add(t_var_2)

		l_s_t_str_func_similar.append(s_t_var_to_remove)

		for t_var_to_remove in s_t_var_to_remove:
			del d_t_var_str_func[t_var_to_remove]

	amount_unique_str_func = len(l_s_t_str_func_similar)
	print(f"amount_variables: {amount_variables}")
	print(f"amount_unique_str_func: {amount_unique_str_func}")
