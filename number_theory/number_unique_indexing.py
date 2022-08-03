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

get_all_combinations_repeat = different_combinations.get_all_combinations_repeat
get_unique_combinations_increment = different_combinations.get_unique_combinations_increment

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

# TODO: implement this function for better performance
def generate_fix_amount_sum(l_tpl, arr, current_idx):
	pass


def generate_fix_amount_sum_bad_impl(m, n, row_sum):
	arr = get_all_combinations_repeat(m=m, n=n).astype('i8')
	arr_part = arr[np.sum(arr, axis=1) == row_sum]
	arr_part_sort = np.sort(arr_part, axis=1)[:, ::-1]
	arr_rec = np.core.records.fromarrays(arr_part_sort.T, dtype=[(f'v_{i}', 'i8') for i in range(0, n)])
	arr_rec_unique = np.unique(arr_rec)

	return arr_rec_unique


def get_arr_unique_vals(m_max, n):
	d_m_to_arr_rec_unique = {}
	l_arr_rec_unique = []

	for m in range(0, min(3, m_max+1)):
		arr_rec_unique = generate_fix_amount_sum_bad_impl(m=2, n=n, row_sum=m)
		l_arr_rec_unique.append(arr_rec_unique[::-1])
		d_m_to_arr_rec_unique[m] = np.array(arr_rec_unique.view('i8').reshape((-1, n)), dtype=np.int64)

	for m in range(3, m_max+1):
		# print(f"m: {m}")
		
		arr_rec_unique = generate_fix_amount_sum_bad_impl(m=m, n=n, row_sum=m)
		l_arr_rec_unique.append(arr_rec_unique[::-1])
		d_m_to_arr_rec_unique[m] = np.array(arr_rec_unique.view('i8').reshape((-1, n)), dtype=np.int64)

	l_key_sort = sorted(d_m_to_arr_rec_unique.keys())
	l_val_sort = [np.sum(np.all(d_m_to_arr_rec_unique[m] != 0, axis=1)) if d_m_to_arr_rec_unique[m].shape[0] > 0 else 0 for m in l_key_sort]

	# print(f"l_key_sort: {l_key_sort}")
	# print(f"l_val_sort: {l_val_sort}")

	arr_unique_vals = np.hstack(l_arr_rec_unique).view((np.int64, n))

	return arr_unique_vals


if __name__ == '__main__':
	m_max = 40

	# check out, if the calculation and the reference values are the same
	for n in range(2, 5): # n == 1 is not working
		print(f'n: {n}')

		arr_unique_vals = get_arr_unique_vals(m_max=m_max, n=n)

		arr_acc_sum = np.empty((arr_unique_vals.shape[0], ), dtype=np.int64)
		arr_acc_mult = np.empty((arr_unique_vals.shape[0], ), dtype=np.int64)

		df = pd.DataFrame(data=arr_unique_vals, columns=[f'v_{i}' for i in range(0, n)], dtype=object)

		for n1 in range(1, n+1):
			arr_comb_incr = get_unique_combinations_increment(m=n, n=n1)

			arr_acc_sum[:] = 0
			for arr_idx in arr_comb_incr:
				arr_acc_mult[:] = 1
				for idx in arr_idx:
					arr_acc_mult *= arr_unique_vals[:, idx]
				arr_acc_sum += arr_acc_mult
			df[f'sum_{n1}'] = arr_acc_sum

		df_ref = pd.DataFrame(data=arr_unique_vals, columns=[f'v_{i}' for i in range(0, n)], dtype=object)
		if n == 1:
			df_ref['sum_1'] = df_ref['v_0']
		if n == 2:
			df_ref['sum_1'] = df_ref['v_0']+df_ref['v_1']
			df_ref['sum_2'] = df_ref['v_0']*df_ref['v_1']
		elif n == 3:
			df_ref['sum_1'] = df_ref['v_0']+df_ref['v_1']+df_ref['v_2']
			df_ref['sum_2'] = (
				df_ref['v_0']*df_ref['v_1']+
				df_ref['v_0']*df_ref['v_2']+
				df_ref['v_1']*df_ref['v_2']
			)
			df_ref['sum_3'] = df_ref['v_0']*df_ref['v_1']*df_ref['v_2']
		elif n == 4:
			df_ref['sum_1'] = df_ref['v_0']+df_ref['v_1']+df_ref['v_2']+df_ref['v_3']
			df_ref['sum_2'] = (
				df_ref['v_0']*df_ref['v_1']+
				df_ref['v_0']*df_ref['v_2']+
				df_ref['v_0']*df_ref['v_3']+
				df_ref['v_1']*df_ref['v_2']+
				df_ref['v_1']*df_ref['v_3']+
				df_ref['v_2']*df_ref['v_3']
			)
			df_ref['sum_3'] = (
				df_ref['v_0']*df_ref['v_1']*df_ref['v_2']+
				df_ref['v_0']*df_ref['v_1']*df_ref['v_3']+
				df_ref['v_0']*df_ref['v_2']*df_ref['v_3']+
				df_ref['v_1']*df_ref['v_2']*df_ref['v_3']
			)
			df_ref['sum_4'] = df_ref['v_0']*df_ref['v_1']*df_ref['v_2']*df_ref['v_3']

		l_columns = [f'v_{i}' for i in range(0, n)]+[f'sum_{i}' for i in range(1, n+1)]
		try:
			assert np.all(df[l_columns]==df_ref[l_columns])
		except:
			print(f"Values are not calculateed correctly for n: {n}!")
			assert False
