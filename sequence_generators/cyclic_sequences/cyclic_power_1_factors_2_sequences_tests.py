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
import sqlite3
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

# own import
# execute before: python3.13 setup_cyclic_multi_factor_pow_seq.py build_ext --inplace
import cyclic_multi_factor_pow_seq_cython
import cyclic_multi_factor_pow_seq_python

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


def simple_function_equal_test_modulo_5():
	modulo = 5
	factors = np.array([0, 2, 0, 0, 0, 0, 0, 4, 4], dtype=np.int64)
	values_c = np.zeros((modulo*modulo*2, ), dtype=np.int64)
	values_p = np.zeros((modulo*modulo*2, ), dtype=np.int64)

	ret_val_c = cyclic_multi_factor_pow_seq_cython.calc_2_value_sequence_pow_2(modulo, factors, values_c)
	ret_val_p = cyclic_multi_factor_pow_seq_python.calc_2_value_sequence_pow_2(modulo, factors, values_p)

	assert np.all(values_c == values_p)


def many_random_pow_2_cycle_modulo_5_tests():
	modulo = 7
	amount_factors = 4000
	all_factors = np.random.randint(0, modulo, (amount_factors, 9), dtype=np.int64)

	all_values_c_single = np.zeros((amount_factors, modulo*modulo*2), dtype=np.int64)
	all_values_p_single = np.zeros((amount_factors, modulo*modulo*2), dtype=np.int64)

	all_values_c_multi = np.zeros((amount_factors * modulo*modulo*2), dtype=np.int64)
	all_values_p_multi = np.zeros((amount_factors * modulo*modulo*2), dtype=np.int64)
	all_found_cycles_c_multi = np.zeros((amount_factors, ), dtype=np.int64)
	all_found_cycles_p_multi = np.zeros((amount_factors, ), dtype=np.int64)

	print('Testing "cyclic_multi_factor_pow_seq_cython.calc_2_value_sequence_pow_2"')
	dt_start_c_single = datetime.datetime.now()
	count_ret_val_c = 0
	for factors, values_c in zip(all_factors, all_values_c_single):
		ret_val_c = cyclic_multi_factor_pow_seq_cython.calc_2_value_sequence_pow_2(modulo, factors, values_c)
		count_ret_val_c += ret_val_c
	dt_end_c_single = datetime.datetime.now()

	print('Testing "cyclic_multi_factor_pow_seq_python.calc_2_value_sequence_pow_2"')
	dt_start_p_single = datetime.datetime.now()
	count_ret_val_p = 0
	for factors, values_p in zip(all_factors, all_values_p_single):
		ret_val_p = cyclic_multi_factor_pow_seq_python.calc_2_value_sequence_pow_2(modulo, factors, values_p)
		count_ret_val_p += ret_val_p
	dt_end_p_single = datetime.datetime.now()

	print('Testing "cyclic_multi_factor_pow_seq_cython.calc_2_value_sequence_pow_2_many"')
	dt_start_c_multi = datetime.datetime.now()
	cyclic_multi_factor_pow_seq_cython.calc_2_value_sequence_pow_2_many(modulo, amount_factors, all_factors.reshape((-1, )), all_values_c_multi, all_found_cycles_c_multi)
	dt_end_c_multi = datetime.datetime.now()

	print('Testing "cyclic_multi_factor_pow_seq_python.calc_2_value_sequence_pow_2_many"')
	dt_start_p_multi = datetime.datetime.now()
	cyclic_multi_factor_pow_seq_python.calc_2_value_sequence_pow_2_many(modulo, amount_factors, all_factors.reshape((-1, )), all_values_p_multi, all_found_cycles_p_multi)
	dt_end_p_multi = datetime.datetime.now()

	assert np.all(all_values_c_single == all_values_p_single)
	assert count_ret_val_c == count_ret_val_p

	assert np.all(all_values_c_multi == all_values_p_multi)
	assert np.all(all_found_cycles_c_multi == all_found_cycles_p_multi)
	assert np.all(all_values_c_single.reshape((-1, )) == all_values_c_multi)
	assert np.all(all_values_p_single.reshape((-1, )) == all_values_p_multi)

	print(f'modulo: {modulo}')
	print(f'amount_factors: {amount_factors}')
	print(f'dt_end_c_single - dt_start_c_single: {(dt_end_c_single-dt_start_c_single).total_seconds()}s')
	print(f'dt_end_p_single - dt_start_p_single: {(dt_end_p_single-dt_start_p_single).total_seconds()}s')
	print(f'dt_end_c_multi - dt_start_c_multi: {(dt_end_c_multi-dt_start_c_multi).total_seconds()}s')
	print(f'dt_end_p_multi - dt_start_p_multi: {(dt_end_p_multi-dt_start_p_multi).total_seconds()}s')


if __name__ == '__main__':
	print('Hello World!')

	simple_function_equal_test_modulo_5()
	many_random_pow_2_cycle_modulo_5_tests()
