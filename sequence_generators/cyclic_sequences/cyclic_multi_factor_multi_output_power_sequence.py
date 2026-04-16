#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.13 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import itertools
import os
import pdb
import random
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

def general_modulo_polynome_function(modulo, t_factor, l_t_power, t_x):
	mod_sum_all = 0
	
	for factor, t_power in zip(t_factor, l_t_power):
		mod_mult_all = factor
		for power, x in zip(t_power, t_x):
			mod_mult_all = (mod_mult_all * x**power) % modulo
		mod_sum_all = (mod_sum_all + mod_mult_all) % modulo

	return mod_sum_all


def general_modulo_cycle_calculator(modulo, amount_factors, l_t_factor, l_l_t_power):
	# TODO 2025.12.17: create a class, where the pre-calc-table (lookup table) is calculated for every single value of
	# (factor * (t_x[0]**power_0 * ... * t_x[i-1]**power_(i-1))) % modulo
	assert amount_factors == 2

	assert len(l_t_factor) == amount_factors
	assert len(l_l_t_power) == amount_factors

	assert all([len(t_factor) == len(l_t_power) for t_factor, l_t_power in zip(l_t_factor, l_l_t_power)])

	assert all(all([len(t_power) == amount_factors for t_power in l_t_power]) for l_t_power in l_l_t_power)
	assert all(all([all([pow_val >= 0 and pow_val < modulo for pow_val in t_power]) for t_power in l_t_power]) for l_t_power in l_l_t_power)
	assert all(len(set(l_t_power)) == len(l_t_power) for l_t_power in l_l_t_power)

	assert modulo % 2 == 0
	t_factor_1 = l_t_factor[0]
	l_t_power_1 = l_l_t_power[0]

	t_factor_2 = l_t_factor[1]
	l_t_power_2 = l_l_t_power[1]

	cycle_len = modulo**2 // 2

	t_x = (0, 0)
	s_t_x = set([t_x])
	l_t_x = [t_x]
	l_called_function_nr = []

	for _ in range(0, cycle_len):
		l_called_function_nr.append(0)
		x_next_1 = general_modulo_polynome_function(modulo=modulo, t_factor=t_factor_1, l_t_power=l_t_power_1, t_x=t_x)
		t_x_next_1 = (t_x[1], x_next_1)
		
		l_t_x.append(t_x_next_1)
		if t_x_next_1 in s_t_x:
			break

		s_t_x.add(t_x_next_1)
		t_x = t_x_next_1


		l_called_function_nr.append(1)
		x_next_2 = general_modulo_polynome_function(modulo=modulo, t_factor=t_factor_2, l_t_power=l_t_power_2, t_x=t_x)
		t_x_next_2 = (t_x[0], x_next_2)

		l_t_x.append(t_x_next_2)
		if t_x_next_2 in s_t_x:
			break

		s_t_x.add(t_x_next_2)
		t_x = t_x_next_2


	true_cycle_len = len(l_t_x) - l_t_x.index(l_t_x[-1]) - 1

	return true_cycle_len, l_t_x, l_called_function_nr



def exec_create_modulo_function(l_t_power):
	amount_factors = len(l_t_power)
	amount_x = len(l_t_power[0])
	assert all([len(t_power) == amount_x for t_power in l_t_power])

	str_func = """
def modulo_polynome_function(modulo, t_factor, t_x):
	return (
{factor_values_lines}
	) % modulo
"""
	
	str_factor_template = '\t\tt_factor[{i_t_factor}]' + ''.join('*t_x[{i_t_x}]**{power_{i_power_i}}'.replace('{i_t_x}', str(i_t_x)).replace('{i_power_i}', str(i_t_x)) for i_t_x in range(0, amount_x))

	l_str_factors_values_power = []
	for i_t_factor, t_power in enumerate(l_t_power, 0):
		str_factor = str_factor_template.replace('{i_t_factor}', str(i_t_factor))
		for i, power_val in enumerate(t_power, 0):
			str_factor = str_factor.replace('{power_{i}}'.replace('{i}', str(i)), str(power_val))

		l_str_factors_values_power.append(str_factor)

	str_func = str_func.replace('{factor_values_lines}', ' +\n'.join(l_str_factors_values_power))

	d_glob = {}
	d_loc = {}

	exec(str_func, d_glob, d_loc)

	return d_loc['modulo_polynome_function']


"""
modulo: 8
found_max_true_cycle_len: 32
- t_factor_1: (7, 5, 7, 7), t_factor_2: (5, 1, 5)
- l_t_power_1: [(4, 5), (0, 2), (1, 0), (0, 0)], l_t_power_2: [(3, 7), (1, 0), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 7), (7, 5), (5, 0), (0, 2), (2, 3), (3, 7), (7, 2), (2, 4), (4, 5), (5, 1), (1, 6), (6, 6), (6, 5), (5, 3), (3, 4), (4, 0), (0, 3), (3, 5), (5, 4), (4, 2), (2, 7), (7, 7), (7, 6), (6, 4), (4, 1), (1, 1), (1, 2), (2, 6), (6, 1), (1, 3), (3, 0), (0, 0)]

t_x = (t_x[1], t_x_next_1)
t_x = (t_x[1], t_x_next_2)

found_max_true_cycle_len: 48
- t_factor_1: (5, 1), t_factor_2: (3, 1, 4)
- l_t_power_1: [(1, 0), (0, 0)], l_t_power_2: [(6, 6), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 1), (0, 5), (5, 1), (5, 0), (0, 2), (0, 6), (6, 1), (6, 5), (5, 7), (5, 6), (6, 2), (6, 6), (6, 7), (6, 3), (3, 7), (3, 6), (6, 0), (6, 4), (4, 7), (4, 3), (3, 5), (3, 4), (4, 0), (4, 4), (4, 5), (4, 1), (1, 5), (1, 4), (4, 6), (4, 2), (2, 5), (2, 1), (1, 3), (1, 2), (2, 6), (2, 2), (2, 3), (2, 7), (7, 3), (7, 2), (2, 4), (2, 0), (0, 3), (0, 7), (7, 1), (7, 0), (0, 4), (0, 0)]

t_x = (t_x[1], t_x_next_1)
t_x = (t_x[0], t_x_next_2)
 
found_max_true_cycle_len: 48
- t_factor_1: (7, 1, 5), t_factor_2: (7, 1, 4)
- l_t_power_1: [(0, 4), (1, 0), (0, 0)], l_t_power_2: [(5, 7), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 5), (0, 1), (1, 4), (1, 0), (0, 6), (0, 2), (2, 5), (2, 1), (1, 6), (1, 2), (2, 6), (2, 2), (2, 7), (2, 3), (3, 6), (3, 2), (2, 0), (2, 4), (4, 7), (4, 3), (3, 0), (3, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 0), (5, 4), (4, 2), (4, 6), (6, 1), (6, 5), (5, 2), (5, 6), (6, 2), (6, 6), (6, 3), (6, 7), (7, 2), (7, 6), (6, 4), (6, 0), (0, 3), (0, 7), (7, 4), (7, 0), (0, 4), (0, 0)]

found_max_true_cycle_len: 48
- t_factor_1: (1, 5, 3), t_factor_2: (4, 1, 4)
- l_t_power_1: [(0, 6), (1, 0), (0, 0)], l_t_power_2: [(2, 6), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 3), (0, 7), (7, 4), (7, 0), (0, 6), (0, 2), (2, 3), (2, 7), (7, 6), (7, 2), (2, 6), (2, 2), (2, 5), (2, 1), (1, 6), (1, 2), (2, 0), (2, 4), (4, 5), (4, 1), (1, 0), (1, 4), (4, 0), (4, 4), (4, 7), (4, 3), (3, 0), (3, 4), (4, 2), (4, 6), (6, 7), (6, 3), (3, 2), (3, 6), (6, 2), (6, 6), (6, 1), (6, 5), (5, 2), (5, 6), (6, 4), (6, 0), (0, 1), (0, 5), (5, 4), (5, 0), (0, 4), (0, 0)]

found_max_true_cycle_len: 48
- t_factor_1: (5, 7, 7), t_factor_2: (7, 1, 4)
- l_t_power_1: [(4, 2), (1, 0), (0, 0)], l_t_power_2: [(4, 0), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 7), (0, 3), (3, 7), (3, 2), (2, 0), (2, 4), (4, 5), (4, 1), (1, 3), (1, 6), (6, 2), (6, 6), (6, 1), (6, 5), (5, 1), (5, 4), (4, 2), (4, 6), (6, 3), (6, 7), (7, 1), (7, 4), (4, 0), (4, 4), (4, 3), (4, 7), (7, 3), (7, 6), (6, 4), (6, 0), (0, 1), (0, 5), (5, 7), (5, 2), (2, 6), (2, 2), (2, 5), (2, 1), (1, 5), (1, 0), (0, 6), (0, 2), (2, 7), (2, 3), (3, 5), (3, 0), (0, 4), (0, 0)]

found_max_true_cycle_len: 48
- t_factor_1: (7, 5, 5), t_factor_2: (7, 1, 4)
- l_t_power_1: [(3, 5), (1, 0), (0, 0)], l_t_power_2: [(6, 0), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 5), (0, 1), (1, 5), (1, 0), (0, 2), (0, 6), (6, 5), (6, 1), (1, 3), (1, 6), (6, 2), (6, 6), (6, 3), (6, 7), (7, 3), (7, 6), (6, 0), (6, 4), (4, 3), (4, 7), (7, 1), (7, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 1), (5, 4), (4, 6), (4, 2), (2, 1), (2, 5), (5, 7), (5, 2), (2, 6), (2, 2), (2, 7), (2, 3), (3, 7), (3, 2), (2, 4), (2, 0), (0, 7), (0, 3), (3, 5), (3, 0), (0, 4), (0, 0)]

found_max_true_cycle_len: 48
- t_factor_1: (6, 7, 5), t_factor_2: (1, 1, 4)
- l_t_power_1: [(2, 1), (1, 0), (0, 0)], l_t_power_2: [(4, 4), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 5), (0, 1), (1, 5), (1, 2), (2, 0), (2, 4), (4, 3), (4, 7), (7, 1), (7, 6), (6, 2), (6, 6), (6, 7), (6, 3), (3, 7), (3, 4), (4, 2), (4, 6), (6, 1), (6, 5), (5, 7), (5, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 1), (5, 6), (6, 4), (6, 0), (0, 7), (0, 3), (3, 5), (3, 2), (2, 6), (2, 2), (2, 3), (2, 7), (7, 3), (7, 0), (0, 6), (0, 2), (2, 5), (2, 1), (1, 3), (1, 0), (0, 4), (0, 0)]

found_max_true_cycle_len: 48
- t_factor_1: (6, 3, 5), t_factor_2: (5, 1, 4)
- l_t_power_1: [(4, 1), (1, 0), (0, 0)], l_t_power_2: [(4, 6), (0, 1), (0, 0)]
- l_called_function_nr: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- l_t_x: [(0, 0), (0, 5), (0, 1), (1, 5), (1, 6), (6, 4), (6, 0), (0, 7), (0, 3), (3, 5), (3, 6), (6, 2), (6, 6), (6, 7), (6, 3), (3, 7), (3, 0), (0, 6), (0, 2), (2, 5), (2, 1), (1, 3), (1, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 1), (5, 2), (2, 0), (2, 4), (4, 3), (4, 7), (7, 1), (7, 2), (2, 6), (2, 2), (2, 3), (2, 7), (7, 3), (7, 4), (4, 2), (4, 6), (6, 1), (6, 5), (5, 7), (5, 0), (0, 4), (0, 0)]


"""
def find_full_cycle_amount_factor_2_for_modulo_8():
	modulo = 8
	l_t_power_all = sorted(list(itertools.product(*[list(range(0, modulo))]*2)), key=lambda x: (-max(x), [-v for v in x[::-1]]))
	# a zero tuple must be always included
	t_power_zeros = l_t_power_all.pop()
	assert t_power_zeros == (0, 0)

	len_l_t_power_1 = 4
	len_l_t_power_2 = 4
	d_true_cycle_len_to_t_l_t_factor_l_l_t_power = {}
	amount_factors = 2
	amount_functions = 2
	cycle_len = modulo**amount_factors // amount_functions

	found_max_true_cycle_len = 0
	found_max_l_t_x = []
	found_max_t_factors = ()
	found_max_l_called_function_nr = ()

	for i_try in range(0, 100000000):
		l_called_function_nr = []

		l_rand_1 = sorted(random.sample(list(range(0, len(l_t_power_all))), len_l_t_power_1-1))
		l_t_power_1 = [l_t_power_all[i] for i in l_rand_1] + [t_power_zeros]
		modulo_polynome_function_1 = exec_create_modulo_function(l_t_power=l_t_power_1)
		t_factor_1 = tuple(random.choices(list(range(0, modulo)), weights=[1]*modulo, k=len(l_t_power_1)))

		l_rand_2 = sorted(random.sample(list(range(0, len(l_t_power_all))), len_l_t_power_2-1))
		l_t_power_2 = [l_t_power_all[i] for i in l_rand_2] + [t_power_zeros]
		modulo_polynome_function_2 = exec_create_modulo_function(l_t_power=l_t_power_2)
		t_factor_2 = tuple(random.choices(list(range(0, modulo)), weights=[1]*modulo, k=len(l_t_power_2)))

		t_x = (0, 0)
		s_t_x = set([t_x])
		l_t_x = [t_x]

		for _ in range(0, cycle_len):
			l_called_function_nr.append(0)
			x_next_1 = modulo_polynome_function_1(modulo=modulo, t_factor=t_factor_1, t_x=t_x)
			t_x_next_1 = (t_x[1], (t_x[0]+(t_x[1]+x_next_1)%3)%modulo)
			
			l_t_x.append(t_x_next_1)
			if t_x_next_1 in s_t_x:
				break

			s_t_x.add(t_x_next_1)
			t_x = t_x_next_1


			l_called_function_nr.append(1)
			x_next_2 = modulo_polynome_function_2(modulo=modulo, t_factor=t_factor_2, t_x=t_x)
			t_x_next_2 = (t_x[1], x_next_2)

			l_t_x.append(t_x_next_2)
			if t_x_next_2 in s_t_x:
				break

			s_t_x.add(t_x_next_2)
			t_x = t_x_next_2


		true_cycle_len = len(l_t_x) - l_t_x.index(l_t_x[-1]) - 1

		if found_max_true_cycle_len <= true_cycle_len:
			found_max_true_cycle_len = true_cycle_len
			found_max_l_t_x = l_t_x
			
			found_max_t_factors = (t_factor_1, t_factor_2)
			found_max_l_l_t_power = [l_t_power_1, l_t_power_2]

			found_max_l_called_function_nr = l_called_function_nr

			print(f"found_max_true_cycle_len: {found_max_true_cycle_len}")
			print('t_x_next_1 = (t_x[1], (t_x[0]+(t_x[1]+x_next_1)%3)%modulo)')
			print('t_x_next_2 = (t_x[1], x_next_2)')
			print(f"- t_factor_1: {t_factor_1}, t_factor_2: {t_factor_2}")
			print(f"- l_t_power_1: {l_t_power_1}, l_t_power_2: {l_t_power_2}")
			print(f"- l_called_function_nr: {l_called_function_nr}")

			print(f"- l_t_x: {l_t_x}")

			if not true_cycle_len in d_true_cycle_len_to_t_l_t_factor_l_l_t_power:
				d_true_cycle_len_to_t_l_t_factor_l_l_t_power[true_cycle_len] = []

			d_true_cycle_len_to_t_l_t_factor_l_l_t_power[true_cycle_len].append(([t_factor_1, t_factor_2], [l_t_power_1, l_t_power_2]))

	print(f"found_max_true_cycle_len: {found_max_true_cycle_len}")
	print(f"found_max_l_t_x: {found_max_l_t_x}")
	print(f"found_max_t_factors: {found_max_t_factors}")
	print(f"found_max_l_called_function_nr: {found_max_l_called_function_nr}")

	return locals()


def find_full_cycle_amount_factor_2(modulo):
	assert modulo % 2 == 0

	l_t_power_all = sorted(list(itertools.product(*[list(range(0, modulo))]*2)), key=lambda x: (-max(x), [-v for v in x[::-1]]))
	# a zero tuple must be always included
	t_power_zeros = l_t_power_all.pop()
	assert t_power_zeros == (0, 0)

	len_l_t_power_1 = 3
	len_l_t_power_2 = 3
	d_true_cycle_len_to_t_l_t_factor_l_l_t_power = {}
	amount_factors = 2
	amount_functions = 2
	cycle_len_full = modulo**amount_factors
	cycle_len = cycle_len_full // amount_functions

	found_max_true_cycle_len = 0
	found_max_l_t_x = []
	found_max_t_factors = ()
	found_max_l_called_function_nr = ()

	for i_try in range(0, 100000000):
		l_called_function_nr = []

		l_rand_1 = sorted(random.sample(list(range(0, len(l_t_power_all))), len_l_t_power_1-1))
		l_t_power_1 = [l_t_power_all[i] for i in l_rand_1] + [t_power_zeros]
		modulo_polynome_function_1 = exec_create_modulo_function(l_t_power=l_t_power_1)
		t_factor_1 = tuple(random.choices(list(range(0, modulo)), weights=[1]*modulo, k=len(l_t_power_1)))

		l_rand_2 = sorted(random.sample(list(range(0, len(l_t_power_all))), len_l_t_power_2-1))
		l_t_power_2 = [l_t_power_all[i] for i in l_rand_2] + [t_power_zeros]
		modulo_polynome_function_2 = exec_create_modulo_function(l_t_power=l_t_power_2)
		t_factor_2 = tuple(random.choices(list(range(0, modulo)), weights=[1]*modulo, k=len(l_t_power_2)))

		t_x = (0, 0)
		s_t_x = set([t_x])
		l_t_x = [t_x]

		for _ in range(0, cycle_len):
			l_called_function_nr.append(0)
			x_next_1 = modulo_polynome_function_1(modulo=modulo, t_factor=t_factor_1, t_x=t_x)
			t_x_next_1 = (t_x[1], x_next_1)
			
			l_t_x.append(t_x_next_1)
			if t_x_next_1 in s_t_x:
				break

			s_t_x.add(t_x_next_1)
			t_x = t_x_next_1


			l_called_function_nr.append(1)
			x_next_2 = modulo_polynome_function_2(modulo=modulo, t_factor=t_factor_2, t_x=t_x)
			t_x_next_2 = (t_x[0], x_next_2)

			l_t_x.append(t_x_next_2)
			if t_x_next_2 in s_t_x:
				break

			s_t_x.add(t_x_next_2)
			t_x = t_x_next_2


		true_cycle_len = len(l_t_x) - l_t_x.index(l_t_x[-1]) - 1

		if found_max_true_cycle_len <= true_cycle_len:
			found_max_true_cycle_len = true_cycle_len
			found_max_l_t_x = l_t_x
			
			found_max_t_factors = (t_factor_1, t_factor_2)
			found_max_l_l_t_power = [l_t_power_1, l_t_power_2]

			found_max_l_called_function_nr = l_called_function_nr

			print(f"found_max_true_cycle_len: {found_max_true_cycle_len}")
			print(f"modulo: {modulo}")
			print('t_x_next_1 = (t_x[1], x_next_1)')
			print('t_x_next_2 = (t_x[0], x_next_2)')
			print(f"- t_factor_1: {t_factor_1}, t_factor_2: {t_factor_2}")
			print(f"- l_t_power_1: {l_t_power_1}, l_t_power_2: {l_t_power_2}")
			print(f"- l_called_function_nr: {l_called_function_nr}")

			print(f"- l_t_x: {l_t_x}")

			if not true_cycle_len in d_true_cycle_len_to_t_l_t_factor_l_l_t_power:
				d_true_cycle_len_to_t_l_t_factor_l_l_t_power[true_cycle_len] = []

			d_true_cycle_len_to_t_l_t_factor_l_l_t_power[true_cycle_len].append(([t_factor_1, t_factor_2], [l_t_power_1, l_t_power_2]))

	print(f"found_max_true_cycle_len: {found_max_true_cycle_len}")
	print(f"found_max_l_t_x: {found_max_l_t_x}")
	print(f"found_max_t_factors: {found_max_t_factors}")
	print(f"found_max_l_called_function_nr: {found_max_l_called_function_nr}")

	return locals()


# Challange 2025.12.16:
# Find somehow any possible way to make a 2 factor full cyclic for modulo 8,
# which means a cycle lenght of 64
if __name__ == '__main__':
	l_ref_values = [
		{
			'modulo': 8,
			'amount_factors': 2,
			'l_t_factor': [(7, 1, 5), (7, 1, 4)],
			'l_l_t_power': [[(0, 4), (1, 0), (0, 0)], [(5, 7), (0, 1), (0, 0)]],
			'true_cycle_len': 48,
			'l_t_x': [(0, 0), (0, 5), (0, 1), (1, 4), (1, 0), (0, 6), (0, 2), (2, 5), (2, 1), (1, 6), (1, 2), (2, 6), (2, 2), (2, 7), (2, 3), (3, 6), (3, 2), (2, 0), (2, 4), (4, 7), (4, 3), (3, 0), (3, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 0), (5, 4), (4, 2), (4, 6), (6, 1), (6, 5), (5, 2), (5, 6), (6, 2), (6, 6), (6, 3), (6, 7), (7, 2), (7, 6), (6, 4), (6, 0), (0, 3), (0, 7), (7, 4), (7, 0), (0, 4), (0, 0)],
			'l_called_function_nr': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		},
		{
			'modulo': 8,
			'amount_factors': 2,
			'l_t_factor': [(1, 5, 3), (4, 1, 4)],
			'l_l_t_power': [[(0, 6), (1, 0), (0, 0)], [(2, 6), (0, 1), (0, 0)]],
			'true_cycle_len': 48,
			'l_t_x': [(0, 0), (0, 3), (0, 7), (7, 4), (7, 0), (0, 6), (0, 2), (2, 3), (2, 7), (7, 6), (7, 2), (2, 6), (2, 2), (2, 5), (2, 1), (1, 6), (1, 2), (2, 0), (2, 4), (4, 5), (4, 1), (1, 0), (1, 4), (4, 0), (4, 4), (4, 7), (4, 3), (3, 0), (3, 4), (4, 2), (4, 6), (6, 7), (6, 3), (3, 2), (3, 6), (6, 2), (6, 6), (6, 1), (6, 5), (5, 2), (5, 6), (6, 4), (6, 0), (0, 1), (0, 5), (5, 4), (5, 0), (0, 4), (0, 0)],
			'l_called_function_nr': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		},
		{
			'modulo': 8,
			'amount_factors': 2,
			'l_t_factor': [(5, 7, 7), (7, 1, 4)],
			'l_l_t_power': [[(4, 2), (1, 0), (0, 0)], [(4, 0), (0, 1), (0, 0)]],
			'true_cycle_len': 48,
			'l_t_x': [(0, 0), (0, 7), (0, 3), (3, 7), (3, 2), (2, 0), (2, 4), (4, 5), (4, 1), (1, 3), (1, 6), (6, 2), (6, 6), (6, 1), (6, 5), (5, 1), (5, 4), (4, 2), (4, 6), (6, 3), (6, 7), (7, 1), (7, 4), (4, 0), (4, 4), (4, 3), (4, 7), (7, 3), (7, 6), (6, 4), (6, 0), (0, 1), (0, 5), (5, 7), (5, 2), (2, 6), (2, 2), (2, 5), (2, 1), (1, 5), (1, 0), (0, 6), (0, 2), (2, 7), (2, 3), (3, 5), (3, 0), (0, 4), (0, 0)],
			'l_called_function_nr': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		},
		{
			'modulo': 8,
			'amount_factors': 2,
			'l_t_factor': [(7, 5, 5), (7, 1, 4)],
			'l_l_t_power': [[(3, 5), (1, 0), (0, 0)], [(6, 0), (0, 1), (0, 0)]],
			'true_cycle_len': 48,
			'l_t_x': [(0, 0), (0, 5), (0, 1), (1, 5), (1, 0), (0, 2), (0, 6), (6, 5), (6, 1), (1, 3), (1, 6), (6, 2), (6, 6), (6, 3), (6, 7), (7, 3), (7, 6), (6, 0), (6, 4), (4, 3), (4, 7), (7, 1), (7, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 1), (5, 4), (4, 6), (4, 2), (2, 1), (2, 5), (5, 7), (5, 2), (2, 6), (2, 2), (2, 7), (2, 3), (3, 7), (3, 2), (2, 4), (2, 0), (0, 7), (0, 3), (3, 5), (3, 0), (0, 4), (0, 0)],
			'l_called_function_nr': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		},
		{
			'modulo': 8,
			'amount_factors': 2,
			'l_t_factor': [(6, 7, 5), (1, 1, 4)],
			'l_l_t_power': [[(2, 1), (1, 0), (0, 0)], [(4, 4), (0, 1), (0, 0)]],
			'true_cycle_len': 48,
			'l_t_x': [(0, 0), (0, 5), (0, 1), (1, 5), (1, 2), (2, 0), (2, 4), (4, 3), (4, 7), (7, 1), (7, 6), (6, 2), (6, 6), (6, 7), (6, 3), (3, 7), (3, 4), (4, 2), (4, 6), (6, 1), (6, 5), (5, 7), (5, 4), (4, 0), (4, 4), (4, 1), (4, 5), (5, 1), (5, 6), (6, 4), (6, 0), (0, 7), (0, 3), (3, 5), (3, 2), (2, 6), (2, 2), (2, 3), (2, 7), (7, 3), (7, 0), (0, 6), (0, 2), (2, 5), (2, 1), (1, 3), (1, 0), (0, 4), (0, 0)],
			'l_called_function_nr': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		}
	]

	for d in l_ref_values:
		print(f'Test values: {d}')

		modulo_ref = d['modulo']
		amount_factors_ref = d['amount_factors']
		l_t_factor_ref = d['l_t_factor']
		l_l_t_power_ref = d['l_l_t_power']
		true_cycle_len_ref = d['true_cycle_len']
		l_t_x_ref = d['l_t_x']
		l_called_function_nr_ref = d['l_called_function_nr']

		true_cycle_len, l_t_x, l_called_function_nr = general_modulo_cycle_calculator(
			modulo=modulo_ref, amount_factors=amount_factors_ref, l_t_factor=l_t_factor_ref, l_l_t_power=l_l_t_power_ref,
		)

		assert true_cycle_len == true_cycle_len_ref
		assert l_t_x == l_t_x_ref
		assert l_called_function_nr == l_called_function_nr_ref

	# d_loc = find_full_cycle_amount_factor_2_for_modulo_8()
	d_loc = find_full_cycle_amount_factor_2(modulo=6)
