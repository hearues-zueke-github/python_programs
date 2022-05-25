#! /usr/bin/python3.10

# pip installed libraries
import dill
import gzip
import keyboard
import os
import requests
import sh
import string
import subprocess
import sys
import time
import tty
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from copy import deepcopy
from hashlib import sha256
from io import StringIO
from memory_tempfile import MemoryTempfile
from multiprocessing import Pool
from PIL import Image
from typing import Dict

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

HOME_DIR = os.path.expanduser("~")
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_serialization', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_serialization.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))
# load_module_dynamically(**dict(var_glob=var_glob, name='utils_graph_theory', path=os.path.join(PYTHON_PROGRAMS_DIR, "graph_theory/utils_graph_theory.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

save_pkl_obj = utils_serialization.save_pkl_obj
load_pkl_obj = utils_serialization.load_pkl_obj

get_all_combinations_repeat = different_combinations.get_all_combinations_repeat
# get_cycles_of_1_directed_graph = utils_graph_theory.get_cycles_of_1_directed_graph

def get_full_mult_cycle_if_possible(a, b, m):
	x = 0
	s = set([0])
	for _ in range(0, m):
		x = (a * x + b) % m
		if x in s:
			break
		s.add(x)

	is_full_cycle = False
	l_cycle = []
	if len(s) == m:
		is_full_cycle = True
		x = 0
		l_cycle = [0]
		for _ in range(0, m-1):
			x = (a * x + b) % m
			l_cycle.append(x)

	return is_full_cycle, l_cycle


def get_full_mult_pow_cycle_if_possible(a, b, m):
	# p = 11
	x = 0
	s = set([0])
	for _ in range(0, m):
		x = (a ^ (x >> 1) + b) % m
		if x in s:
			break
		s.add(x)

	is_full_cycle = False
	l_cycle = []
	if len(s) == m:
		is_full_cycle = True
		x = 0
		l_cycle = [0]
		for _ in range(0, m-1):
			x = (a ^ (x >> 1) + b) % m
			l_cycle.append(x)

	return is_full_cycle, l_cycle


def get_full_xor_cycle_if_possible(a, b, m):
	x = 0
	s = set([0])
	for _ in range(0, m):
		x = (a ^ x + b) % m
		if x in s:
			break
		s.add(x)

	is_full_cycle = False
	l_cycle = []
	if len(s) == m:
		is_full_cycle = True
		x = 0
		l_cycle = [0]
		for _ in range(0, m-1):
			x = (a ^ x + b) % m
			l_cycle.append(x)

	return is_full_cycle, l_cycle


if __name__ == '__main__':
	n = 2

	d_l_cycle_info_mult = {}
	d_m_to_l_cycle_mult = {}

	d_l_cycle_info_mult_pow = {}
	d_m_to_l_cycle_mult_pow = {}

	d_l_cycle_info_xor = {}
	d_m_to_l_cycle_xor = {}
	d_m_to_d_t_cycle_to_l_info = {}

	for m in range(2, 31):
		if m % 4 != 0:
			print(f"m: {m}, ", end='')
		else:
			print(f"m: {m}, ")
		
		arr_comb = get_all_combinations_repeat(m=m, n=n)

		d_l_cycle_mult = {}
		l_cycle_info_mult = []
		for a, b in arr_comb:
			is_full_cycle, l_cycle = get_full_mult_cycle_if_possible(a=a, b=b, m=m)

			if is_full_cycle:
				l_cycle_info_mult.append((a, b))
				d_l_cycle_mult[(a, b)] = l_cycle

		d_l_cycle_info_mult[m] = l_cycle_info_mult
		d_m_to_l_cycle_mult[m] = d_l_cycle_mult


		d_l_cycle_mult_pow = {}
		l_cycle_info_mult_pow = []
		for a, b in arr_comb:
			is_full_cycle, l_cycle = get_full_mult_pow_cycle_if_possible(a=a, b=b, m=m)

			if is_full_cycle:
				l_cycle_info_mult_pow.append((a, b))
				d_l_cycle_mult_pow[(a, b)] = l_cycle

		d_l_cycle_info_mult_pow[m] = l_cycle_info_mult_pow
		d_m_to_l_cycle_mult_pow[m] = d_l_cycle_mult_pow
		

		d_l_cycle_xor = {}
		l_cycle_info_xor = []
		for a, b in arr_comb:
			is_full_cycle, l_cycle = get_full_xor_cycle_if_possible(a=a, b=b, m=m)

			if is_full_cycle:
				l_cycle_info_xor.append((a, b))
				d_l_cycle_xor[(a, b)] = l_cycle

		d_l_cycle_info_xor[m] = l_cycle_info_xor
		d_m_to_l_cycle_xor[m] = d_l_cycle_xor

		d_comb = {}
		for t_key, l in d_l_cycle_xor.items():
			t_val = tuple(l)
			if t_val not in d_comb:
				d_comb[t_val] = []
			d_comb[t_val].append(t_key)

		d_m_to_d_t_cycle_to_l_info[m] = d_comb

	m = 16
	d_mult = d_m_to_l_cycle_mult[m]
	d_xor = d_m_to_l_cycle_xor[m]

	d_comb_xor = {}
	for t_key, l in d_xor.items():
		t_val = tuple(l)
		if t_val not in d_comb_xor:
			d_comb_xor[t_val] = []
		d_comb_xor[t_val].append(('xor', t_key))

	d_comb = deepcopy(d_comb_xor)
	for t_key, l in d_mult.items():
		t_val = tuple(l)
		if t_val not in d_comb:
			d_comb[t_val] = []
		d_comb[t_val].append(('mult', t_key))
