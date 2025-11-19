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

def define_rotation_functions(field, rows, cols):
	d_function_rotate = {}

	for i_row in range(0, rows - 1):
		for i_col in range(0, cols - 1):
			def get_func_rotate_once(row, col):
				def rot():
					field[row+0, col+0], field[row+0, col+1], field[row+1, col+1], field[row+1, col+0] = \
					field[row+1, col+0], field[row+0, col+0], field[row+0, col+1], field[row+1, col+1]
				return rot
			d_function_rotate[f'{i_row},{i_col},1'] = get_func_rotate_once(row=i_row, col=i_col)

			def get_func_rotate_twice(row, col):
				def rot():
					field[row+0, col+0], field[row+0, col+1], field[row+1, col+1], field[row+1, col+0] = \
					field[row+1, col+1], field[row+1, col+0], field[row+0, col+0], field[row+0, col+1]
				return rot
			d_function_rotate[f'{i_row},{i_col},2'] = get_func_rotate_twice(row=i_row, col=i_col)

			def get_func_rotate_three_times(row, col):
				def rot():
					field[row+0, col+0], field[row+0, col+1], field[row+1, col+1], field[row+1, col+0] = \
					field[row+0, col+1], field[row+1, col+1], field[row+1, col+0], field[row+0, col+0]
				return rot
			d_function_rotate[f'{i_row},{i_col},3'] = get_func_rotate_three_times(row=i_row, col=i_col)

	return d_function_rotate


def test_define_rotation_functions_on_2x2():
	rows = 2
	cols = 2
	field = (np.arange(0, rows * cols, dtype=np.uint64) + 1).reshape((rows, cols))
	d_function_rotate = define_rotation_functions(field=field, rows=rows, cols=cols)

	# test the clockwise move
	d_function_rotate['0,0,1']()
	assert np.all(field == np.array([[3, 1], [4, 2]], dtype=np.uint64))
	d_function_rotate['0,0,1']()
	assert np.all(field == np.array([[4, 3], [2, 1]], dtype=np.uint64))
	d_function_rotate['0,0,1']()
	assert np.all(field == np.array([[2, 4], [1, 3]], dtype=np.uint64))
	d_function_rotate['0,0,1']()
	assert np.all(field == np.array([[1, 2], [3, 4]], dtype=np.uint64))

	# test the double clockwise or double anti-clockwise move
	d_function_rotate['0,0,2']()
	assert np.all(field == np.array([[4, 3], [2, 1]], dtype=np.uint64))
	d_function_rotate['0,0,2']()
	assert np.all(field == np.array([[1, 2], [3, 4]], dtype=np.uint64))

	# test the anti-clockwise move
	d_function_rotate['0,0,3']()
	assert np.all(field == np.array([[2, 4], [1, 3]], dtype=np.uint64))
	d_function_rotate['0,0,3']()
	assert np.all(field == np.array([[4, 3], [2, 1]], dtype=np.uint64))
	d_function_rotate['0,0,3']()
	assert np.all(field == np.array([[3, 1], [4, 2]], dtype=np.uint64))
	d_function_rotate['0,0,3']()
	assert np.all(field == np.array([[1, 2], [3, 4]], dtype=np.uint64))


def test_define_rotation_functions_on_3x3():
	rows = 3
	cols = 3
	field = (np.arange(0, rows * cols, dtype=np.uint64) + 1).reshape((rows, cols))
	d_function_rotate = define_rotation_functions(field=field, rows=rows, cols=cols)

	field_orig = field.copy()

	# make a neutral move by doing all n moves in reverse
	d_function_rotate['0,0,1']()
	d_function_rotate['0,1,1']()

	d_function_rotate['0,1,3']()
	d_function_rotate['0,0,3']()
	assert np.all(field == field_orig)

	d_function_rotate['0,0,1']()
	d_function_rotate['0,1,1']()
	d_function_rotate['1,1,2']()
	d_function_rotate['1,0,3']()
	
	d_function_rotate['1,0,1']()
	d_function_rotate['1,1,2']()
	d_function_rotate['0,1,3']()
	d_function_rotate['0,0,3']()
	assert np.all(field == field_orig)


test_define_rotation_functions_on_2x2()
test_define_rotation_functions_on_3x3()

if __name__ == '__main__':

	rows = 3
	cols = 3

	field = (np.arange(0, rows * cols, dtype=np.uint64) + 1).reshape((rows, cols))
	field_orig = field.copy()

	print(f'field:\n{field}')

	d_function_rotate = define_rotation_functions(field=field, rows=rows, cols=cols)

	l_move_base_0_0 = ['0,0,1', '0,0,2', '0,0,3', ]
	l_move_base_0_1 = ['0,1,1', '0,1,2', '0,1,3', ]
	l_move_base_1_0 = ['1,0,1', '1,0,2', '1,0,3', ]
	l_move_base_1_1 = ['1,1,1', '1,1,2', '1,1,3', ]
	
	l_move_base = (
		l_move_base_0_0 +
		l_move_base_0_1 +
		l_move_base_1_0 +
		l_move_base_1_1
	)

	d_move_base_to_l_next_move = {
		l_move_base_0_0[0]: l_move_base_0_1 + l_move_base_1_0 + l_move_base_1_1,
		l_move_base_0_0[1]: l_move_base_0_1 + l_move_base_1_0 + l_move_base_1_1,
		l_move_base_0_0[2]: l_move_base_0_1 + l_move_base_1_0 + l_move_base_1_1,

		l_move_base_0_1[0]: l_move_base_0_0 + l_move_base_1_0 + l_move_base_1_1,
		l_move_base_0_1[1]: l_move_base_0_0 + l_move_base_1_0 + l_move_base_1_1,
		l_move_base_0_1[2]: l_move_base_0_0 + l_move_base_1_0 + l_move_base_1_1,

		l_move_base_1_0[0]: l_move_base_0_1 + l_move_base_0_0 + l_move_base_1_1,
		l_move_base_1_0[1]: l_move_base_0_1 + l_move_base_0_0 + l_move_base_1_1,
		l_move_base_1_0[2]: l_move_base_0_1 + l_move_base_0_0 + l_move_base_1_1,

		l_move_base_1_1[0]: l_move_base_0_1 + l_move_base_1_0 + l_move_base_0_0,
		l_move_base_1_1[1]: l_move_base_0_1 + l_move_base_1_0 + l_move_base_0_0,
		l_move_base_1_1[2]: l_move_base_0_1 + l_move_base_1_0 + l_move_base_0_0,
	}

	d_str_field_to_d_l_move = {','.join([str(v) for v in field.reshape((-1, ))]): {0: []}}
	d_str_field_to_d_l_move_next = {}

	for move in l_move_base:
		field[:] = field_orig
		d_function_rotate[move]()
		str_field = ','.join([str(v) for v in field.reshape((-1, ))])
		
		d_str_field_to_d_l_move[str_field] = {1: [[move]]}
		d_str_field_to_d_l_move_next[str_field] = {1: [[move]]}

	amount_moves = 2
	while d_str_field_to_d_l_move_next:
		print(f'amount_moves: {amount_moves}')
		print(f'len(d_str_field_to_d_l_move_next): {len(d_str_field_to_d_l_move_next)}')

		temp_d_str_field_to_d_l_move_next = {}

		for str_field, d_l_move in d_str_field_to_d_l_move_next.items():
			field_2_orig = np.array([int(v) for v in str_field.split(',')], dtype=np.uint64).reshape((rows, cols))
			for l_move in d_l_move[amount_moves - 1]:
				for move in d_move_base_to_l_next_move[l_move[-1]]:
					field[:] = field_2_orig
					d_function_rotate[move]()
					str_field_2 = ','.join([str(v) for v in field.reshape((-1, ))])
					if not str_field_2 in d_str_field_to_d_l_move:
						d_str_field_to_d_l_move[str_field_2] = {amount_moves: [l_move + [move]]}
						temp_d_str_field_to_d_l_move_next[str_field_2] = {amount_moves: [l_move + [move]]}
					else:
						d_l_move = d_str_field_to_d_l_move[str_field_2]
						if not amount_moves in d_l_move:
							d_l_move[amount_moves] = [l_move + [move]]

		print(f'len(d_str_field_to_d_l_move): {len(d_str_field_to_d_l_move)}')
		d_str_field_to_d_l_move_next = temp_d_str_field_to_d_l_move_next
		amount_moves += 1

	sys.exit()

	d_str_field_to_l_move = {','.join([str(v) for v in field.reshape((-1, ))]): []}
	d_str_field_to_l_move_next = {}

	for move in l_move_base:
		field[:] = field_orig
		d_function_rotate[move]()
		str_field = ','.join([str(v) for v in field.reshape((-1, ))])
		
		d_str_field_to_l_move[str_field] = [move]
		d_str_field_to_l_move_next[str_field] = [move]

	amount_moves = 2
	while d_str_field_to_l_move_next:
		print(f'amount_moves: {amount_moves}')
		print(f'len(d_str_field_to_l_move_next): {len(d_str_field_to_l_move_next)}')

		temp_d_str_field_to_l_move_next = {}

		for str_field, l_move in d_str_field_to_l_move_next.items():
			field_2_orig = np.array([int(v) for v in str_field.split(',')], dtype=np.uint64).reshape((rows, cols))

			for move in l_move_base:
				field[:] = field_2_orig
				d_function_rotate[move]()
				str_field = ','.join([str(v) for v in field.reshape((-1, ))])
				if not str_field in d_str_field_to_l_move:
					d_str_field_to_l_move[str_field] = l_move + [move]
					temp_d_str_field_to_l_move_next[str_field] = l_move + [move]

		print(f'len(d_str_field_to_l_move): {len(d_str_field_to_l_move)}')
		d_str_field_to_l_move_next = temp_d_str_field_to_l_move_next
		amount_moves += 1


	# find all moves, where the first row is unchanged
	d_str_field_first_row_unchanged_to_l_l_move = {}
	for str_field, l_move in d_str_field_to_l_move.items():
		if str_field[:5] != '1,2,3':
			continue

		if not str_field in d_str_field_first_row_unchanged_to_l_l_move:
			d_str_field_first_row_unchanged_to_l_l_move[str_field] = []

		d_str_field_first_row_unchanged_to_l_l_move[str_field].append(l_move)

	# find the move with the least amount of moves
	d_str_field_first_row_unchanged_to_l_move_min = {}
	for str_field, l_l_move in d_str_field_first_row_unchanged_to_l_l_move.items():
		if len(l_l_move) == 1:
			d_str_field_first_row_unchanged_to_l_move_min[str_field] = l_l_move[0]

	# >>> np.unique([len(v) for v in d_str_field_first_row_unchanged_to_l_move_min.values()], return_counts=True)
	# (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([  1,   6,  18,  48,  78, 217, 296,  55,   1]))

	# >>> np.unique([len(v) for v in d_str_field_to_l_move_min.values()], return_counts=True)
	# (array([0, 1, 2, 3, 4, 5, 6, 7]), array([  1,   6,  18,  51, 112, 325, 203,   4]))

	d_move_base_to_mirror_horizontal_move = {
		'1,0,1': '1,1,3',
		'1,0,2': '1,1,2',
		'1,0,3': '1,1,1',
		'1,1,1': '1,0,3',
		'1,1,2': '1,0,2',
		'1,1,3': '1,0,1',
		'0,1,1': '0,0,3',
		'0,1,2': '0,0,2',
		'0,1,3': '0,0,1',
		'0,0,1': '0,1,3',
		'0,0,2': '0,1,2',
		'0,0,3': '0,1,1',
	}

	d_str_field_to_l_move_mirror = {}
	for str_field, l_move in d_str_field_first_row_unchanged_to_l_move_min.items():
		l_move_mirror = [d_move_base_to_mirror_horizontal_move[move] for move in l_move]
		field[:] = field_orig
		for move in l_move_mirror:
			d_function_rotate[move]()
		str_field = ','.join([str(v) for v in field.reshape((-1, ))])
		if not str_field in d_str_field_to_l_move_mirror:
			d_str_field_to_l_move_mirror[str_field] = l_move_mirror
		else:
			assert False

	d_str_field_to_l_move_min = {}
	for str_field in d_str_field_to_l_move_mirror.keys():
		l_move = d_str_field_first_row_unchanged_to_l_move_min[str_field]
		l_move_mirror = d_str_field_to_l_move_mirror[str_field]
		if l_move == l_move_mirror:
			d_str_field_to_l_move_min[str_field] = l_move
			continue
		if len(l_move) < len(l_move_mirror):
			d_str_field_to_l_move_min[str_field] = l_move
		elif len(l_move) > len(l_move_mirror):
			d_str_field_to_l_move_min[str_field] = l_move_mirror
		else:
			for move, move_mirror in zip(l_move, l_move_mirror):
				if move < move_mirror:
					d_str_field_to_l_move_min[str_field] = l_move
					break
				elif move > move_mirror:
					d_str_field_to_l_move_min[str_field] = l_move_mirror
					break
