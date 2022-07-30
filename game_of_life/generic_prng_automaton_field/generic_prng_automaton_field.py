#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import ast
import cv2
import datetime
import dill
import gzip
import imageio
import os
import pdb
import re
import sys
import traceback

import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt

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

from numpy.random import Generator, PCG64

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
load_module_dynamically(**dict(var_glob=var_glob, name='utils_random', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils/utils_random.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_array', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils/utils_array.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_cluster', path=os.path.join(PYTHON_PROGRAMS_DIR, "clustering/utils_cluster.py")))

mkdirs = utils.mkdirs

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_random_seed_256_bits = utils_random.get_random_seed_256_bits

increment_arr_uint8 = utils_array.increment_arr_uint8

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

class Field2d:
	__slot__ = ["frame", "height", "width", "d_var_arr"]

	def __init__(self, frame, height, width, l_seed=[0x00]):
		self.frame = frame
		self.height = height
		self.width = width
		
		self.arr_with_frame = np.zeros((self.height+self.frame*2, self.width+self.frame*2), dtype=np.uint8)

		self.init_d_var_arr()

		self.arr_seed_base_orig = np.array(l_seed, dtype=np.uint8)

		self.arr_seed_func_orig = np.hstack((self.arr_seed_base_orig, np.array([0x01], dtype=np.uint8)))
		self.arr_seed_pix_orig = np.hstack((self.arr_seed_base_orig, np.array([0x02], dtype=np.uint8)))
		
		self.arr_seed_func = self.arr_seed_func_orig.copy()
		self.arr_seed_pix = self.arr_seed_pix_orig.copy()
		
		self.reset_rnds()


	def increment_arr_seeds(self):
		increment_arr_uint8(arr=self.arr_seed_func)
		increment_arr_uint8(arr=self.arr_seed_pix)


	def reset_rnds(self):
		self.rnd_func = Generator(bit_generator=PCG64(seed=self.arr_seed_func))
		self.rnd_pix = Generator(bit_generator=PCG64(seed=self.arr_seed_pix))


	def init_d_var_arr(self):
		l_up = [('u', i, -i, 0) for i in range(self.frame, 0, -1)]
		l_down = [('d', i, i, 0) for i in range(1, self.frame+1)]
		l_left = [('l', i, 0, -i) for i in range(self.frame, 0, -1)]
		l_right = [('r', i, 0, i) for i in range(1, self.frame+1)]

		self.d_var_dir_many_ltrs = {"n": (0, 0)} # many letters as a variable name
		self.d_var_dir_mult_nmbr = {"n": (0, 0)} # the number multiplies the variable name

		for var, amount, dir_y, dir_x in l_up+l_down+l_left+l_right:
			self.d_var_dir_many_ltrs[var*amount] = (dir_y, dir_x)
			self.d_var_dir_mult_nmbr[var+str(amount)] = (dir_y, dir_x)

		for var_1, amount_1, dir_y_1, dir_x_1 in l_up+l_down:
			for var_2, amount_2, dir_y_2, dir_x_2 in l_left+l_right:
				self.d_var_dir_many_ltrs[var_1*amount_1+var_2*amount_2] = (dir_y_1+dir_y_2, dir_x_1+dir_x_2)
				self.d_var_dir_mult_nmbr[var_1+str(amount_1)+var_2+str(amount_2)] = (dir_y_1+dir_y_2, dir_x_1+dir_x_2)

		self.d_var_dir_all = self.d_var_dir_many_ltrs | self.d_var_dir_mult_nmbr

		self.d_var_arr = {}
		for var, (dir_y, dir_x) in self.d_var_dir_all.items():
			self.d_var_arr[var] = self.arr_with_frame[self.frame+dir_y:self.frame+dir_y+self.height, self.frame+dir_x:self.frame+dir_x+self.width]


	def set_points_random(self, density: float):
		assert (density >= 0.) and (density <= 1.)

		amount_all_points = self.height*self.width
		amount_points = int(amount_all_points*density)
		arr_num = self.rnd_pix.choice(np.arange(0, amount_all_points), size=amount_points, replace=False)
		arr_y = arr_num // self.width
		arr_x = arr_num % self.width

		self.set_points(arr_y=arr_y, arr_x=arr_x)


	def set_points(self, arr_y: np.ndarray, arr_x: np.ndarray):
		self.arr_with_frame[:] = 0 # reset every value to 0 first
		self.arr_with_frame[arr_y+self.frame, arr_x+self.frame] = 1
		self.redefine_frame()


	def set_arr_no_frame(self, arr_no_frame: np.ndarray):
		assert arr_no_frame.shape == (self.height, self.width)
		assert arr_no_frame.dtype == self.arr_with_frame.dtype
		self.arr_with_frame[self.frame:-self.frame, self.frame:-self.frame] = arr_no_frame
		self.redefine_frame()


	def redefine_frame(self):
		self.arr_with_frame[:, :self.frame] = self.arr_with_frame[:, -self.frame*2:-self.frame]
		self.arr_with_frame[:, -self.frame:] = self.arr_with_frame[:, self.frame:self.frame*2]
		self.arr_with_frame[:self.frame, :] = self.arr_with_frame[-self.frame*2:-self.frame, :]
		self.arr_with_frame[-self.frame:, :] = self.arr_with_frame[self.frame:self.frame*2, :]


	def get_arr_no_frame(self):
		return self.arr_with_frame[self.frame:-self.frame, self.frame:-self.frame].copy()


	def define_functions(self, func_str: str):
		self.func_str = func_str
		self.d_func_other = {}
		func_str_other = "def inv(a): return a ^ 0x01"
		# numpy as np is needed for the global dict
		exec(func_str_other, {'np': np}, self.d_func_other)
		self.d_func = {}

		exec(func_str, {key: val for key, val in self.d_var_arr.items()} | self.d_func_other | {'np': np}, self.d_func)
		assert 'l_func' in self.d_func
		try:
			for d in self.d_func['l_func']:
				assert isinstance(d, dict)
				function_name = d['function_name']
				func = d['func']
				arr = func()
				assert arr.shape == (self.height, self.width)
				assert arr.dtype == self.arr_with_frame.dtype
		except:
			print(f"Problem in the function with the name '{function_name}'")
			assert False

		self.l_func = self.d_func['l_func']

	# only when one function is used, finding duplicates would make sense
	def play_the_field(self, iter_round_max=300):
		l_arr_no_frame = [self.get_arr_no_frame()]

		l_func = self.l_func
		arr_func_nr = self.rnd_func.integers(0, len(l_func), (iter_round_max, ))
		for iter_round, function_nr in enumerate(arr_func_nr, 0):
			func = l_func[function_nr]['func']

			arr_no_frame = func().copy()
			self.set_arr_no_frame(arr_no_frame=arr_no_frame)
			
			arr_no_frame_2 = self.get_arr_no_frame()
			assert np.all(arr_no_frame == arr_no_frame_2)

			# is_dup_found = False
			# for i, arr in enumerate(l_arr_no_frame):
			# 	if np.all(arr == arr_no_frame):
			# 		is_dup_found = True
			# 		break

			l_arr_no_frame.append(arr_no_frame)
			# if is_dup_found:
			# 	print(f"i_dup: {i}, len(l_arr_no_frame): {len(l_arr_no_frame)}")
			# 	break

		return l_arr_no_frame


def plot_all_field_in_one_picture(l_arr: List[np.ndarray], rows: int, columns: int, border_width: int) -> np.ndarray:
	assert all([isinstance(arr, np.ndarray) for arr in l_arr])
	shape = l_arr[0].shape
	# check, if all arr's are the same size!
	assert all([arr.shape == shape for arr in l_arr])

	height, width = shape

	if len(l_arr) > rows * columns:
		l_arr = l_arr[:rows*columns]

	pix_height = rows*height + (rows+1)*border_width
	pix_width = columns*width + (columns+1)*border_width
	pix_comb = np.zeros((pix_height, pix_width), dtype=np.uint8)
	pix_comb[:] = 128 # this is the default color of the frames
	for i, arr in enumerate(l_arr, 0):
		y_off = border_width+(height+border_width)*(i // columns)
		x_off = border_width+(width+border_width)*(i % columns)
		pix_comb[y_off:y_off+height, x_off:x_off+width] = arr*255

	return pix_comb


class Function():
	def __init__(self):
		self.function_nr = None
		self.function_name = None
		self.tpl_equation = None


	def __repr__(self):
		return self.__str__()
	

	def __str__(self):
		return (
			"Function(" +
			f"function_nr={self.function_nr}, " +
			f"function_name='{self.function_name}', " +
			f"tpl_equation={tuple(str(equation) for equation in self.tpl_equation)}" +
			")"
		)


	def __lt__(self, other):
		assert isinstance(other, Function)

		return self.tpl_equation < other.tpl_equation


class Equation():
	def __init__(self):
		self.equation_nr = None
		self.tpl_var = None
		self.tpl_range = None
		# TODO: add the sign of each variable too!


	def __repr__(self):
		return self.__str__()
	

	def __str__(self):
		return (
			"Equation(" +
			f"equation_nr={self.equation_nr}, " +
			f"tpl_var={self.tpl_var}, " +
			f"tpl_range={self.tpl_range}" +
			")"
		)


	def __lt__(self, other):
		assert isinstance(other, Equation)

		return self.tpl_var < other.tpl_var


def generate_tpl_function_func_str(
	field_2d: Field2d,
	l_seed_body: Union[List[int], np.ndarray],
	l_seed_range: Union[List[int], np.ndarray],
) -> Tuple[Tuple[Function], str]:
	arr_seed_body = np.array(l_seed_body, dtype=np.uint8)
	rnd_body = Generator(bit_generator=PCG64(seed=arr_seed_body))
	arr_seed_range = np.array(l_seed_range, dtype=np.uint8)
	rnd_range = Generator(bit_generator=PCG64(seed=arr_seed_range))

	min_amount_func = 1
	max_amount_func = 2

	d_var_dir_mult_nmbr = field_2d.d_var_dir_mult_nmbr
	l_var = sorted(d_var_dir_mult_nmbr.keys())

	min_amount_equation = 1
	max_amount_equation = 2

	min_amount_var = 1
	max_amount_var = 4
	assert max_amount_var <= len(l_var)

	s_function = set()
	amount_function_should = rnd_body.integers(min_amount_func, max_amount_func)
	for _ in range(0, amount_function_should):
		# first gather all equations with the variables
		s_equation = set()
		amount_equation_should = rnd_body.integers(min_amount_equation, max_amount_equation)
		for _ in range(0, amount_equation_should):
			amount_var = rnd_body.integers(min_amount_var, max_amount_var)
			arr_var = rnd_body.choice(l_var, size=amount_var, replace=False)
			tpl_var = tuple(sorted(arr_var.tolist()))

			len_tpl_var = len(tpl_var)

			# TODO: add the negative numbers too! not only plus is possible, but also minus!
			arr_idx_range = rnd_range.integers(0, 2, (len_tpl_var+1, ))
			arr_idx_range[:rnd_range.integers(1, len_tpl_var+1)] = 1 # set 1 to all values to 1, at least one must be always 1!
			rnd_range.shuffle(arr_idx_range)

			arr_idx_range_pad = np.pad(arr_idx_range, (1, 1), mode="constant", constant_values=0)
			
			arr_idx_1 = np.where((arr_idx_range_pad[1:-1] != arr_idx_range_pad[:-2]) & ((arr_idx_range_pad[:-2] == 0)))[0]
			arr_idx_2 = np.where((arr_idx_range_pad[1:-1] != arr_idx_range_pad[2:]) & ((arr_idx_range_pad[2:] == 0)))[0]
		
			l_range = [(idx_1, idx_2+1) for idx_1, idx_2 in zip(arr_idx_1, arr_idx_2)]
			tpl_range = tuple(l_range)

			equation = Equation()
			equation.tpl_var = tpl_var
			equation.tpl_range = tpl_range

			if equation not in s_equation:
				s_equation.add(equation)

		tpl_equation = tuple(sorted(s_equation))
		amount_equation = len(tpl_equation)
		assert amount_equation >= 1

		for equation_nr, equation in enumerate(tpl_equation, 0):
			equation.equation_nr = equation_nr
		
		function = Function()
		function.tpl_equation = tpl_equation

		if function not in s_function:
			s_function.add(function)

	tpl_function = tuple(sorted(s_function))
	amount_function = len(tpl_function)
	assert amount_function >= 1

	for function_nr, function in enumerate(tpl_function, 0):
		function.function_nr = function_nr
		function.function_name = f"func_{function_nr}"


	# now parse the tpl_function part into a function string
	func_str = ""

	for function in tpl_function:
		func_part_str = ""
		func_part_str += f"def {function.function_name}():\n"
		tpl_equation = function.tpl_equation

		l_var_b = []
		for equation in tpl_equation:
			equation_nr = equation.equation_nr
			
			var_a = f"a_{equation_nr}"
			var_sum = " + ".join(equation.tpl_var)
			func_part_str += f"\t{var_a} = {var_sum}\n"
			
			var_b = f"b_{equation_nr}"
			l_var_b.append(var_b)
			l_var_logic_and_var_range = [f"(({var_a} >= {i1}) & ({var_a} < {i2}))" for i1, i2 in  equation.tpl_range]
			var_logic_or_var_range = " | ".join(l_var_logic_and_var_range)
			func_part_str += f"\t{var_b} = ({var_logic_or_var_range}).astype(np.uint8)\n\n"

		var_logic_or_var_b = ' | '.join(l_var_b)
		func_part_str += f"\treturn {var_logic_or_var_b}\n\n"

		func_str += func_part_str

	func_str += (
		"l_func = [\n" +
		"".join(f"\t{{'function_name': '{function.function_name}', 'func': {function.function_name}}},\n" for function in tpl_function) +
		"]"
	)

	return tpl_function, func_str


AMOUNT_PARTS_USING = 8

def calculate_new_df_stats_gol_fields(d_params):
	frame = d_params['frame']
	height = d_params['height']
	width = d_params['width']
	density = d_params['density']
	iter_round_max = d_params['iter_round_max']
	starting_bytearray = d_params['starting_bytearray']
	amount_gol_game = d_params['amount_gol_game']

	l_seed_field = get_random_seed_256_bits(starting_bytearray=starting_bytearray)
	field_2d = Field2d(frame=frame, height=height, width=width, l_seed=l_seed_field)

	# pix_plain = np.zeros((height, width), dtype=np.uint8)
	# pix_plain[:] = 128

	l_d_data_raw = []
	l_seed_func_str_body = get_random_seed_256_bits(starting_bytearray=starting_bytearray)
	l_seed_func_str_range = get_random_seed_256_bits(starting_bytearray=starting_bytearray)

	for gol_game_nr in range(0, amount_gol_game):
		increment_arr_uint8(arr=l_seed_field)
		increment_arr_uint8(arr=l_seed_func_str_body)
		increment_arr_uint8(arr=l_seed_func_str_range)

		# path_dir_gol_board = os.path.join(path_dir_game_of_life, f"gol_board_gol_game_nr_{gol_game_nr:03}")
		# mkdirs(path=path_dir_gol_board)
		
		# print(f"gol_game_nr: {gol_game_nr:3}")

		d_data_raw = {}
		d_data_raw["frame"] = frame
		d_data_raw["height"] = height
		d_data_raw["width"] = width

		l_d_data_raw.append(d_data_raw)

		field_2d.increment_arr_seeds()
		d_data_raw["arr_seed_func"] = field_2d.arr_seed_func.copy()
		d_data_raw["arr_seed_pix"] = field_2d.arr_seed_pix.copy()
		
		field_2d.reset_rnds()
		
		d_data_raw["l_seed_func_str_body"] = l_seed_func_str_body.copy()
		d_data_raw["l_seed_func_str_range"] = l_seed_func_str_range.copy()

		tpl_function, func_str = generate_tpl_function_func_str(
			field_2d=field_2d,
			l_seed_body=l_seed_func_str_body,
			l_seed_range=l_seed_func_str_range,
		)
		d_data_raw["tpl_function"] = tpl_function
		d_data_raw["func_str"] = func_str
		
		field_2d.define_functions(func_str=func_str)

		field_2d.set_points_random(density=density)
		l_arr_no_frame = field_2d.play_the_field(iter_round_max=iter_round_max)

		d_data_raw["l_arr_no_frame"] = l_arr_no_frame

		# pix_comb = plot_all_field_in_one_picture(l_arr=l_arr_no_frame, rows=8, columns=6, border_width=2)
		# Image.fromarray(pix_comb).save(os.path.join(path_dir_gol_board, f"example_gol_board.png"))

		# with open(os.path.join(path_dir_gol_board, "func_str.py"), "w") as f:
		# 	f.write(func_str)

		# with imageio.get_writer(os.path.join(path_dir_game_of_life, f'example_gif_gol_game_nr_{gol_game_nr:03}.gif'), mode='I', fps=20) as writer:
		# 	for _ in range(0, 5):
		# 		writer.append_data(l_arr_no_frame[0] * 255)
		# 	for arr_no_frame in l_arr_no_frame[1:80]:
		# 		writer.append_data(arr_no_frame * 255)
		# 	for _ in range(0, 5):
		# 		writer.append_data(pix_plain)

	l_column = [
		'frame',
		'height',
		'width',
		'arr_seed_func',
		'arr_seed_pix',
		'l_seed_func_str_body',
		'l_seed_func_str_range',
		'tpl_var',
		'tpl_range',
		'amount_points_total',
		'arr_amount_1_dot',
		'arr_amount_2_dot',
		'arr_amount_norm',
		'arr_amount_norm_diff_part',
	]
	d_data_stats = {column: [] for column in l_column}
	for d_data_raw in l_d_data_raw:
		d_data_stats['frame'].append(d_data_raw['frame'])
		d_data_stats['height'].append(d_data_raw['height'])
		d_data_stats['width'].append(d_data_raw['width'])

		d_data_stats['arr_seed_func'].append(d_data_raw['arr_seed_func'])
		d_data_stats['arr_seed_pix'].append(d_data_raw['arr_seed_pix'])
		d_data_stats['l_seed_func_str_body'].append(d_data_raw['l_seed_func_str_body'])
		d_data_stats['l_seed_func_str_range'].append(d_data_raw['l_seed_func_str_range'])
		
		tpl_function = d_data_raw['tpl_function']
		assert len(tpl_function) == 1
		tpl_equation = tpl_function[0].tpl_equation
		assert len(tpl_equation) == 1

		equation = tpl_equation[0]
		d_data_stats['tpl_var'].append(equation.tpl_var)
		d_data_stats['tpl_range'].append(equation.tpl_range)

		arr_arr_no_frame = np.array(d_data_raw['l_arr_no_frame'])

		amount_points_total = np.multiply.reduce(arr_arr_no_frame.shape[1:])
		d_data_stats['amount_points_total'].append(amount_points_total)
		
		arr_amount_1_dot = np.vstack((
			np.sum(np.sum(arr_arr_no_frame == 0, axis=2), axis=1),
			np.sum(np.sum(arr_arr_no_frame == 1, axis=2), axis=1),
		)).T

		arr_2x2_arr_arr_no_frame = np.empty((2, 2) + arr_arr_no_frame.shape, dtype=np.uint8)
		arr_2x2_arr_arr_no_frame[0, 0] = arr_arr_no_frame
		arr_2x2_arr_arr_no_frame[0, 1] = np.roll(arr_arr_no_frame, shift=-1, axis=2)
		arr_2x2_arr_arr_no_frame[1, 0] = np.roll(arr_arr_no_frame, shift=-1, axis=1)
		arr_2x2_arr_arr_no_frame[1, 1] = np.roll(arr_2x2_arr_arr_no_frame[0, 1], shift=-1, axis=1)

		# all possible 2 cell combinations
		l_cell_pos = [
			((0, 0), (0, 1)),
			((0, 0), (1, 0)),
			((0, 0), (1, 1)),
			((1, 0), (0, 1)),
		]

		# all possible 2 cell activation
		l_cell_activation = [(0, 0), (0, 1), (1, 0), (1, 1)]

		arr_amount_2_dot = np.vstack(tuple(
			np.sum(np.sum((arr_2x2_arr_arr_no_frame[c_y_1, c_x_1] == a_1) & (arr_2x2_arr_arr_no_frame[c_y_2, c_x_2] == a_2), axis=2), axis=1)
			for (c_y_1, c_x_1), (c_y_2, c_x_2) in l_cell_pos
			for a_1, a_2 in l_cell_activation
		)).T

		d_data_stats['arr_amount_1_dot'].append(arr_amount_1_dot)
		d_data_stats['arr_amount_2_dot'].append(arr_amount_2_dot)

		arr_amount_all = np.hstack((arr_amount_1_dot, arr_amount_2_dot))
		# arr_amount_norm = np.flip(arr_amount_all / amount_points_total, axis=0)
		# arr_amount_norm_diff_part = np.abs(arr_amount_norm[0:AMOUNT_PARTS_USING] - arr_amount_norm[1:AMOUNT_PARTS_USING+1])
		arr_amount_norm = arr_amount_all / amount_points_total
		arr_amount_norm_diff_part = np.abs(arr_amount_norm[0:AMOUNT_PARTS_USING] - arr_amount_norm[1:AMOUNT_PARTS_USING+1])

		d_data_stats['arr_amount_norm'].append(arr_amount_norm)
		d_data_stats['arr_amount_norm_diff_part'].append(arr_amount_norm_diff_part)

	df_stats = pd.DataFrame(data=d_data_stats, columns=l_column, dtype=object)

	return df_stats


if __name__ == '__main__':
	path_dir_game_of_life = os.path.join(TEMP_DIR, "game_of_life")
	if not os.path.exists(path_dir_game_of_life):
		mkdirs(path=path_dir_game_of_life)

	# TODO: move this into a function too...
	d_params_base = dict(
		frame=4,
		height=64,
		width=64,
		density=0.56,
		iter_round_max=300,
		amount_gol_game=20,
	)

	# # only for a single useage!
	# d_params = d_params_base | dict(starting_bytearray=bytearray([0x00]))
	# df_stats = calculate_new_df_stats_gol_fields(d_params=d_params)


	mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

	mult_proc_mng.define_new_func('func_calculate_new_df_stats_gol_fields', calculate_new_df_stats_gol_fields)
	l_arguments = [(d_params_base | dict(starting_bytearray=bytearray([x])), ) for x in range(0, 100)]
	l_ret = mult_proc_mng.do_new_jobs(
		['func_calculate_new_df_stats_gol_fields']*len(l_arguments),
		l_arguments,
	)
	print("len(l_ret): {}".format(len(l_ret)))

	del mult_proc_mng

	# combine all the df_stats_parts from the l_ret variable
	df_stats = pd.concat(l_ret)


	# TODO: make this more configuable with a extern config file
	# combine the new df_stats with the previous one, which was saved previously, and save the new one again into a temporary file
	path_file_saved_df_stats = os.path.join(path_dir_game_of_life, 'df_stats_prev.pkl')

	if not os.path.exists(path_file_saved_df_stats):
		with open(path_file_saved_df_stats, 'wb') as f:
			pkl.dump(df_stats, f)
		df_stats_prev = df_stats
	else:
		with open(path_file_saved_df_stats, 'rb') as f:
			df_stats_prev_load = pkl.load(f)

		df_stats_prev = pd.concat((df_stats_prev_load, df_stats))

		with open(path_file_saved_df_stats, 'wb') as f:
			pkl.dump(df_stats_prev, f)


	arr_point_mat = np.array(df_stats_prev['arr_amount_norm_diff_part'].values.tolist())
	arr_point = arr_point_mat.reshape((-1, arr_point_mat.shape[1]*arr_point_mat.shape[2]))
	# df_point = pd.DataFrame(data=arr_point)
	# df_point_unique = df_point.drop_duplicates(subset=df_point.columns, keep='first')

	arr_point_tpl = np.core.records.fromarrays(arr_point.T, dtype=[(f'c{i}', 'f8') for i in range(0, arr_point.shape[1])])
	u_arr_point_tpl = np.unique(arr_point_tpl)
	arr_point_unique = u_arr_point_tpl.view((np.float64, len(u_arr_point_tpl.dtype.names)))

	iterations = 100

	min_amount_cluster = 2
	max_amount_cluster = 15

	# arr_arr_y_s_mean, arr_arr_y_s_mean_argsort, arr_best_cluster_amount = utils_cluster.find_best_fitting_cluster_amount_multiprocessing(
	arr_arr_y_s_mean, arr_best_cluster_amount = utils_cluster.find_best_fitting_cluster_amount_multiprocessing(
		max_try_nr=20,
		arr_point=arr_point_unique,
		min_amount_cluster=min_amount_cluster,
		max_amount_cluster=max_amount_cluster,
		iterations=iterations,
		amount_proc=mp.cpu_count(),
	)

	print(f"arr_point.shape: {arr_point.shape}")
	print(f"arr_point_unique.shape: {arr_point_unique.shape}")

	# TODO: find/explore the different possible characteristics for the different game of life fields/functions etc.

	plt.figure()

	plt.title("Cluster silhouette median values")

	plt.xlabel("Cluster amount")
	plt.ylabel("Silhouette value")

	arr_x_cluster_amount = np.arange(min_amount_cluster, max_amount_cluster+1)
	arr_y_silhoutte_median = np.median(arr_arr_y_s_mean, axis=0)
	plt.plot(arr_x_cluster_amount, arr_y_silhoutte_median, linestyle='', marker='o', label=f"Silhouette median per cluster amount")

	plt.legend()
