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

class HashSha256PRNG:
	def __init__(self, l_bytearr):
		self.hashing_algorithm = sha256
		self.l_state = [self.hashing_algorithm(bytearr).digest() for bytearr in l_bytearr]
		self.i_state = 0
		self.amount_state = len(self.l_state)
		self.l_t_i_state = [
			(i_state_prev, i_state_curr, i_state_next)
			for i_state_prev, i_state_curr, i_state_next
			in zip(
				[self.amount_state - 1] + list(range(0, self.amount_state - 1)),
				list(range(0, self.amount_state)),
				list(range(1, self.amount_state)) + [1],
			)
		]

		# do an update of
		# state_noew <- state_prev xor hash(state_now)
		# for enforcing an avalanche effect

		print('Before re-hashing:')
		self.print_state()

		# must do at least 2 full rounds to fully activate the avalanche effect
		for _ in range(0, 2):
			for i in range(0, self.amount_state):		
				self.calc_next_random_bytearray()

		print('After re-hashing:')
		self.print_state()


	def calc_next_random_bytearray(self):
		# calc next hash
		arr = np.frombuffer(self.l_state[self.i_state], dtype=np.uint8)

		i_state_prev, i_state_curr, i_state_next  = self.l_t_i_state[self.i_state]
		self.l_state[i_state_curr] = (
			(
				np.frombuffer(self.hashing_algorithm(self.l_state[i_state_prev]).digest(), dtype=np.uint8) ^
				np.frombuffer(self.l_state[i_state_curr], dtype=np.uint8)
			) +
			np.frombuffer(self.l_state[i_state_next], dtype=np.uint8)
		).tobytes()

		self.i_state += 1
		if self.i_state >= self.amount_state:
			self.i_state = 0

		return arr


	def print_state(self):
		print('Print l_state of prng:')
		for i in range(0, self.amount_state):
			print(f'- self.l_state[{i}]: {" ".join([f"{v:02X}" for v in list(self.l_state[i])])}')


	def generate_random_u8_values(self, length_u8):
		assert length_u8 > 0

		arr = np.empty((length_u8, ), dtype=np.uint8)

		full_rounds = length_u8 // 32
		rest_bytes = length_u8 % 32

		for i in range(0, full_rounds):
			arr[32*i:32*(i+1)] = self.calc_next_random_bytearray()

		if rest_bytes > 0:
			arr[32*full_rounds:] = self.calc_next_random_bytearray()[:rest_bytes]

		return arr


	def generate_random_u32_values(self, length_u32):
		assert length_u32 > 0

		arr_u8 = self.generate_random_u8_values(length_u8=length_u32*4)

		return arr_u8.view(np.uint32)


	def generate_random_u64_values(self, length_u64):
		assert length_u64 > 0

		arr_u8 = self.generate_random_u8_values(length_u8=length_u64*8)

		return arr_u8.view(np.uint64)


	def generate_random_u8_values_modulo(self, length_u8, modulo):
		assert length_u8 > 0
		assert modulo > 0x00 and modulo <= 0xFF

		arr = self.generate_random_u8_values(length_u8=length_u8)
		arr %= modulo

		return arr


	def generate_shuffle_index_array(self, amount_elements):
		arr_random = self.generate_random_u64_values(length_u64=amount_elements)
		arr_index = np.argsort(arr_random)
		return arr_index


def general_modulo_polynome_function(modulo, t_factor, l_t_power, t_x):
	mod_sum_all = 0
	
	for factor, t_power in zip(t_factor, l_t_power):
		mod_mult_all = factor
		for power, x in zip(t_power, t_x):
			mod_mult_all = (mod_mult_all * x**power) % modulo
		mod_sum_all = (mod_sum_all + mod_mult_all) % modulo

	return mod_sum_all


def generate_test_suite(modulo, amount_factors):
	l_t_power_all = list(itertools.product(*[list(range(0, modulo))]*amount_factors))
	l_t_x_all = list(itertools.product(*[list(range(1, modulo))]*amount_factors))
	l_factor = list(range(1, modulo))

	d_all_values = {
		'modulo': modulo,
		'amount_factors': amount_factors,
		'l_t_power_all': l_t_power_all,
		'l_t_x_all': l_t_x_all,
		'l_factor': l_factor,
	}

	# total combianations: (modulo - 1)**(1 + amount_factors) * modulo**amount_factors
	# e.g. modulo = 8 and amount_factors = 2, amount of total combinations are 21952

	return d_all_values


def get_seconds_from_datetime_diff(dt_diff):
	return dt_diff.days * 86_400 + dt_diff.seconds + dt_diff.microseconds / 1_000_000


def calc_simple_statistics(l_value):
	return {
		'min': np.min(l_value),
		'q1': np.quantile(l_value, 0.25),
		'median': np.median(l_value),
		'q3': np.quantile(l_value, 0.75),
		'max': np.max(l_value),
		'mean': np.mean(l_value),
		'std': np.std(l_value),
	}


def benchmark_different_approaches_of_calculating_next_values_for_amount_factors_2(prng, modulo=8, amount_rounds=10, multiply_test_suite=10):
	print('Called: benchmark_different_approaches_of_calculating_next_values_for_amount_factors_2')
	print(f'- modulo: {modulo}')
	print(f'- amount_rounds: {amount_rounds}')
	print(f'- multiply_test_suite: {multiply_test_suite}')

	amount_factors = 2
	d_all_values = generate_test_suite(modulo=modulo, amount_factors=amount_factors)
	# TODO 2025.12.18T09:14 : create a lookup-table as a pre-calc dict
	# TODO: do a benchmark between dict and list
	# TODO: create a SHA256 based PRNG to be used in C/C++ too

	l_t_all = []
	l_t_one_all = []
	l_mod_num_all = []
	d_t_to_mod_num = {}
	d_t_one_to_mod_num = {}
	arr_lut_multiarray = np.empty((modulo-1, modulo-1, modulo-1, modulo, modulo), dtype=np.uint8)
	for factor in d_all_values['l_factor']:
		for t_x in d_all_values['l_t_x_all']:
			for t_power in d_all_values['l_t_power_all']:
				t = (factor, t_x, t_power)
				t_one = (factor, ) + t_x +t_power
				l_t_all.append(t)
				l_t_one_all.append(t_one)

				assert not t in d_t_to_mod_num
				assert not t_one in d_t_one_to_mod_num
				mod_num = (factor * t_x[0]**t_power[0] * t_x[1]**t_power[1]) % modulo
				l_mod_num_all.append(mod_num)
				d_t_to_mod_num[t] = mod_num
				d_t_one_to_mod_num[t_one] = mod_num
				arr_lut_multiarray[factor-1, t_x[0]-1, t_power[0]-1, t_x[1], t_power[1]] = mod_num

	print(f'len(d_t_to_mod_num): {len(d_t_to_mod_num)}')

	l_t_many = l_t_all*multiply_test_suite
	print(f'len(l_t_many): {len(l_t_many)}')

	l_test_u8_calc_next_val_time = []
	l_test_u8_use_lut_as_dict_time = []
	l_test_u8_use_lut_t_one_as_dict_time = []
	l_test_u8_use_lut_as_list_time = []
	l_test_u8_use_lut_in_one_as_list_time = []
	l_test_u8_use_lut_as_multiarray_time = []

	arr_test_u8_calc_next_val = np.empty((len(l_t_many), ), dtype=np.uint8)
	arr_test_u8_use_lut_as_dict = np.empty((len(l_t_many), ), dtype=np.uint8)
	arr_test_u8_use_lut_t_one_as_dict = np.empty((len(l_t_many), ), dtype=np.uint8)
	arr_test_u8_use_lut_as_list = np.empty((len(l_t_many), ), dtype=np.uint8)
	arr_test_u8_use_lut_in_one_as_list = np.empty((len(l_t_many), ), dtype=np.uint8)
	arr_test_u8_use_lut_as_multiarray = np.empty((len(l_t_many), ), dtype=np.uint8)

	for i_round in range(0, amount_rounds):
		print(f'\ni_round: {i_round}')
		arr_index = prng.generate_shuffle_index_array(amount_elements=len(l_t_many))
		l_t_mix = [l_t_many[i] for i in arr_index]
		l_t_mix_in_one = [(factor, )+t_x+t_power for factor, t_x, t_power in l_t_mix]

		dt_test_u8_calc_next_val_start = datetime.datetime.now()
		for i, (factor, t_x, t_power) in enumerate(l_t_mix, 0):
			arr_test_u8_calc_next_val[i] = (factor * t_x[0]**t_power[0] * t_x[1]**t_power[1]) % modulo
		dt_test_u8_calc_next_val_end = datetime.datetime.now()

		dt_test_u8_use_lut_as_dict_start = datetime.datetime.now()
		for i, t in enumerate(l_t_mix, 0):
			arr_test_u8_use_lut_as_dict[i] = d_t_to_mod_num[t]
		dt_test_u8_use_lut_as_dict_end = datetime.datetime.now()

		dt_test_u8_use_lut_t_one_as_dict_start = datetime.datetime.now()
		for i, t in enumerate(l_t_mix_in_one, 0):
			arr_test_u8_use_lut_t_one_as_dict[i] = d_t_one_to_mod_num[t]
		dt_test_u8_use_lut_t_one_as_dict_end = datetime.datetime.now()

		dt_test_u8_use_lut_as_list_start = datetime.datetime.now()
		for i, (factor, t_x, t_power) in enumerate(l_t_mix, 0):
			mult_1 = modulo
			mult_2 = mult_1 * modulo
			mult_3 = mult_2 * (modulo - 1)
			mult_4 = mult_3 * (modulo - 1)
			index = (factor-1)*mult_4 + (t_x[0]-1)*mult_3 + (t_x[1]-1)*mult_2 + t_power[0]*mult_1 + t_power[1]
			arr_test_u8_use_lut_as_list[i] = l_mod_num_all[index]
		dt_test_u8_use_lut_as_list_end = datetime.datetime.now()

		dt_test_u8_use_lut_in_one_as_list_start = datetime.datetime.now()
		for i, (f_0, f_1, f_2, f_3, f_4) in enumerate(l_t_mix_in_one, 0):
			mult_1 = modulo
			mult_2 = mult_1 * modulo
			mult_3 = mult_2 * (modulo - 1)
			mult_4 = mult_3 * (modulo - 1)
			index = (f_0-1)*mult_4 + (f_1-1)*mult_3 + (f_2-1)*mult_2 + f_3*mult_1 + f_4
			arr_test_u8_use_lut_in_one_as_list[i] = l_mod_num_all[index]
		dt_test_u8_use_lut_in_one_as_list_end = datetime.datetime.now()

		dt_test_u8_use_lut_as_multiarray_start = datetime.datetime.now()
		for i, (f_0, f_1, f_2, f_3, f_4) in enumerate(l_t_mix_in_one, 0):
			arr_test_u8_use_lut_as_multiarray[i] = arr_lut_multiarray[f_0-1, f_1-1, f_2-1, f_3, f_4]
		dt_test_u8_use_lut_as_multiarray_end = datetime.datetime.now()

		np.all(arr_test_u8_calc_next_val == arr_test_u8_use_lut_as_dict)
		np.all(arr_test_u8_calc_next_val == arr_test_u8_use_lut_t_one_as_dict)
		np.all(arr_test_u8_calc_next_val == arr_test_u8_use_lut_as_list)
		np.all(arr_test_u8_calc_next_val == arr_test_u8_use_lut_in_one_as_list)
		np.all(arr_test_u8_calc_next_val == arr_test_u8_use_lut_as_multiarray)

		l_test_u8_calc_next_val_time.append(get_seconds_from_datetime_diff(dt_test_u8_calc_next_val_end - dt_test_u8_calc_next_val_start))
		l_test_u8_use_lut_as_dict_time.append(get_seconds_from_datetime_diff(dt_test_u8_use_lut_as_dict_end - dt_test_u8_use_lut_as_dict_start))
		l_test_u8_use_lut_t_one_as_dict_time.append(get_seconds_from_datetime_diff(dt_test_u8_use_lut_t_one_as_dict_end - dt_test_u8_use_lut_t_one_as_dict_start))
		l_test_u8_use_lut_as_list_time.append(get_seconds_from_datetime_diff(dt_test_u8_use_lut_as_list_end - dt_test_u8_use_lut_as_list_start))
		l_test_u8_use_lut_in_one_as_list_time.append(get_seconds_from_datetime_diff(dt_test_u8_use_lut_in_one_as_list_end - dt_test_u8_use_lut_in_one_as_list_start))
		l_test_u8_use_lut_as_multiarray_time.append(get_seconds_from_datetime_diff(dt_test_u8_use_lut_as_multiarray_end - dt_test_u8_use_lut_as_multiarray_start))

		print('Current stats:')
		d_stats_test_u8_calc_next_val = calc_simple_statistics(l_test_u8_calc_next_val_time)
		d_stats_test_u8_use_lut_as_dict = calc_simple_statistics(l_test_u8_use_lut_as_dict_time)
		d_stats_test_u8_use_lut_t_one_as_dict = calc_simple_statistics(l_test_u8_use_lut_t_one_as_dict_time)
		d_stats_test_u8_use_lut_as_list = calc_simple_statistics(l_test_u8_use_lut_as_list_time)
		d_stats_test_u8_use_lut_in_one_as_list = calc_simple_statistics(l_test_u8_use_lut_in_one_as_list_time)
		d_stats_test_u8_use_lut_as_multiarray = calc_simple_statistics(l_test_u8_use_lut_as_multiarray_time)
		print(f'- d_stats_test_u8_calc_next_val["mean"]: {d_stats_test_u8_calc_next_val["mean"]:.6f}')
		print(f'- d_stats_test_u8_use_lut_as_dict["mean"]: {d_stats_test_u8_use_lut_as_dict["mean"]:.6f}')
		print(f'- d_stats_test_u8_use_lut_t_one_as_dict["mean"]: {d_stats_test_u8_use_lut_t_one_as_dict["mean"]:.6f}')
		print(f'- d_stats_test_u8_use_lut_as_list["mean"]: {d_stats_test_u8_use_lut_as_list["mean"]:.6f}')
		print(f'- d_stats_test_u8_use_lut_in_one_as_list["mean"]: {d_stats_test_u8_use_lut_in_one_as_list["mean"]:.6f}')
		print(f'- d_stats_test_u8_use_lut_as_multiarray["mean"]: {d_stats_test_u8_use_lut_as_multiarray["mean"]:.6f}')

	print(f'l_test_u8_calc_next_val_time: {l_test_u8_calc_next_val_time}')
	print(f'l_test_u8_use_lut_as_dict_time: {l_test_u8_use_lut_as_dict_time}')
	print(f'l_test_u8_use_lut_t_one_as_dict_time: {l_test_u8_use_lut_t_one_as_dict_time}')
	print(f'l_test_u8_use_lut_as_list_time: {l_test_u8_use_lut_as_list_time}')
	print(f'l_test_u8_use_lut_in_one_as_list_time: {l_test_u8_use_lut_in_one_as_list_time}')
	print(f'l_test_u8_use_lut_as_multiarray_time: {l_test_u8_use_lut_as_multiarray_time}')

	print('')
	d_stats_test_u8_calc_next_val = calc_simple_statistics(l_test_u8_calc_next_val_time)
	d_stats_test_u8_use_lut_as_dict = calc_simple_statistics(l_test_u8_use_lut_as_dict_time)
	d_stats_test_u8_use_lut_t_one_as_dict = calc_simple_statistics(l_test_u8_use_lut_t_one_as_dict_time)
	d_stats_test_u8_use_lut_as_list = calc_simple_statistics(l_test_u8_use_lut_as_list_time)
	d_stats_test_u8_use_lut_in_one_as_list = calc_simple_statistics(l_test_u8_use_lut_in_one_as_list_time)
	d_stats_test_u8_use_lut_as_multiarray = calc_simple_statistics(l_test_u8_use_lut_as_multiarray_time)
	print(f'd_stats_test_u8_calc_next_val["mean"]: {d_stats_test_u8_calc_next_val["mean"]:.6f}')
	print(f'd_stats_test_u8_use_lut_as_dict["mean"]: {d_stats_test_u8_use_lut_as_dict["mean"]:.6f}')
	print(f'd_stats_test_u8_use_lut_t_one_as_dict["mean"]: {d_stats_test_u8_use_lut_t_one_as_dict["mean"]:.6f}')
	print(f'd_stats_test_u8_use_lut_as_list["mean"]: {d_stats_test_u8_use_lut_as_list["mean"]:.6f}')
	print(f'd_stats_test_u8_use_lut_in_one_as_list["mean"]: {d_stats_test_u8_use_lut_in_one_as_list["mean"]:.6f}')
	print(f'd_stats_test_u8_use_lut_as_multiarray["mean"]: {d_stats_test_u8_use_lut_as_multiarray["mean"]:.6f}')

	d = {
		'modulo': modulo,
		"amount_factors": amount_factors,
		'd_stats_test_u8_calc_next_val': d_stats_test_u8_calc_next_val,
		'd_stats_test_u8_use_lut_as_dict': d_stats_test_u8_use_lut_as_dict,
		'd_stats_test_u8_use_lut_t_one_as_dict': d_stats_test_u8_use_lut_t_one_as_dict,
		'd_stats_test_u8_use_lut_as_list': d_stats_test_u8_use_lut_as_list,
		'd_stats_test_u8_use_lut_in_one_as_list': d_stats_test_u8_use_lut_in_one_as_list,
		'd_stats_test_u8_use_lut_as_multiarray': d_stats_test_u8_use_lut_as_multiarray,
	}



if __name__ == '__main__':
	prng = HashSha256PRNG(l_bytearr=[b'']*20)

	# d_mod_8 = benchmark_different_approaches_of_calculating_next_values_for_amount_factors_2(
	# 	prng=prng, modulo=8, amount_rounds=10, multiply_test_suite=200,
	# )

	d_mod_10 = benchmark_different_approaches_of_calculating_next_values_for_amount_factors_2(
		prng=prng, modulo=10, amount_rounds=10, multiply_test_suite=50,
	)

	# d_mod_15 = benchmark_different_approaches_of_calculating_next_values_for_amount_factors_2(
	# 	prng=prng, modulo=15, amount_rounds=10, multiply_test_suite=10,
	# )

	# arr_calculated_mod_u8 = np.empty((len(l_t_many), ), dtype=np.uint8)
	# arr_calculated_mod_u16 = np.empty((len(l_t_many), ), dtype=np.uint16)
	# arr_calculated_mod_u32 = np.empty((len(l_t_many), ), dtype=np.uint32)
	# arr_calculated_mod_u64 = np.empty((len(l_t_many), ), dtype=np.uint64)

	# l_calculated_mod_u8_time = []
	# l_calculated_mod_u16_time = []
	# l_calculated_mod_u32_time = []
	# l_calculated_mod_u64_time = []

	# for i_round in range(0, 30):
	# 	print(f'i_round: {i_round}')
	# 	arr_index = prng.generate_shuffle_index_array(amount_elements=len(l_t_many))
	# 	l_t_mix = [l_t_many[i] for i in arr_index]

	# 	dt_test_u8_start = datetime.datetime.now()
	# 	for i, (factor, t_x, t_power) in enumerate(l_t_mix, 0):
	# 		arr_calculated_mod_u8[i] = (factor * t_x[0]**t_power[0] * t_x[1]**t_power[1]) % modulo
	# 	dt_test_u8_end = datetime.datetime.now()

	# 	dt_test_u16_start = datetime.datetime.now()
	# 	for i, (factor, t_x, t_power) in enumerate(l_t_mix, 0):
	# 		arr_calculated_mod_u16[i] = (factor * t_x[0]**t_power[0] * t_x[1]**t_power[1]) % modulo
	# 	dt_test_u16_end = datetime.datetime.now()

	# 	dt_test_u32_start = datetime.datetime.now()
	# 	for i, (factor, t_x, t_power) in enumerate(l_t_mix, 0):
	# 		arr_calculated_mod_u32[i] = (factor * t_x[0]**t_power[0] * t_x[1]**t_power[1]) % modulo
	# 	dt_test_u32_end = datetime.datetime.now()

	# 	dt_test_u64_start = datetime.datetime.now()
	# 	for i, (factor, t_x, t_power) in enumerate(l_t_mix, 0):
	# 		arr_calculated_mod_u64[i] = (factor * t_x[0]**t_power[0] * t_x[1]**t_power[1]) % modulo
	# 	dt_test_u64_end = datetime.datetime.now()

	# 	l_calculated_mod_u8_time.append(get_seconds_from_datetime_diff(dt_test_u8_end - dt_test_u8_start))
	# 	l_calculated_mod_u16_time.append(get_seconds_from_datetime_diff(dt_test_u16_end - dt_test_u16_start))
	# 	l_calculated_mod_u32_time.append(get_seconds_from_datetime_diff(dt_test_u32_end - dt_test_u32_start))
	# 	l_calculated_mod_u64_time.append(get_seconds_from_datetime_diff(dt_test_u64_end - dt_test_u64_start))

	# print(f'l_calculated_mod_u8_time: {l_calculated_mod_u8_time}')
	# print(f'l_calculated_mod_u16_time: {l_calculated_mod_u16_time}')
	# print(f'l_calculated_mod_u32_time: {l_calculated_mod_u32_time}')
	# print(f'l_calculated_mod_u64_time: {l_calculated_mod_u64_time}')
	# print('')
	# d_stats_u8 = calc_simple_statistics(l_calculated_mod_u8_time)
	# d_stats_u16 = calc_simple_statistics(l_calculated_mod_u16_time)
	# d_stats_u32 = calc_simple_statistics(l_calculated_mod_u32_time)
	# d_stats_u64 = calc_simple_statistics(l_calculated_mod_u64_time)
	# print(f'Taken time for the u8 test: {d_stats_u8["mean"]}')
	# print(f'Taken time for the u16 test: {d_stats_u16["mean"]}')
	# print(f'Taken time for the u32 test: {d_stats_u32["mean"]}')
	# print(f'Taken time for the u64 test: {d_stats_u64["mean"]}')
