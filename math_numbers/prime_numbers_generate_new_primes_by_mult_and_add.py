#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.13 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import decimal
import dill
import gzip
import itertools
import math
import os
import pdb
import re
import sqlite3
import sys
import time
import traceback

import numpy as np # need installation from pip
import pandas as pd # need installation from pip
import multiprocessing as mp

import matplotlib.pyplot as plt # need installation from pip

from collections import defaultdict
from copy import deepcopy, copy
from dataclasses import dataclass
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
load_module_dynamically(**dict(var_glob=var_glob, name='prime_numbers_fun', path=os.path.join(PYTHON_PROGRAMS_DIR, "math_numbers/prime_numbers_fun.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_primes = prime_numbers_fun.get_primes
get_primes_list = prime_numbers_fun.get_primes_list
get_primes_list_part = prime_numbers_fun.get_primes_list_part


OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def main():
	d_arg_name_to_arg_val = {
		'prime_max': 11,
	}

	for arg in sys.argv[1:]:
		arg_name, arg_val_str = arg.split('=')
		
		if arg_name == 'prime_max':
			d_arg_name_to_arg_val[arg_name] = int(arg_val_str)

	prime_max = d_arg_name_to_arg_val['prime_max']

	# arr_prime = np.fromfile('/tmp/arr_primes.bytes', dtype=np.uint64, count=-1)
	# arr_prime = np.fromfile('/tmp/arr_primes.bytes', dtype=np.uint64, count=1_000_000)
	arr_prime = np.fromfile('/tmp/arr_primes.bytes', dtype=np.uint64, count=400_000_000)
	print(f'arr_prime.shape: {arr_prime.shape}')
	arr_prime = arr_prime.astype(np.int64)
	# arr_prime = arr_prime[:200000000].astype(np.int64)
	# arr_prime = arr_prime[:75000000].astype(np.int64)
	# arr_prime = arr_prime[:7500000].astype(np.int64)
	# globals().update(locals())
	# return


	# arr_prime = get_primes_list(n_max=1000000)
	# s_prime = set(arr_prime)

	max_num = int(math.sqrt(arr_prime[-1] / 2))

	for max_i, prime in enumerate(arr_prime, 1):
		if prime >= max_num:
			break

	arr_prime_small = arr_prime[:max_i-1]
	print(f'arr_prime_small.shape: {arr_prime_small.shape}')

	def calc_amount_to_diff_prime(arr_two_prime_mult):
		assert len(arr_two_prime_mult_part.shape) == 1
		assert np.all(np.sort(np.unique(arr_two_prime_mult)) == arr_two_prime_mult)

		i_prime_lower = 0
		prime_lower = arr_prime[i_prime_lower]
		i_prime_upper = 1
		prime_upper = arr_prime[i_prime_upper]

		# print('Starting:')
		# print(f'- i_prime_lower: {i_prime_lower}, i_prime_upper: {i_prime_upper}')
		# print(f'- prime_lower: {prime_lower}, prime_upper: {prime_upper}')

		d_diff_to_amount_to_diff_prime = {}
		for i, two_prime_mult in enumerate(arr_two_prime_mult, 1):
			if i % 1000000 == 0:
				print(f'i: {i}')
			
			# print(f'\ntwo_prime_mult: {two_prime_mult}')

			is_increasing = True
			while prime_lower > two_prime_mult or two_prime_mult > prime_upper:
				i_mult_next = 0
				i_mult = 1
				
				if prime_lower > two_prime_mult:
					while prime_lower > two_prime_mult:
						# print('prime_lower > two_prime_mult')

						i_prime_lower -= i_mult
						prime_lower = arr_prime[i_prime_lower]
						i_prime_upper -= i_mult
						prime_upper = arr_prime[i_prime_upper]

						# print(f'- i_prime_lower: {i_prime_lower}, i_prime_upper: {i_prime_upper}')
						# print(f'- prime_lower: {prime_lower}, prime_upper: {prime_upper}')
						# print(f'- i_mult_next: {i_mult_next}, i_mult: {i_mult}')
						# input()

						i_mult_next += 1
						if i_mult_next % 2 == 0:
							i_mult_next = 0
							i_mult *= 2
				elif two_prime_mult > prime_upper:
					while two_prime_mult > prime_upper:
						# print('two_prime_mult > prime_upper')

						i_prime_lower += i_mult
						prime_lower = arr_prime[i_prime_lower]
						i_prime_upper += i_mult
						prime_upper = arr_prime[i_prime_upper]

						# print(f'- i_prime_lower: {i_prime_lower}, i_prime_upper: {i_prime_upper}')
						# print(f'- prime_lower: {prime_lower}, prime_upper: {prime_upper}')
						# print(f'- i_mult_next: {i_mult_next}, i_mult: {i_mult}')
						# input()
						
						i_mult_next += 1
						if i_mult_next % 2 == 0:
							i_mult_next = 0
							i_mult *= 2

			diff_1 = prime_lower - two_prime_mult
			diff_2 = prime_upper - two_prime_mult

			if diff_1 not in d_diff_to_amount_to_diff_prime:
				d_diff_to_amount_to_diff_prime[diff_1] = 0
			if diff_2 not in d_diff_to_amount_to_diff_prime:
				d_diff_to_amount_to_diff_prime[diff_2] = 0

			d_diff_to_amount_to_diff_prime[diff_1] += 1
			d_diff_to_amount_to_diff_prime[diff_2] += 1

		return d_diff_to_amount_to_diff_prime

	d_prime_to_arr_two_prime_mult_part = {}
	d_prime_to_d_diff_to_amount_to_diff_prime = {}
	for prime_1 in arr_prime_small[:300]:
		print(f'prime_1: {prime_1}')
		arr_two_prime_mult_part = prime_1 * arr_prime_small
		d_prime_to_arr_two_prime_mult_part[prime_1] = arr_two_prime_mult_part
		d_prime_to_d_diff_to_amount_to_diff_prime[prime_1] = calc_amount_to_diff_prime(arr_two_prime_mult=arr_two_prime_mult_part)

	arr_all_primes = np.unique(list(d_prime_to_d_diff_to_amount_to_diff_prime.keys()))
	arr_all_diffs = np.unique([key for keys in d_prime_to_d_diff_to_amount_to_diff_prime.values() for key in keys])

	d_prime_to_index = {diff: i for i, diff in enumerate(arr_all_primes, 0)}
	d_diff_to_index = {diff: i for i, diff in enumerate(arr_all_diffs, 0)}

	arr_histogram = np.zeros((arr_all_primes.shape[0], arr_all_diffs.shape[0]), dtype=np.int64)
	for prime in arr_all_primes:
		for diff, amount_to_diff in d_prime_to_d_diff_to_amount_to_diff_prime[prime].items():
			arr_histogram[d_prime_to_index[prime], d_diff_to_index[diff]] = amount_to_diff

	min_val = np.min(arr_histogram)
	max_val = np.max(arr_histogram)

	rgb_f64_1 = np.array((0xFF, 0x00, 0x00), dtype=np.uint8).astype(np.float64)
	rgb_f64_2 = np.array((0x00, 0x00, 0xFF), dtype=np.uint8).astype(np.float64)

	arr_range = np.arange(0, 256).astype(np.float64)
	arr_1 = arr_range / arr_range[-1]
	arr_2 = arr_1[::-1]

	arr_rgb_f64_mean = rgb_f64_1 * arr_1.reshape((-1, 1)) + rgb_f64_2 * arr_2.reshape((-1, 1))
	arr_rgb_lut = arr_rgb_f64_mean.astype(np.uint8)

	arr_rgb_index = ((arr_histogram.astype(np.float64) + min_val) / max_val * (arr_rgb_lut.shape[0] - 1)).astype(np.int64)
	arr_rgb_histogram = arr_rgb_lut[arr_rgb_index]
	# create interpolation of colors

	img = Image.fromarray(arr_rgb_histogram)
	img = img.resize(size=(img.width * 5, img.height * 5), resample=Image.Resampling.NEAREST)
	img.save('/tmp/image.png')


	globals().update(locals())
	sys.exit()


	fig, ax = plt.subplots()
	im = ax.imshow(arr_histogram)

	# Show all ticks and label them with the respective list entries
	ax.set_xticks(range(len(arr_all_diffs)), labels=arr_all_diffs,
				  rotation=45, ha="right", rotation_mode="anchor")
	ax.set_yticks(range(len(arr_all_primes)), labels=arr_all_primes)

	ax.set_title("Per prime the amount of diffs to the nearest prime\nof each prime_1 * prime_2 number")
	fig.tight_layout()
	plt.show()

	globals().update(locals())
	sys.exit()
	
	arr_two_prime_mult = np.outer(arr_prime_small, arr_prime_small)

	arr_unq = np.sort(np.unique(arr_two_prime_mult.reshape((-1, ))))
	assert arr_unq[-1] < arr_prime[-1]

	# find the nearest prime differences from the two prime mult numbers
	i_prime_lower = 0
	prime_lower = arr_prime[i_prime_lower]
	i_prime_upper = 1
	prime_upper = arr_prime[i_prime_upper]

	d_diff_to_amount_to_diff_prime = {}
	d_diff_abs_smallest_to_amount_to_diff_prime_smallest_diff = {}
	d_diff_diff_to_amount_of_diff_diff = {}
	for i, two_prime_mult in enumerate(arr_unq, 1):
		if i % 1000000 == 0:
			print(f'i: {i}')

		while two_prime_mult > prime_upper:
			i_prime_lower += 1
			prime_lower = arr_prime[i_prime_lower]
			i_prime_upper += 1
			prime_upper = arr_prime[i_prime_upper]

		diff_1 = prime_lower - two_prime_mult
		diff_2 = prime_upper - two_prime_mult
		diff_1_abs = abs(diff_1)

		diff_diff = diff_2 - diff_1_abs

		if diff_1_abs < diff_2:
			diff_smallest = diff_1_abs
		else:
			diff_smallest = diff_2

		if diff_1 not in d_diff_to_amount_to_diff_prime:
			d_diff_to_amount_to_diff_prime[diff_1] = 0
		if diff_2 not in d_diff_to_amount_to_diff_prime:
			d_diff_to_amount_to_diff_prime[diff_2] = 0

		if diff_diff not in d_diff_diff_to_amount_of_diff_diff:
			d_diff_diff_to_amount_of_diff_diff[diff_diff] = 0

		if diff_smallest not in d_diff_abs_smallest_to_amount_to_diff_prime_smallest_diff:
			d_diff_abs_smallest_to_amount_to_diff_prime_smallest_diff[diff_smallest] = 0

		d_diff_to_amount_to_diff_prime[diff_1] += 1
		d_diff_to_amount_to_diff_prime[diff_2] += 1

		d_diff_diff_to_amount_of_diff_diff[diff_diff] += 1
		
		d_diff_abs_smallest_to_amount_to_diff_prime_smallest_diff[diff_smallest] += 1

	arr_x_diff_prime = np.sort(list(d_diff_to_amount_to_diff_prime.keys()))
	arr_y_diff_prime = np.array([d_diff_to_amount_to_diff_prime[diff] for diff in arr_x_diff_prime])

	arr_x_diff_diff = np.sort(list(d_diff_diff_to_amount_of_diff_diff.keys()))
	arr_y_diff_diff = np.array([d_diff_diff_to_amount_of_diff_diff[diff] for diff in arr_x_diff_diff])

	arr_x_diff_smallest = np.sort(list(d_diff_abs_smallest_to_amount_to_diff_prime_smallest_diff.keys()))
	arr_y_diff_smallest = np.array([d_diff_abs_smallest_to_amount_to_diff_prime_smallest_diff[diff] for diff in arr_x_diff_smallest])

	fig_1, ax_1 = plt.subplots()

	ax_1.plot(arr_x_diff_prime, arr_y_diff_prime, linestyle='', marker='o')

	ax_1.set_xlabel('Diffs from prime to two prime mult')
	ax_1.set_ylabel('Amount of diffs')
	ax_1.set_title('p1 * p2 +- diff to the neighbor primes')

	fig_2, ax_2 = plt.subplots()

	ax_2.plot(arr_x_diff_diff, arr_y_diff_diff, linestyle='', marker='o')

	ax_2.set_xlabel('diff_2 - abs(diff_1)')
	ax_2.set_ylabel('Amount of diffs diff')
	ax_2.set_title('Diffs of diff')

	fig_3, ax_3 = plt.subplots()

	ax_3.plot(arr_x_diff_smallest, arr_y_diff_smallest, linestyle='', marker='o')

	ax_3.set_xlabel('smallest(abs(diff_1), diff_2)')
	ax_3.set_ylabel('Amount of diffs smallest')
	ax_3.set_title('Diffs of smallest diff')

	plt.show()

	arr_argsort_diff_prime = np.argsort(arr_y_diff_prime)[::-1]
	arr_x_diff_prime_argsort = arr_x_diff_prime[arr_argsort_diff_prime]
	arr_y_diff_prime_argsort = arr_y_diff_prime[arr_argsort_diff_prime]

	arr_argsort_diff_diff = np.argsort(arr_y_diff_diff)[::-1]
	arr_x_diff_diff_argsort = arr_x_diff_diff[arr_argsort_diff_diff]
	arr_y_diff_diff_argsort = arr_y_diff_diff[arr_argsort_diff_diff]

	arr_argsort_diff_smallest = np.argsort(arr_y_diff_smallest)[::-1]
	arr_x_diff_smallest_argsort = arr_x_diff_smallest[arr_argsort_diff_smallest]
	arr_y_diff_smallest_argsort = arr_y_diff_smallest[arr_argsort_diff_smallest]

	arr_cumsum_procentage_not_prime = 1 - np.cumsum(arr_y_diff_prime_argsort) / np.sum(arr_y_diff_prime_argsort)
	arr_x_diff_prime_argsort_p_0_1 = arr_x_diff_prime_argsort[arr_cumsum_procentage_not_prime > 0.1]
	arr_x_diff_prime_argsort_p_0_01 = arr_x_diff_prime_argsort[arr_cumsum_procentage_not_prime > 0.01]
	arr_x_diff_prime_argsort_p_0_001 = arr_x_diff_prime_argsort[arr_cumsum_procentage_not_prime > 0.001]

	globals().update(locals())
	sys.exit()

	l_t_prime_1_prime_2_prime_mult = []
	for i, prime_1 in enumerate(arr_prime_small[:-1], 1):
		for prime_2 in arr_prime_small[i:]:
			l_t_prime_1_prime_2_prime_mult.append((prime_1, prime_2, prime_1 * prime_2))

	assert l_t_prime_1_prime_2_prime_mult[-1][2] < max(arr_prime)

	d_addition_to_l_prime_additional = {}
	# for additional in range(2, 401, 2):
	for additional in [2, 2*3, 2*3*5, 2*3*5*7, 2*3*5*7*11]:
		print(f"additional: {additional}")
		for prime_1, prime_2, prime_mult in l_t_prime_1_prime_2_prime_mult:
			prime_mult_add = prime_mult + additional

			if prime_mult_add in s_prime:
				if additional not in d_addition_to_l_prime_additional:
					d_addition_to_l_prime_additional[additional] = []

				d_addition_to_l_prime_additional[additional].append((prime_1, prime_2, prime_mult_add))

	l_t_addition_length = sorted([(addition, len(l_prime_additional)) for addition, l_prime_additional in d_addition_to_l_prime_additional.items()])

	l_addition, l_length = list(zip(*l_t_addition_length))

	fig, ax = plt.subplots()

	ax.bar([str(addition) for addition in l_addition], l_length)

	ax.set_xlabel('Additional factor to add')
	ax.set_ylabel('Amount of p1*p2+additional primes')
	ax.set_title('p1 * p2 + additional')

	plt.show()

	globals().update(locals())


if __name__ == "__main__":
	main()
