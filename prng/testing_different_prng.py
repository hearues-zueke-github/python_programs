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
import warnings
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

warnings.filterwarnings('ignore')

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
load_module_dynamically(**dict(var_glob=var_glob, name='prime_numbers_fun', path=os.path.join(PYTHON_PROGRAMS_DIR, "math_numbers/prime_numbers_fun.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

save_pkl_obj = utils_serialization.save_pkl_obj
load_pkl_obj = utils_serialization.load_pkl_obj

get_primes = prime_numbers_fun.get_primes

def get_unique_count(arr):
	return np.unique(arr, return_counts=True)


class SimpleLinearPRNG():
	def __init__(self, m, a, b, x=0):
		self.m = m
		self.a = a % self.m
		self.b = b % self.m
		self.x = x % self.m

	def calc_next(self):
		self.x = (self.a * self.x + self.b) % self.m
		return self.x


class RandomNumberDevice():

	def __init__(self, arr_seed_uint8, length_uint8=128):
		assert isinstance(arr_seed_uint8, np.ndarray)
		assert len(arr_seed_uint8.shape) == 1
		assert arr_seed_uint8.dtype == np.uint8(0).dtype

		self.length_uint8 = length_uint8
		self.block_size = 32
		assert self.length_uint8 % self.block_size == 0
		self.amount_block = self.length_uint8 // self.block_size

		self.vector_constant = np.arange(1, self.block_size + 1, dtype=np.uint8)

		self.mask_uint64_float64 = np.uint64(0x1fffffffffffff)

		self.arr_seed_uint8 = arr_seed_uint8.copy()

		self.length_values_uint8 = self.length_uint8
		self.length_values_uint64 = self.length_values_uint8 // 8
		self.idx_values_mult_uint64 = 0
		self.idx_values_xor_uint64 = 0
		self.idx_values_xor_uint64 = 0

		self.arr_mult_x = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_mult_a = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_mult_b = np.zeros((self.length_values_uint64, ), dtype=np.uint64)

		self.arr_xor_x = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_xor_a = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_xor_b = np.zeros((self.length_values_uint64, ), dtype=np.uint64)

		self.min_val_float64 = np.float64(2)**-53

		self.init_state()
		# self.init_prng_values()


	def init_state(self):
		self.arr_state_uint8 = np.zeros((self.length_uint8, ), dtype=np.uint8)

		self.arr_state_uint64 = self.arr_state_uint8.view(np.uint64)

		length = self.arr_seed_uint8.shape[0]
		i = 0
		while i < length - self.length_uint8:
			self.arr_state_uint8[:] ^= self.arr_seed_uint8[i:i+self.length_uint8]
			i += self.length_uint8

		if i == 0:
			self.arr_state_uint8[:length] ^= self.arr_seed_uint8
		elif i % self.length_uint8 != 0:
			self.arr_state_uint8[:i%self.length_uint8] ^= self.arr_seed_uint8[i:]


		# print(f"i: {0}, arr_state_uint8:")
		# for j in range(0, self.amount_block):
		# 	s = ''.join(map(lambda x: f'{x:02X}', self.arr_state_uint8[self.block_size*(j + 0):self.block_size*(j + 1)]))
		# 	print(f"- j: {j:2}, s: {s}")
		
		self.next_hashing_state()
		self.arr_mult_x[:] = self.arr_state_uint64
		self.next_hashing_state()
		self.arr_mult_a[:] = self.arr_state_uint64
		self.next_hashing_state()
		self.arr_mult_b[:] = self.arr_state_uint64
		
		self.next_hashing_state()
		self.arr_xor_x[:] = self.arr_state_uint64
		self.next_hashing_state()
		self.arr_xor_a[:] = self.arr_state_uint64
		self.next_hashing_state()
		self.arr_xor_b[:] = self.arr_state_uint64

		self.arr_mult_a[:] = 1 + self.arr_mult_a - (self.arr_mult_a % 4)
		self.arr_mult_b[:] = 1 + self.arr_mult_b - (self.arr_mult_b % 2)

		self.arr_xor_a[:] = 0 + self.arr_xor_a - (self.arr_xor_a % 2)
		self.arr_xor_b[:] = 1 + self.arr_xor_b - (self.arr_xor_b % 2)

		print(f"i: {i}, arr_state_uint8:")
		for j in range(0, self.amount_block):
			s = ''.join(map(lambda x: f'{x:02X}', self.arr_state_uint8[self.block_size*(j + 0):self.block_size*(j + 1)]))
			print(f"- j: {j:2}, s: {s}")


	def next_hashing_state(self):
		for i in range(0, self.amount_block):
			idx_blk_0 = (i + 0) % self.amount_block
			idx_blk_1 = (i + 1) % self.amount_block

			idx_0_0 = self.block_size * (idx_blk_0 + 0)
			idx_0_1 = self.block_size * (idx_blk_0 + 1)
			idx_1_0 = self.block_size * (idx_blk_1 + 0)
			idx_1_1 = self.block_size * (idx_blk_1 + 1)
			arr_part_0 = self.arr_state_uint8[idx_0_0:idx_0_1]
			arr_part_1 = self.arr_state_uint8[idx_1_0:idx_1_1]

			if np.all(arr_part_0 == arr_part_1):
				arr_part_1 ^= self.vector_constant

			arr_hash_0 = np.array(list(sha256(arr_part_0.data).digest()), dtype=np.uint8)
			arr_hash_1 = np.array(list(sha256(arr_part_1.data).digest()), dtype=np.uint8)
			self.arr_state_uint8[idx_1_0:idx_1_1] ^= arr_hash_0 ^ arr_hash_1 ^ arr_part_0


	# def init_prng_values(self):
	# 	self.arr_x[:] = self.arr_state_uint64[self.length_values_uint64*0:self.length_values_uint64*1]
	# 	self.arr_a[:] = self.arr_state_uint64[self.length_values_uint64*1:self.length_values_uint64*2]
	# 	self.arr_b[:] = self.arr_state_uint64[self.length_values_uint64*2:self.length_values_uint64*3]

	# 	self.arr_a[:] = 1 + self.arr_a - (self.arr_a % 4)
	# 	self.arr_b[:] = 1 + self.arr_b - (self.arr_b % 2)


	def calc_next_uint64(self, amount):
		arr = np.empty((amount, ), dtype=np.uint64)

		x_xor = self.arr_xor_x[self.idx_values_xor_uint64]
		for i in range(0, amount):
			x_mult = self.arr_mult_x[self.idx_values_mult_uint64]
			a_mult = self.arr_mult_a[self.idx_values_mult_uint64]
			b_mult = self.arr_mult_b[self.idx_values_mult_uint64]
			x_mult_new = ((a_mult * x_mult) + b_mult) ^ x_xor
			self.arr_mult_x[self.idx_values_mult_uint64] = x_mult_new
			arr[i] = x_mult_new

			self.idx_values_mult_uint64 += 1
			if self.idx_values_mult_uint64 >= self.length_values_uint64:
				self.idx_values_mult_uint64 = 0

				a_xor = self.arr_xor_a[self.idx_values_xor_uint64]
				b_xor = self.arr_xor_b[self.idx_values_xor_uint64]
				self.arr_xor_x[self.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

				self.idx_values_xor_uint64 += 1
				if self.idx_values_xor_uint64 >= self.length_values_uint64:
					self.idx_values_xor_uint64 = 0

				x_xor = self.arr_xor_x[self.idx_values_xor_uint64]

		return arr


	def calc_next_float64(self, amount):
		arr = self.calc_next_uint64(amount=amount)

		return self.min_val_float64 * (arr & self.mask_uint64_float64).astype(np.float64)


if __name__ == '__main__':
	print("Hello World!")

	amount = 10**2
	# amount = 10**5

	arr_seed_uint8 = np.array([0x00], dtype=np.uint8)

	t_1 = time.time()
	rnd_own = RandomNumberDevice(arr_seed_uint8=arr_seed_uint8, length_uint8=128)
	print(f"rnd_own.arr_mult_x: {rnd_own.arr_mult_x.tolist()}")
	print(f"rnd_own.arr_mult_a: {rnd_own.arr_mult_a.tolist()}")
	print(f"rnd_own.arr_mult_b: {rnd_own.arr_mult_b.tolist()}")
	print(f"rnd_own.arr_xor_x: {rnd_own.arr_xor_x.tolist()}")
	print(f"rnd_own.arr_xor_a: {rnd_own.arr_xor_a.tolist()}")
	print(f"rnd_own.arr_xor_b: {rnd_own.arr_xor_b.tolist()}")
	t_2 = time.time()
	arr = rnd_own.calc_next_uint64(amount=amount)
	print(f"arr: {arr.tolist()}")
	t_3 = time.time()
	arr_float64 = rnd_own.calc_next_float64(amount=amount)
	print(f"arr_float64: {arr_float64.tolist()}")
	t_4 = time.time()

	t_diff_1 = t_2 - t_1
	t_diff_2 = t_3 - t_2
	t_diff_3 = t_4 - t_3

	print(f"t_diff_1: {t_diff_1}")
	print(f"t_diff_2: {t_diff_2}")
	print(f"t_diff_3: {t_diff_3}")

	print(f"np.min(arr_float64): {np.min(arr_float64)}, np.max(arr_float64): {np.max(arr_float64)}")

	def calc_simple_stats(arr):
		val_min = np.min(arr)
		val_max = np.max(arr)
		val_median = np.median(arr)
		val_mean = np.mean(arr)
		val_std = np.std(arr)
		return dict(
			min=val_min,
			max=val_max,
			median=val_median,
			mean=val_mean,
			std=val_std,
		)

	t2_1 = time.time()
	rnd_np_1 = Generator(bit_generator=PCG64(seed=arr_seed_uint8))
	t2_2 = time.time()
	arr_np_1_uint64 = rnd_np_1.integers(0, 2**64, (amount, ), dtype=np.uint64)
	t2_3 = time.time()
	arr_np_1_float64 = rnd_np_1.random((amount, ))
	t2_4 = time.time()

	t_diff_1 = t2_2 - t2_1
	t_diff_2 = t2_3 - t2_2
	t_diff_3 = t2_4 - t2_3

	print(f"t_diff_1: {t_diff_1}")
	print(f"t_diff_2: {t_diff_2}")
	print(f"t_diff_3: {t_diff_3}")

	d_own_float64 = calc_simple_stats(arr=arr_float64)
	d_np_1_float64 = calc_simple_stats(arr=arr_np_1_float64)

	arr_float64_lower = arr_float64[arr_float64 < 0.5]
	arr_float64_higher = arr_float64[arr_float64 >= 0.5]
	arr_np_1_float64_lower = arr_np_1_float64[arr_np_1_float64 < 0.5]
	arr_np_1_float64_higher = arr_np_1_float64[arr_np_1_float64 >= 0.5]

	d_own_float64_lower = calc_simple_stats(arr=arr_float64_lower)
	d_own_float64_higher = calc_simple_stats(arr=arr_float64_higher)
	
	d_np_1_float64_lower = calc_simple_stats(arr=arr_np_1_float64_lower)
	d_np_1_float64_higher = calc_simple_stats(arr=arr_np_1_float64_higher)

	print()
	print(f"d_own_float64: {sorted(d_own_float64.items())}")
	print(f"d_np_1_float64: {sorted(d_np_1_float64.items())}")
	print()
	print(f"d_own_float64_lower: {sorted(d_own_float64_lower.items())}")
	print(f"d_np_1_float64_lower: {sorted(d_np_1_float64_lower.items())}")
	print()
	print(f"d_own_float64_higher: {sorted(d_own_float64_higher.items())}")
	print(f"d_np_1_float64_higher: {sorted(d_np_1_float64_higher.items())}")
	print()

	for epsilon in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
		arr_uniform = np.arange(epsilon, 1., epsilon)
		d_uniform = calc_simple_stats(arr=arr_uniform)
		print(f"epsilon: {epsilon}, d_uniform: {sorted(d_uniform.items())}")

	sys.exit()

	amount = 2**16*50

	arr_seed_1 = np.array([0x00000000], dtype=np.uint32)
	arr_seed_2 = np.array([0x00000001], dtype=np.uint32)

	t1_1 = time.time()
	rnd_1 = RandomDeviceBetter(arr_seed=arr_seed_1)
	t1_2 = time.time()
	arr_1 = rnd_1.calc_next_uint16(amount=amount)
	t1_3 = time.time()

	# rnd_2 = RandomDeviceBetter(arr_seed=arr_seed_1)
	# arr_2 = rnd_2.calc_next_uint16(amount=amount)

	# rnd_3 = RandomDeviceBetter(arr_seed=arr_seed_1)
	# arr_3 = rnd_3.calc_next_uint16(amount=amount)

	t2_1 = time.time()
	rnd_np_1 = Generator(bit_generator=PCG64(seed=arr_seed_1))
	t2_2 = time.time()
	arr_np_1 = rnd_np_1.integers(0, 2**16, (amount, ))
	t2_3 = time.time()

	u1, c1 = get_unique_count(arr_1)
	u2, c2 = get_unique_count(arr_np_1)

	uc1, cc1 = get_unique_count(c1)
	uc2, cc2 = get_unique_count(c2)

	u_diff_1, c_diff_1 = get_unique_count(np.diff(arr_1))
	u_diff_2, c_diff_2 = get_unique_count(np.diff(arr_np_1))

	uc_diff_1, cc_diff_1 = get_unique_count(c_diff_1)
	uc_diff_2, cc_diff_2 = get_unique_count(c_diff_2)

	print(f"t1_2-t1_1: {t1_2-t1_1}")
	print(f"t1_3-t1_2: {t1_3-t1_2}")
	print()
	print(f"t2_2-t2_1: {t2_2-t2_1}")
	print(f"t2_3-t2_2: {t2_3-t2_2}")
	print()

	def calc_simple_stats(arr):
		val_min = np.min(arr)
		val_max = np.max(arr)
		val_median = np.median(arr)
		val_mean = np.mean(arr)
		val_std = np.std(arr)
		return dict(
			min=val_min,
			max=val_max,
			median=val_median,
			mean=val_mean,
			std=val_std,
		)

	dc1 = calc_simple_stats(arr=c1)
	dc2 = calc_simple_stats(arr=c2)

	dcc1 = calc_simple_stats(arr=cc1)
	dcc2 = calc_simple_stats(arr=cc2)

	dc_diff_1 = calc_simple_stats(arr=c_diff_1)
	dc_diff_2 = calc_simple_stats(arr=c_diff_2)

	dcc_diff_1 = calc_simple_stats(arr=cc_diff_1)
	dcc_diff_2 = calc_simple_stats(arr=cc_diff_2)

	print(f"dc1: {dc1}")
	print(f"dc2: {dc2}")
	print()
	print(f"dcc1: {dcc1}")
	print(f"dcc2: {dcc2}")
	print()

	print(f"dc_diff_1: {dc_diff_1}")
	print(f"dc_diff_2: {dc_diff_2}")
	print()
	print(f"dcc_diff_1: {dcc_diff_1}")
	print(f"dcc_diff_2: {dcc_diff_2}")
	print()

	fig, axs = plt.subplots(nrows=2, ncols=1)

	ax = axs[0]
	ax.plot(np.arange(0, len(u1)), u1, marker='.', markersize=2, linestyle='', linewidth=0.5)

	ax = axs[1]
	ax.plot(np.arange(0, len(c1)), c1, marker='.', markersize=2, linestyle='', linewidth=0.5)

	plt.show()

	sys.exit()

	n = 6
	m = 2**n

	length = m*20

	def get_new_l_slprng():
		l_slprng = [
			SimpleLinearPRNG(m=m, a=(1+4*4)%m, b=(1+5*2)%m, x=0),
			SimpleLinearPRNG(m=m, a=(1+3*4)%m, b=(1+9*2)%m, x=1),
			SimpleLinearPRNG(m=m, a=(1+6*4)%m, b=(1+3*2)%m, x=2),
			SimpleLinearPRNG(m=m, a=(1+10*4)%m, b=(1+8*2)%m, x=3),
			SimpleLinearPRNG(m=m, a=(1+14*4)%m, b=(1+7*2)%m, x=4),
		]
		return l_slprng

	l_slprng = get_new_l_slprng()
	for i in range(0, len(l_slprng)):
		slprng = l_slprng[i]
		l = []
		for _ in range(0, m*5):
			val = slprng.calc_next()
			slprng.x += 3
			l.append(val)

		arr = np.array(l)
		u, c = np.unique(arr, return_counts=True)

		print(f"i: {i}, np.all(c[0] == c): {np.all(c[0] == c)}")

		arr = np.array(l)
		pix = np.zeros((m, m), dtype=np.uint8)
		pix[(arr[:-1], arr[1:])] = 255
		img = Image.fromarray(pix)
		img.save(os.path.join(TEMP_DIR, f'slprng_nr_{i:02}.png'))
		# img.show()

	amount_repeat = 200

	l_slprng = get_new_l_slprng()
	l = []
	amount = len(l_slprng)
	for i in range(0, m*amount*amount_repeat):
		i_idx = i % amount
		l.append(l_slprng[i_idx].calc_next())
		l_slprng[i_idx].x = (l_slprng[i_idx].x + i) % m

	arr_1 = np.array(l)
	pix_1 = np.zeros((m, m), dtype=np.uint8)
	pix_1[(arr_1[:-1], arr_1[1:])] = 1
	img = Image.fromarray(pix_1 * 255)
	img.save(os.path.join(TEMP_DIR, f'slprng_comb_1.png'))

	l_slprng = get_new_l_slprng()
	l = []
	
	# l_idx = [0]
	# for i in range(1, amount):
	# 	l_idx = l_idx + [i] + l_idx
	
	# l_idx_orig = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
	# l_idx_orig = l_idx_orig[:15]
	l_idx_orig = list(get_primes(n=10000))
	l_idx = [v % amount for v in l_idx_orig]
	amount_idx = len(l_idx)

	amount = len(l_slprng)
	for i in range(0, m*amount*amount_repeat):
		# l.append(l_slprng[i % amount].calc_next())
		l.append(l_slprng[l_idx[i % amount_idx]].calc_next())

	arr_2 = np.array(l)
	pix_2 = np.zeros((m, m), dtype=np.uint8)
	pix_2[(arr_2[:-1], arr_2[1:])] = 1
	img = Image.fromarray(pix_2 * 255)
	img.save(os.path.join(TEMP_DIR, f'slprng_comb_2.png'))

	print(f"np.sum(pix_1 == 1): {np.sum(pix_1 == 1)}")
	print(f"np.sum(pix_2 == 1): {np.sum(pix_2 == 1)}")

	sys.exit()

	r_11 = 5
	r_12 = 2
	a_1 = (1 + r_11 * 4) % m
	b_1 = (1 + r_12 * 2) % m
	slprng_1 = SimpleLinearPRNG(m=m, a=a_1, b=b_1, x=0)
	l_1 = [slprng_1.x]
	for _ in range(1, length):
		# x = (a_1 * x + b_1) % m
		# b_1 = (b_1 + 2) % m
		# l_1.append(x)
		l_1.append(slprng_1.calc_next())

	# r_21 = 8
	# r_22 = 3
	# a_2 = (1 + r_21 * 4) % m
	# b_2 = (1 + r_22 * 2) % m
	# x = 2
	# l_2 = [x]
	# for _ in range(1, length):
	# 	x = (a_2 * x + b_2) % m
	# 	l_2.append(x) 

	# r_211 = 8
	# r_212 = 3
	# a_21 = (1 + r_211 * 4) % m
	# b_21 = (1 + r_212 * 2) % m

	# r_221 = 9
	# r_222 = 4
	# a_22 = (1 + r_221 * 4) % m
	# b_22 = (1 + r_222 * 2) % m

	l_slprng_2 = [
		SimpleLinearPRNG(m=m, a=(1+4*4)%m, b=(1+5*2)%m, x=0),
		SimpleLinearPRNG(m=m, a=(1+3*4)%m, b=(1+9*2)%m, x=0),
		SimpleLinearPRNG(m=m, a=(1+6*4)%m, b=(1+3*2)%m, x=0),
		SimpleLinearPRNG(m=m, a=(1+10*4)%m, b=(1+8*2)%m, x=0),
		SimpleLinearPRNG(m=m, a=(1+14*4)%m, b=(1+7*2)%m, x=0),
	]

	# x_1 = 0
	# x_2 = 0
	l_2 = [l_slprng_2[(l_slprng_2[0].calc_next() % 3) + 1].calc_next()]
	for i in range(1, length):
		# x_1 = (a_21 * x_1 + b_21) % m
		# x_2 = (a_22 * x_2 + b_22) % m
		# l_2.append(x_1)
		l_2.append(l_slprng_2[(l_slprng_2[0].calc_next() % 3) + 1].calc_next())

	arr_1 = np.array(l_1)
	arr_2 = np.array(l_2)
	arr_comb = (np.array(l_1) + np.array(l_2)) % m

	arr_x = np.arange(0, arr_1.shape[0])

	# plt.figure()

	# # plt.title(f"n: {n}, m: {m},\nr_11: {r_11}, r_12: {r_12}, a_1: {a_1}, b_1: {b_1},\nr_211: {r_211}, r_212: {r_212}, a_21: {a_21}, b_21: {b_21}")

	# plt.plot(arr_x, arr_1, color='#0000FF', marker='o', markersize=1, linestyle='')
	# plt.plot(arr_x, arr_2, color='#00FF00', marker='o', markersize=1, linestyle='')

	# plt.plot(arr_x, arr_comb, color='#FF0000', marker='o', markersize=2, linestyle='')


	# plt.figure()

	# plt.title(f"n: {n}, m: {m},\nr_11: {r_11}, r_12: {r_12}, a_1: {a_1}, b_1: {b_1},\nr_211: {r_211}, r_212: {r_212}, a_21: {a_21}, b_21: {b_21}")

	# p1, = plt.plot(arr_x[:-1], np.diff(arr_1) % m, color='#0000FF', marker='o', markersize=1, linestyle='', label='arr_1')
	# p2, = plt.plot(arr_x[:-1], np.diff(arr_2) % m, color='#00FF00', marker='o', markersize=1, linestyle='', label='arr_2')
	# p3, = plt.plot(arr_x[:-1], np.diff(arr_comb) % m, color='#FF0000', marker='o', markersize=1, linestyle='', label='arr_comb')

	# plt.legend(handles=[p1, p2, p3])


	plt.figure()
	p1, = plt.plot(arr_x[:-1], np.diff(arr_1) % m, color='#0000FF', marker='o', markersize=1, linestyle='', label='arr_1')
	plt.title('arr_1')
	plt.legend(handles=[p1])

	plt.figure()
	p1, = plt.plot(arr_x[:-1], np.diff(arr_2) % m, color='#00FF00', marker='o', markersize=1, linestyle='', label='arr_2')
	plt.title('arr_2')
	plt.legend(handles=[p1])

	plt.figure()
	p1, = plt.plot(arr_x[:-1], np.diff(arr_comb) % m, color='#FF0000', marker='o', markersize=1, linestyle='', label='arr_comb')
	plt.title('arr_comb')
	plt.legend(handles=[p1])

	plt.show()

	pix = np.zeros((m, arr_1.shape[0], 3), dtype=np.uint8)

	pix[l_1, arr_x] ^= np.array((0, 0, 0xFF), dtype=np.uint8)
	pix[l_2, arr_x] ^= np.array((0, 0xFF, 0), dtype=np.uint8)
	pix[arr_comb, arr_x] ^= np.array((0xFF, 0, 0), dtype=np.uint8)

	img = Image.fromarray(pix)
	img.save(os.path.join(TEMP_DIR, f'temp_img_n_{n}_m_{m}.png'))
	# img.show()
