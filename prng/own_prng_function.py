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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

save_pkl_obj = utils_serialization.save_pkl_obj
load_pkl_obj = utils_serialization.load_pkl_obj

class RandomDevice():

	def __init__(self, arr_seed):
		assert isinstance(arr_seed, np.ndarray)
		assert arr_seed.dtype == np.uint8(0).dtype

		self.arr_seed = arr_seed.copy()

		self.arr_state = np.zeros((256, ), dtype=np.uint8)
		min_amount = min(self.arr_seed.shape[0], self.arr_state.shape[0])
		self.arr_state[:min_amount] = self.arr_seed[:min_amount]

		self.float_size = 4
		self.block_size = 32
		self.arr_idx = np.arange(0, 256+1, self.block_size)
		self.arr_idx_pair = np.vstack((self.arr_idx[:-1], self.arr_idx[1:])).T

		self.amount_pair = self.arr_idx_pair.shape[0]

		self.idx1 = 0
		self.idx2 = 1

		self.block_idx = self.block_size
		self.block_idx_max = self.arr_idx.shape[0]

		self.min_val_float32 = np.float32(2)**-32


	def calc_next_float(self):
		if self.block_idx % self.block_size == 0:
			self.calc_next_hash()

		arr = self.arr_state[self.block_idx:self.block_idx+self.float_size]
		self.block_idx += self.float_size
		if self.block_idx >= self.block_idx_max:
			self.block_idx = 0

		val = self.min_val_float32 * arr.view(np.uint32)[0]
		return val


	def calc_next_hash(self):
		# print(f"self.idx1: {self.idx1}, self.idx2: {self.idx2}")
		idx11, idx12 = self.arr_idx_pair[self.idx1]
		idx21, idx22 = self.arr_idx_pair[self.idx2]

		arr1 = self.arr_state[idx11:idx12]
		arr2 = self.arr_state[idx21:idx22]
		
		arr_hash = np.frombuffer(sha256(arr1.data).digest(), dtype=np.uint8)

		# print(f"- arr1:\n{arr1}")
		# print(f"- arr2:\n{arr2}")
		# print(f"- arr_hash:\n{arr_hash}")
		self.arr_state[idx21:idx22] ^= arr_hash
		# print(f"- arr2 ^= arr_hash:\n{arr2}")
		# print()

		self.idx1 = self.idx2
		self.idx2 = (self.idx2 + 1) % self.amount_pair


class RandomDeviceBetter():

	def __init__(self, arr_seed):
		assert isinstance(arr_seed, np.ndarray)
		assert arr_seed.dtype == np.uint32(0).dtype

		self.arr_seed = arr_seed.copy()

		self.arr_state = np.zeros((64, ), dtype=np.uint32)
		self.arr_state_next = np.zeros((64, ), dtype=np.uint32)
		min_amount = min(self.arr_seed.shape[0], self.arr_state.shape[0])
		self.arr_state[:min_amount] = self.arr_seed[:min_amount]

		self.float_size = 4
		self.block_size = 8
		self.arr_idx = np.arange(0, 64+1, self.block_size)
		self.arr_idx_pair = np.vstack((self.arr_idx[:-1], self.arr_idx[1:])).T

		self.amount_pair = self.arr_idx_pair.shape[0]

		self.idx1 = 0
		self.idx2 = 1

		self.block_idx = self.block_size
		self.block_idx_max = 64

		for _ in range(0, self.block_size):
			self.calc_next_hash()

		self.arr_a = self.arr_state.copy()
		self.arr_a = self.arr_a // 4 + 1

		for _ in range(0, self.block_size):
			self.calc_next_hash()

		self.arr_b = self.arr_state.copy()
		self.arr_b = self.arr_b // 2 + 1

		for _ in range(0, self.block_size):
			self.calc_next_hash()

		self.min_val_float32 = np.float32(2)**-32


	def calc_next_hash(self):
		idx11, idx12 = self.arr_idx_pair[self.idx1]
		idx21, idx22 = self.arr_idx_pair[self.idx2]

		arr1 = self.arr_state[idx11:idx12]
		arr2 = self.arr_state[idx21:idx22]
		
		arr_hash = np.frombuffer(sha256(arr1.data).digest(), dtype=np.uint32)

		self.arr_state[idx21:idx22] ^= arr_hash

		self.idx1 = self.idx2
		self.idx2 = (self.idx2 + 1) % self.amount_pair


	def calc_next_float(self, amount=64):
		arr = np.empty((amount, ), dtype=np.float32)

		for i in range(0, amount - amount % 64, 64):
			self.arr_state_next[:] = self.arr_a
			self.arr_state_next *= self.arr_state
			self.arr_state_next += self.arr_b
			self.arr_state[:] = self.arr_state_next
			arr[i:i+64] = self.min_val_float32 * self.arr_state

		self.arr_state_next[:] = self.arr_a
		self.arr_state_next *= self.arr_state
		self.arr_state_next += self.arr_b
		self.arr_state[:] = self.arr_state_next

		amount_last = amount % 64
		if amount_last > 0:
			arr[-amount_last:] = self.min_val_float32 * self.arr_state[:amount_last]

		return arr


if __name__ == '__main__':
	print("Hello World!")

	# arr_state = np.zeros((256, ), dtype=np.uint8)

	arr_seed = np.array([0x00, 0x00, 0x01], dtype=np.uint32)

	# rnd = RandomDeviceBetter(arr_seed=arr_seed.astype(np.uint32))
	# sys.exit()

	for i_pow in range(0, 6):
		n = 10**i_pow
		print(f"\nn: {n}")

		time_own_1 = time.time()
		rnd = RandomDeviceBetter(arr_seed=arr_seed)
		arr = rnd.calc_next_float(amount=n)
		time_own_2 = time.time()

		print(f"min(arr):  {np.min(arr)}")
		print(f"max(arr):  {np.max(arr)}")
		print(f"median(arr):  {np.median(arr)}")
		print(f"mean(arr):  {np.mean(arr)}")
		print(f"std(arr):  {np.std(arr)}")
		print()

		time_np_1 = time.time()
		rnd_piece = Generator(bit_generator=PCG64(seed=arr_seed))
		arr_2 = rnd_piece.random((n, ))
		time_np_2 = time.time()

		print(f"min(arr_2):  {np.min(arr_2)}")
		print(f"max(arr_2):  {np.max(arr_2)}")
		print(f"median(arr_2):  {np.median(arr_2)}")
		print(f"mean(arr_2):  {np.mean(arr_2)}")
		print(f"std(arr_2):  {np.std(arr_2)}")
		print()

		diff_time_own = time_own_2 - time_own_1
		print(f"diff_time_own: {diff_time_own}")

		diff_time_np = time_np_2 - time_np_1
		print(f"diff_time_np: {diff_time_np}")

		print(f"diff_time_own / diff_time_np: {diff_time_own / diff_time_np}")

	# n = 4
	# arr_a = np.zeros((n, n), dtype=np.uint32)
	# arr_b = np.zeros((n, n), dtype=np.uint32)
	# arr_c = np.zeros((n, n), dtype=np.uint32)


	# # arr_state = np.zeros((64, ), dtype=np.uint8)
	# arr_state[:min(arr_seed.shape[0], arr_state.shape[0])] = arr_seed

	# arr_idx = np.arange(0, 256+1, 32)
	# arr_idx_pair = np.vstack((arr_idx[:-1], arr_idx[1:])).T

	# amount_pair = arr_idx_pair.shape[0]

	# idx1 = 0
	# idx2 = 1

	# for _ in range(0, 10):
	# 	print(f"idx1: {idx1}, idx2: {idx2}")
	# 	idx11, idx12 = arr_idx_pair[idx1]
	# 	idx21, idx22 = arr_idx_pair[idx2]

	# 	arr1 = arr_state[idx11:idx12]
	# 	arr2 = arr_state[idx21:idx22]
		
	# 	arr_hash = np.frombuffer(sha256(arr1.data).digest(), dtype=np.uint8)

	# 	print(f"- arr1:\n{arr1}")
	# 	print(f"- arr2:\n{arr2}")
	# 	print(f"- arr_hash:\n{arr_hash}")
	# 	arr_state[idx21:idx22] ^= arr_hash
	# 	print(f"- arr2 ^= arr_hash:\n{arr2}")
	# 	print()

	# 	idx1 = idx2
	# 	idx2 = (idx2 + 1) % amount_pair

	# print(f"arr_state: {arr_state}")
	# arr_hash1 = np.frombuffer(sha256(arr_state.data).digest(), dtype=np.uint32)
	# arr_hash2 = np.frombuffer(sha256((arr_state[8*0:8*1] ^ arr_hash1).data).digest(), dtype=np.uint32)
	# arr_hash3 = np.frombuffer(sha256((arr_state[8*1:8*2] ^ arr_hash2).data).digest(), dtype=np.uint32)
	# arr_hash4 = np.frombuffer(sha256((arr_state[8*2:8*3] ^ arr_hash3).data).digest(), dtype=np.uint32)
	# arr_hash5 = np.frombuffer(sha256((arr_state[8*3:8*4] ^ arr_hash4).data).digest(), dtype=np.uint32)


