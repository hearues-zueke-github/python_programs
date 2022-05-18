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

from random_number_device import RandomNumberDevice

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


def do_some_simple_statistical_analysis(name, arr):
	print(f"simple stats for {name}:")

	d = calc_simple_stats(arr=arr)

	arr_lower = arr[arr < 0.5]
	arr_higher = arr[arr >= 0.5]

	d_lower = calc_simple_stats(arr=arr_lower)
	d_higher = calc_simple_stats(arr=arr_higher)
	
	print(f"- d: {sorted(d.items())}")
	print()
	print(f"- d_lower: {sorted(d_lower.items())}")
	print(f"- d_higher: {sorted(d_higher.items())}")
	print()


if __name__ == '__main__':
	print("Hello World!")

	amount = 5*10**6

	arr_seed_uint8 = np.array([0x01], dtype=np.uint8)

	t_1 = time.time()
	rnd_own = RandomNumberDevice(arr_seed_uint8=arr_seed_uint8, length_uint8=1024)
	print(f"rnd_own.arr_mult_x: {rnd_own.arr_mult_x.tolist()}")
	print(f"rnd_own.arr_mult_a: {rnd_own.arr_mult_a.tolist()}")
	print(f"rnd_own.arr_mult_b: {rnd_own.arr_mult_b.tolist()}")
	print(f"rnd_own.arr_xor_x: {rnd_own.arr_xor_x.tolist()}")
	print(f"rnd_own.arr_xor_a: {rnd_own.arr_xor_a.tolist()}")
	print(f"rnd_own.arr_xor_b: {rnd_own.arr_xor_b.tolist()}")
	t_2 = time.time()
	arr1 = rnd_own.calc_next_uint64(amount=10)
	arr2 = rnd_own.calc_next_uint64(amount=11)
	arr3 = rnd_own.calc_next_uint64(amount=12)
	arr4 = rnd_own.calc_next_uint64(amount=13)
	arr5 = rnd_own.calc_next_uint64(amount=23)
	arr6 = rnd_own.calc_next_uint64(amount=5)
	arr7 = rnd_own.calc_next_uint64(amount=4)
	arr8 = rnd_own.calc_next_uint64(amount=31)
	print(f"arr1: {arr1.tolist()}")
	print(f"arr2: {arr2.tolist()}")
	print(f"arr3: {arr3.tolist()}")
	print(f"arr4: {arr4.tolist()}")
	print(f"arr5: {arr5.tolist()}")
	print(f"arr6: {arr6.tolist()}")
	print(f"arr7: {arr7.tolist()}")
	print(f"arr8: {arr8.tolist()}")

	arr = rnd_own.calc_next_uint64(amount=amount)
	print(f"arr: {arr[:20].tolist()}")
	t_3 = time.time()
	arr_float64 = rnd_own.calc_next_float64(amount=amount)
	print(f"arr_float64: {arr_float64[:20].tolist()}")
	t_4 = time.time()

	t_diff_1 = t_2 - t_1
	t_diff_2 = t_3 - t_2
	t_diff_3 = t_4 - t_3

	print(f"t_diff_1: {t_diff_1}")
	print(f"t_diff_2: {t_diff_2}")
	print(f"t_diff_3: {t_diff_3}")

	print(f"np.min(arr_float64): {np.min(arr_float64)}, np.max(arr_float64): {np.max(arr_float64)}")

	# numpy is only for reference for the needed time
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

	do_some_simple_statistical_analysis(name="arr_float64", arr=arr_float64)
	do_some_simple_statistical_analysis(name="arr_np_1_float64", arr=arr_np_1_float64)

	print("for reference only (uniform distribution):")
	for epsilon in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
		arr_uniform = np.arange(epsilon, 1., epsilon)
		d_uniform = calc_simple_stats(arr=arr_uniform)
		print(f"- epsilon: {epsilon}, d_uniform: {sorted(d_uniform.items())}")
