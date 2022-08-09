#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import time
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
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def calculate_cycles_power_2(m, n_0, n_1, arr_comb_part, only_full_cycles=True, stop_by_finding_first_full_cycle=False):
	len_cyclic = m**2+1
	# d_cyclic_pow_2_unique_better = {}
	d_cyclic_pow_2_all = {}
	arr_factors = np.zeros((n_0, n_1), dtype=np.int64)
	arr_const_n_0 = np.arange(0, n_0).reshape((n_0, 1))
	arr_const_n_1 = np.arange(0, n_1).reshape((1, n_1))
	# TODO: create a dict for each value with the power modulo values
	for row in arr_comb_part:
		# print(f"row: {row}")
		arr_factors[:] = row.reshape((n_0, n_1))
					
		l_x = [0, 0]
		s_x = set([(0, 0)])
		x_i0 = 0
		x_i1 = 0

		for _ in range(0, m**2):
			x_i2 = np.sum((arr_factors * (x_i0**arr_const_n_0)) * (x_i1**arr_const_n_1)) % m
			
			t = (x_i1, x_i2)
			if t in s_x:
				break

			x_i0 = x_i1
			x_i1 = x_i2
			l_x.append(x_i1)
			s_x.add((x_i0, x_i1))

		x_i2 = np.sum((arr_factors * (x_i0**arr_const_n_0)) * (x_i1**arr_const_n_1)) % m
		x_i0 = x_i1
		x_i1 = x_i2

		# if len(l_x) == len_cyclic and l_x[0] == x_i0 and l_x[1] == x_i1:
		# 	d_cyclic_pow_2_unique_better[tuple(row.tolist())] = tuple(l_x[:-1])

		if l_x[0] == x_i0 and l_x[1] == x_i1:
			len_l_x = len(l_x)-1
			is_full_cycle = len_l_x == m**2
			
			if only_full_cycles and not is_full_cycle:
				continue

			if len_l_x not in d_cyclic_pow_2_all:
				d_cyclic_pow_2_all[len_l_x] = {}
			d_cyclic_pow_2_all[len_l_x][tuple(row.tolist())] = tuple(l_x[:-1])

			if is_full_cycle and stop_by_finding_first_full_cycle:
				break

	return d_cyclic_pow_2_all


if __name__ == '__main__':
	# TODO: add either reading params from file/s or from program arguments
	m = 6

	n_0 = 2
	n_1 = 3
	n_mult = n_0 * n_1

	# find the max upperbound for the exponent, which can be split up in 
	max_n = 10**7
	max_exponent = 0
	while m**(max_exponent + 1) < max_n:
		max_exponent += 1

	amount_cpu = mp.cpu_count() # use this, if you want to fully load your cpu
	# amount_cpu = int(mp.cpu_count() *0.75) + 1
	amount_worker_proc = amount_cpu - 1

	amount_part = amount_worker_proc * 4

	mult_proc_mng = MultiprocessingManager(cpu_count=amount_cpu)

	def func_wrap(d):
		return calculate_cycles_power_2(**d)


	def execute_next_batch(mult_proc_mng, amount_part, d_cyclic_pow_2_all, arr_comb, d_params_base):
		amount_comb = arr_comb.shape[0]

		amount_full_part = amount_comb // amount_part
		amount_one_more = amount_comb % amount_part

		l_amount_part = [amount_full_part+1]*amount_one_more + [amount_full_part]*(amount_part-amount_one_more)
		arr_acc_idx = np.cumsum([0] + l_amount_part)

		l_args = [
			(
				d_params_base | {
				'arr_comb_part': arr_comb[idx_1:idx_2],
			}, ) # must be a tuple for the arg!
			for idx_1, idx_2 in zip(arr_acc_idx[:-1], arr_acc_idx[1:])
		]
		l_ret = mult_proc_mng.do_new_jobs(
			['func_wrap']*len(l_args),
			l_args,
		)
		print("len(l_ret): {}".format(len(l_ret)))

		for d_src in l_ret:
			for k, v in d_src.items():
				if k not in d_cyclic_pow_2_all:
					d_cyclic_pow_2_all[k] = {}
				d_cyclic_pow_2_all[k].update(v)

	mult_proc_mng.define_new_func('func_wrap', func_wrap)

	d_params_base = {
		'm': m,
		'n_0': n_0,
		'n_1': n_1,
		'only_full_cycles': True,
		'stop_by_finding_first_full_cycle': False,
		# 'stop_by_finding_first_full_cycle': True,
	}
	
	d_cyclic_pow_2_all = {}
	if n_mult > max_exponent: # split the parts in mini batch works
		arr_comb_orig = get_all_combinations_repeat(m=m, n=max_exponent)
		arr_comb_prefix = get_all_combinations_repeat(m=m, n=n_mult-max_exponent)

		# remove the first prefix value 0, because all of the sequences needs at least one constant value!
		arr_comb_prefix = arr_comb_prefix[arr_comb_prefix[:, 0] != 0]

		# mix the arr_comb_prefix a bit, if needed
		arr_rnd_idx = np.random.permutation(np.arange(0, arr_comb_prefix.shape[0]))
		arr_comb_prefix = arr_comb_prefix[arr_rnd_idx]

		# arr_rnd_idx = np.random.permutation(np.arange(0, arr_comb.shape[0]))
		# arr_comb = arr_comb[arr_rnd_idx]

		# length of arr_prefix_mat is n_mult - max_exponent, if n_mult >= max_exponent
		arr_prefix_mat = np.zeros((arr_comb_orig.shape[0], n_mult-max_exponent), dtype=np.uint8)
		arr_comb = np.hstack((arr_prefix_mat, arr_comb_orig))
		
		for row_prefix in arr_comb_prefix:
			print(f"row_prefix: {row_prefix}")
			arr_comb[:, :n_mult-max_exponent] = row_prefix

			time_1 = time.time()
			execute_next_batch(
				mult_proc_mng=mult_proc_mng,
				amount_part=amount_part,
				d_cyclic_pow_2_all=d_cyclic_pow_2_all,
				arr_comb=arr_comb,
				d_params_base=d_params_base,
			)
			time_2 = time.time()

			time_diff = time_2 - time_1
			print(f"Needed time for arr_comb.shape = {arr_comb.shape}: {time_diff:.5f}s")

			if m**2 in d_cyclic_pow_2_all:
				print(f"At least one cycle was found with the length m**2 = {m**2}")
				break
	else:
		arr_comb = get_all_combinations_repeat(m=m, n=n_mult)
		
		time_1 = time.time()
		execute_next_batch(
			mult_proc_mng=mult_proc_mng,
			amount_part=amount_part,
			d_cyclic_pow_2_all=d_cyclic_pow_2_all,
			arr_comb=arr_comb,
			d_params_base=d_params_base,
		)
		time_2 = time.time()

		time_diff = time_2 - time_1
		print(f"Needed time for arr_comb.shape = {arr_comb.shape}: {time_diff:.5f}s")

	# # testing the responsivness again!
	# mult_proc_mng.test_worker_threads_response()
	del mult_proc_mng

	print(f"m: {m}")

	df = pd.DataFrame(data=list(d_cyclic_pow_2_all[m**2].items()), columns=['tpl_factor', 'tpl_l_x'], dtype=object)
	df.sort_values(by=['tpl_l_x', 'tpl_factor'], inplace=True)
	u, c = np.unique(df['tpl_l_x'].values, return_counts=True)

	df_unique = df.drop_duplicates(subset=['tpl_l_x'], keep='first').copy()
	df_unique.reset_index(inplace=True, drop=True)

	df_unique['df_stats_cluster'] = None
	df_unique['t_best_cluster'] = None
	for row_idx, row in df_unique.iterrows():
		# print(f"row_idx: {row_idx}")

		t = row['tpl_l_x']
		arr = np.array(t, dtype=np.int8) # m is very probably not bigger than 127
		arr_diff = (np.roll(arr, -1, 0) - arr) % m

		if arr_diff[0] == arr_diff[-1]:
			assert False and "This should never happen!"

		arr_idx_bool = np.hstack(((True, ), arr_diff[:-1] != arr_diff[1:], (True, )))
		arr_idx_pos = np.where(arr_idx_bool)[0]

		d = {}
		for idx_1, idx_2, v_diff in zip(arr_idx_pos[:-1], arr_idx_pos[1:], arr_diff):
			length = idx_2 - idx_1
			t = (v_diff, length)

			if t not in d:
				d[t] = 0
			d[t] += 1

		df_stats_cluster = pd.DataFrame(
			data=[(k[1], k[0], v) for k, v in d.items()],
			columns=['length', 'v_diff', 'amount'],
			dtype=np.int64,
		).sort_values(
			by=['length', 'v_diff', 'amount'],
			ascending=[False, True, True],
		)
		row['df_stats_cluster'] = df_stats_cluster
		row['t_best_cluster'] = tuple(df_stats_cluster.iloc[0].values.tolist())

	# take from the best cluster the best of all, where the length is max and the v_diff the smallest.
	df_t_best_cluster = pd.DataFrame(
		data=df_unique['t_best_cluster'].values.tolist(),
		columns=['length', 'v_diff', 'amount'],
		dtype=np.int64,
	)
	df_t_best_cluster['idx'] = df_t_best_cluster.index.values

	df_t_best_cluster_sort = df_t_best_cluster.sort_values(
		by=['length', 'v_diff', 'amount'],
		ascending=[False, True, True],
	).reset_index(
		drop=True,
	)

	ser_t_best_cluster = df_t_best_cluster_sort.iloc[0]
	print(f"\nser_t_best_cluster:\n{ser_t_best_cluster}")
