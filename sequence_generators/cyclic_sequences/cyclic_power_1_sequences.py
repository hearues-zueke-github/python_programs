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

# Needed for excel tables
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import column_index_from_string
from openpyxl.styles.borders import Border, Side, BORDER_THIN, BORDER_MEDIUM
from openpyxl.styles import Alignment, borders, Font

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
load_module_dynamically(**dict(var_glob=var_glob, name='utils_wb', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils/utils_wb.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
create_new_sheet = utils_wb.create_new_sheet
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	l_column_stat_data = ['m', 'n_0', 'df_len', 'u_len', 'df_ref_len', 'u_ref_len']
	d_stat_data = {column: [] for column in l_column_stat_data}
	
	d_n_0_to_df_unique_combined = {}

	m = 12
	for n_0 in range(2, 6):
		print(f"m: {m}, n_0: {n_0}")
		
		len_cyclic = m**1
		d_cyclic_pow_1_all_better = {}
		arr_factors = np.zeros((n_0, ), dtype=np.int64)
		arr_const_n_0 = np.arange(0, n_0).reshape((n_0, ))
		for row in get_all_combinations_repeat(m=m, n=n_0):

			arr_factors[:] = row.reshape((n_0, ))

			l_x = [0]
			s_x = set([(0, )])
			x_i0 = 0

			for _ in range(0, m**2):
				x_i1 = np.sum(arr_factors * (x_i0**arr_const_n_0)) % m
				
				t = (x_i1, )
				if t in s_x:
					break

				x_i0 = x_i1
				l_x.append(x_i0)
				s_x.add((x_i0, ))

			x_i1 = np.sum(arr_factors * (x_i0**arr_const_n_0)) % m
			x_i0 = x_i1

			if l_x[0] == x_i0:
				len_l_x = len(l_x)
				if len_l_x not in d_cyclic_pow_1_all_better:
					d_cyclic_pow_1_all_better[len_l_x] = {}
				d_cyclic_pow_1_all_better[len_l_x][tuple(row.tolist())] = tuple(l_x)

		print(f"m: {m}")

		df = pd.DataFrame(
			data=list(d_cyclic_pow_1_all_better[m].items()),
			columns=['tpl_factor', 'tpl_l_x'],
			dtype=object,
		)
		df.sort_values(by=['tpl_factor', 'tpl_l_x'], inplace=True)
		# df.sort_values(by=['tpl_l_x', 'tpl_factor'], inplace=True)
		u, c = np.unique(df['tpl_l_x'].values, return_counts=True)

		df_unique = df.drop_duplicates(subset=['tpl_l_x'], keep='first').copy()
		df_unique.reset_index(inplace=True, drop=True)

		df_unique_sort = df_unique.sort_values(by=['tpl_factor']).reset_index(drop=True)
		df_unique_combined = pd.DataFrame(
			data=np.hstack((
				np.array(df_unique_sort['tpl_factor'].values.tolist()),
				np.array(df_unique_sort['tpl_l_x'].values.tolist()),
			)),
			columns=(
				[f'f_{i}' for i in range(0, len(df_unique_sort['tpl_factor'].iloc[0]))]+
				[f'x_{i}' for i in range(0, len(df_unique_sort['tpl_l_x'].iloc[0]))]
			),
			dtype=np.int64,
		)
		d_n_0_to_df_unique_combined[n_0] = df_unique_combined

		len_cyclic = m**1
		d_cyclic_pow_1 = {}
		d_cyclic_pow_1_all = {}
		for a in range(0, m):
			for b in range(0, m):
				l_x = [0]
				s_x = set([0])
				x = 0

				for _ in range(0, m):
					x_new = (a * x + b) % m
					if x_new in s_x:
						break

					x = x_new
					l_x.append(x)
					s_x.add(x)

				x_last = (a * x + b) % m
				if len(l_x) == len_cyclic and l_x[0] == x_last:
					d_cyclic_pow_1[(a, b)] = l_x

				if l_x[0] == x_last:
					len_l_x = len(l_x)
					if len_l_x not in d_cyclic_pow_1_all:
						d_cyclic_pow_1_all[len_l_x] = []
					d_cyclic_pow_1_all[len_l_x].append(((a, b), l_x))

		df_ref = pd.DataFrame(data=[(k, tuple(v)) for k, v in d_cyclic_pow_1_all[m]], columns=['tpl_factor', 'tpl_l_x'], dtype=object)
		u_ref, c_ref = np.unique(df_ref.sort_values(by=['tpl_l_x', 'tpl_factor'])['tpl_l_x'].values, return_counts=True)

		print(f"m: {m}")
		print(f"n_0: {n_0}")
		print(f"- df.shape: {df.shape}")
		print(f"- u.shape: {u.shape}")
		print(f"- df_ref.shape: {df_ref.shape}")
		print(f"- u_ref.shape: {u_ref.shape}")

		d_stat_data['m'].append(m)
		d_stat_data['n_0'].append(n_0)
		d_stat_data['df_len'].append(df.shape[0])
		d_stat_data['u_len'].append(u.shape[0])
		d_stat_data['df_ref_len'].append(df_ref.shape[0])
		d_stat_data['u_ref_len'].append(u_ref.shape[0])


	df_stat_data = pd.DataFrame(data=d_stat_data, columns=l_column_stat_data, dtype=object)
	print(f"df_stat_data:\n{df_stat_data}")


	wb = openpyxl.Workbook()
	del wb['Sheet']

	for n_0, df in d_n_0_to_df_unique_combined.items():
		lst_build = [df.columns.tolist()] + df.values.tolist()
		column_widths = []
		create_new_sheet(
			wb=wb,
			sheet_name=f'n_0__{n_0}',
			list_build=lst_build,
			column_widths=column_widths,
			wrap_first_row=True,
			first_row_height=30,
			freeze_row=1,
			freeze_column=n_0,
		)

	file_path_tmp = os.path.join(
		TEMP_DIR,
		f"cycle_power_1_sequences_m_{m}_dt_{datetime.datetime.now().strftime('%Y_%m_%d')}.xlsx"
	)
	wb.save(file_path_tmp)

	print(f"Saved file '{file_path_tmp}'")
