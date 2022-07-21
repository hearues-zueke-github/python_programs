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

get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
	s_tpl_available = set()
	s_tpl_used = set([(0, ), (1, )])

	current_used_n = 1
	l = [0, 1]

	current_used_n += 1
	arr_comb = get_all_combinations_repeat(m=2, n=current_used_n)
	for row_comb in arr_comb:
		s_tpl_available.add(tuple(row_comb.tolist()))

	iter_nr = 2

	while iter_nr < 1000:
		# remove every single combinations of lengths
		for length in range(1, current_used_n+1):
			for i in range(0, len(l)-length+1):
				tpl = tuple(l[i:i+length])
				if tpl in s_tpl_available:
					s_tpl_available.remove(tpl)
					s_tpl_used.add(tpl)
		
		is_bit_found = False
		while not is_bit_found:
			for length in range(1, current_used_n):
				l_part = l[-length:]
				for bit in range(0, 2):
					tpl = tuple(l_part + [bit])
					if tpl in s_tpl_available:
						s_tpl_available.remove(tpl)
						s_tpl_used.add(tpl)
						is_bit_found = True
						break
				if is_bit_found:
					break

			if not is_bit_found:
				current_used_n += 1
				arr_comb = get_all_combinations_repeat(m=2, n=current_used_n)
				for row_comb in arr_comb:
					s_tpl_available.add(tuple(row_comb.tolist()))

				length = current_used_n
				for i in range(0, len(l)-length+1):
					tpl = tuple(l[i:i+length])
					if tpl in s_tpl_available:
						s_tpl_available.remove(tpl)
						s_tpl_used.add(tpl)

		iter_nr += 1

		l.append(bit)

		# print(f"iter_nr: {iter_nr}")
		# print(f"- current_used_n: {current_used_n}")
		# print(f"- s_tpl_available: {s_tpl_available}")
		# print(f"- s_tpl_used: {s_tpl_used}")
		# print(f"- is_bit_found: {is_bit_found}")
		# print(f"- bit: {bit}")
		# print(f"- l: {l}")

	# see the sequence A334941
	print(f"l: {l}")
