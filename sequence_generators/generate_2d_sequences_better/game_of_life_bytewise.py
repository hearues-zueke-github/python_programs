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

if __name__ == '__main__':
	print("Hello World!")

	rows = 200
	cols = 200
	arr_field = np.random.randint(0, 256, (rows, cols)).astype(np.uint8)

	print(f'arr_field: {arr_field}')
	print(f'arr_field.shape: {arr_field.shape}')

	# arr_mult_field = np.random.randint(0, 256, (9, )).astype(np.uint8)
	arr_mult_field = np.random.randint(0, 2, (9, )).astype(np.uint8)

	dir_path = os.path.join(TEMP_DIR, 'game_of_life_bytewise')
	mkdirs(dir_path)

	img = Image.fromarray(arr_field)
	img.save(os.path.join(dir_path, 'gol_bytewise_0000.png'))


	for i_round in range(1, 101):
		print(f'i_round: {i_round}')
		arr_l = np.roll(arr_field, 1, axis=1)
		arr_r = np.roll(arr_field, -1, axis=1)
		arr_u = np.roll(arr_field, 1, axis=0)
		arr_d = np.roll(arr_field, -1, axis=0)

		arr_lu = np.roll(arr_u, 1, axis=1)
		arr_ru = np.roll(arr_u, -1, axis=1)
		arr_ld = np.roll(arr_d, 1, axis=1)
		arr_rd = np.roll(arr_d, -1, axis=1)

		arr_field[:] = (
			arr_field * arr_mult_field[0] ^
			arr_l * arr_mult_field[1] ^
			arr_r * arr_mult_field[2] ^
			arr_u * arr_mult_field[3] ^
			arr_d * arr_mult_field[4] ^
			arr_lu * arr_mult_field[5] ^
			arr_ru * arr_mult_field[6] ^
			arr_ld * arr_mult_field[7] ^
			arr_rd * arr_mult_field[8]
		)

		img = Image.fromarray(arr_field)
		img.save(os.path.join(dir_path, f'gol_bytewise_{i_round:04}.png'))
