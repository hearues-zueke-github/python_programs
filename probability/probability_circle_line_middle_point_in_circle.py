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

import matplotlib.pyplot as plt

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

PI = 3.14159265358979323846

if __name__ == '__main__':
	n = 1001
	delta_alpha = 2 * PI / n

	r = 1.
	arr_x = np.array([r*np.cos(delta_alpha * i) for i in range(0, n)])
	arr_y = np.array([r*np.sin(delta_alpha * i) for i in range(0, n)])

	# create a discret 2d histogram bin
	bins = 1000 # amount of bins per axis
	arr_hist = np.zeros((bins, bins), dtype=np.int64)

	min_x, max_x = np.min(arr_x), np.max(arr_x)
	min_y, max_y = np.min(arr_y), np.max(arr_y)

	# add a small epsilon to the min/max values
	min_x -= 1e-30
	min_y -= 1e-30
	max_x += 1e-30
	max_y += 1e-30

	diff_x = max_x - min_x
	diff_y = max_y - min_y

	ax = plt.gca()
	ax.axis('equal')

	arr_x_1 = arr_x.copy()
	arr_y_1 = arr_y.copy()

	l_middle_x = []
	l_middle_y = []
	for _ in range(0, n // 2):
		arr_x_1 = np.roll(arr_x_1, 1)
		arr_y_1 = np.roll(arr_y_1, 1)

		for x1, x2, y1, y2 in zip(arr_x, arr_x_1, arr_y, arr_y_1):
			# plt.plot((x1, x2), (y1, y2), '-', color='#000000') # plot the lines of each point combinations!

			x3 = (x1 + x2) / 2
			y3 = (y1 + y2) / 2

			l_middle_x.append(x3)
			l_middle_y.append(y3)

			bin_x = int((x3 - min_x) / diff_x * bins)
			bin_y = int((y3 - min_y) / diff_y * bins)
			arr_hist[bin_y, bin_x] += 1


	arr_hist_byte = (arr_hist / np.max(arr_hist) * 255.999).astype(np.uint8)
	img = Image.fromarray(arr_hist_byte)
	img.show()
	
	plt.plot(l_middle_x, l_middle_y, ".", color="#FF00FF", markersize=0.5)

	plt.plot(arr_x, arr_y, "o", color='#0000FF', markersize=2.)
	plt.show(block=False)

	arr_x_slice = min_x + np.arange(0, bins) * (diff_x / bins)
	arr_y_slice = np.mean(arr_hist[bins//2-5:bins//2+5], 0)

	plt.figure()
	plt.plot(arr_x_slice, arr_y_slice, '.-', color='#0000FF', markersize=1., linewidth=0.5)
	plt.show(block=False)
