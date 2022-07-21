#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import ast
import cv2
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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

class Field2d:
	__slot__ = ["frame", "height", "width", "d_var_arr"]

	def __init__(self, frame, height, width):
		self.frame = frame
		self.height = height
		self.width = width
		
		self.arr_with_frame = np.zeros((self.height+self.frame*2, self.width+self.frame*2), dtype=np.uint8)

		self.init_d_var_arr()


	def init_d_var_arr(self):
		l_up = [('u', i, -i, 0) for i in range(frame, 0, -1)]
		l_down = [('d', i, i, 0) for i in range(1, frame+1)]
		l_left = [('l', i, 0, -i) for i in range(frame, 0, -1)]
		l_right = [('r', i, 0, i) for i in range(1, frame+1)]

		d_var_dir = {"n": (0, 0)}

		for var, amount, dir_y, dir_x in l_up+l_down+l_left+l_right:
			d_var_dir[var*amount] = (dir_y, dir_x)
			d_var_dir[var+str(amount)] = (dir_y, dir_x)

		for var_1, amount_1, dir_y_1, dir_x_1 in l_up+l_down:
			for var_2, amount_2, dir_y_2, dir_x_2 in l_left+l_right:
				d_var_dir[var_1*amount_1+var_2*amount_2] = (dir_y_1+dir_y_2, dir_x_1+dir_x_2)
				d_var_dir[var_1+str(amount_1)+var_2+str(amount_2)] = (dir_y_1+dir_y_2, dir_x_1+dir_x_2)

		self.d_var_arr = {}
		for var, (dir_y, dir_x) in d_var_dir.items():
			self.d_var_arr[var] = self.arr_with_frame[frame+dir_y:frame+dir_y+height, frame+dir_x:frame+dir_x+width]


	def set_points(self, arr_y: np.ndarray, arr_x: np.ndarray):
		self.arr_with_frame[arr_y+self.frame, arr_x+self.frame] = 1
		self.redefine_frame()


	def set_arr_no_frame(self, arr_no_frame: np.ndarray):
		assert arr_no_frame.shape == (self.height, self.width)
		assert arr_no_frame.dtype == self.arr_with_frame.dtype
		self.arr_with_frame[self.frame:-self.frame, self.frame:-self.frame] = arr_no_frame
		self.redefine_frame()


	def redefine_frame(self):
		self.arr_with_frame[:, :self.frame] = self.arr_with_frame[:, -self.frame*2:-self.frame]
		self.arr_with_frame[:, -self.frame:] = self.arr_with_frame[:, self.frame:self.frame*2]
		self.arr_with_frame[:self.frame, :] = self.arr_with_frame[-self.frame*2:-self.frame, :]
		self.arr_with_frame[-self.frame:, :] = self.arr_with_frame[self.frame:self.frame*2, :]


	def get_arr_no_frame(self):
		return self.arr_with_frame[self.frame:-self.frame, self.frame:-self.frame].copy()


	def define_functions(self, func_str: str):
		self.d_func_other = {}
		func_str_other = """
def inv(a): return (~(a.astype(np.bool)))+np.uint8(0)
"""
		# numpy as np is needed for the global dict
		exec(func_str_other, {'np': np}, self.d_func_other)
		self.d_func = {}

		exec(func_str, {key: val for key, val in field_2d.d_var_arr.items()} | self.d_func_other, self.d_func)
		assert 'l_func' in self.d_func
		try:
			for d in self.d_func['l_func']:
				assert isinstance(d, dict)
				func_name = d['func_name']
				func = d['func']
				arr = func()
				assert arr.shape == (self.height, self.width)
				assert arr.dtype == self.arr_with_frame.dtype
		except:
			print(f"Problem with the function with the name '{func_name}'")
			traceback.print_exc()
			assert False

		self.l_func = self.d_func['l_func']


def plot_all_field_in_one_picture(l_arr: List[np.ndarray], rows: int, columns: int, border_width: int) -> np.ndarray:
	assert all([isinstance(arr, np.ndarray) for arr in l_arr])
	shape = l_arr[0].shape
	# check, if all arr's are the same size!
	assert all([arr.shape == shape for arr in l_arr])

	height, width = shape

	if len(l_arr) > rows * columns:
		l_arr = l_arr[:rows*columns]

	pix_height = rows*height + (rows+1)*border_width
	pix_width = columns*width + (columns+1)*border_width
	pix_comb = np.zeros((pix_height, pix_width), dtype=np.uint8)
	pix_comb[:] = 128 # this is the default color of the frames
	for i, arr in enumerate(l_arr, 0):
		y_off = border_width+(height+border_width)*(i // columns)
		x_off = border_width+(width+border_width)*(i % columns)
		pix_comb[y_off:y_off+height, x_off:x_off+width] = arr*255

	return pix_comb
	# img = Image.fromarray(pix_comb)
	# img.show()
	# img.save("/tmp/example_pic.png")


if __name__ == '__main__':
	frame = 4
	
	height = 90
	width = 120

	field_2d = Field2d(frame=frame, height=height, width=width)

	n = int(height*width*0.56)
	arr_y = np.random.randint(0, height, (n, ))
	arr_x = np.random.randint(0, width, (n, ))
	
	field_2d.set_points(arr_y=arr_y, arr_x=arr_x)

	# arr_no_frame_1 = field_2d.get_arr_no_frame()
	# field_2d.set_arr_no_frame(arr_no_frame=arr_no_frame_1)
	# arr_no_frame_2 = field_2d.get_arr_no_frame()
	# assert np.all(arr_no_frame_1 == arr_no_frame_2)
	# sys.exit()

	# example, how to define a function for the next field
	func_str = """
def func_0(): return inv(l1) # move to the right the whole field
# def func_0(): return r4 | r2 & inv(l1) # move to the right the whole field
def func_1(): return u1 & n | d1 & r1 | n & l1 & r1
l_func=[
	{'func_name': 'func_0', 'func': func_0},
	# {'func_name': 'func_1', 'func': func_1},
]
"""

	field_2d.define_functions(func_str=func_str)

	l_arr_no_frame = [field_2d.get_arr_no_frame()]

	# func = field_2d.l_func[0]['func']

	l_func = field_2d.l_func
	for iter_round in range(0, 300):
		func = l_func[iter_round % len(l_func)]['func']

		arr_no_frame = func().copy()
		field_2d.set_arr_no_frame(arr_no_frame=arr_no_frame)
		
		arr_no_frame_2 = field_2d.get_arr_no_frame()
		assert np.all(arr_no_frame == arr_no_frame_2)

		is_dup_found = False
		for i, arr in enumerate(l_arr_no_frame):
			if np.all(arr == arr_no_frame):
				is_dup_found = True
				break

		l_arr_no_frame.append(arr_no_frame)
		if is_dup_found:
			print(f"i_dup: {i}, len(l_arr_no_frame): {len(l_arr_no_frame)}")
			break

	# # this can be used for animating the images
	# while True:
	# 	for arr in l_arr_no_frame:
	# 		cv2.imshow('image', arr*255); cv2.waitKey(200)

	pix_comb = plot_all_field_in_one_picture(l_arr=l_arr_no_frame, rows=5, columns=4, border_width=2)
	Image.fromarray(pix_comb).save("/tmp/example_pic.png")

	# pix_height = len(l_arr_no_frame)*height + (len(l_arr_no_frame)+1)
	# pix_width = width + 1+1
	# pix_comb_y = np.zeros((pix_height, pix_width), dtype=np.uint8)
	# pix_comb_y[:] = 128
	# for i, arr in enumerate(l_arr_no_frame, 0):
	# 	y_off = 1+(height+1)*i
	# 	x_off = 1+(width+1)*0
	# 	pix_comb_y[y_off:y_off+height, x_off:x_off+width] = arr*255

	# img = Image.fromarray(pix_comb_y)
	# # img.show()
	# img.save("/tmp/example_pic.png")

	# pix_comb_y_diff = np.zeros((pix_height, pix_width), dtype=np.uint8)
	# pix_comb_y_diff[:] = 128
	# for i, (arr1, arr2) in enumerate(zip(l_arr_no_frame, l_arr_no_frame[1:]+l_arr_no_frame[:1]), 0):
	# 	y_off = 1+(height+1)*i
	# 	x_off = 1+(width+1)*0
	# 	pix_comb_y_diff[y_off:y_off+height, x_off:x_off+width] = (arr1^arr2)*255

	# img_diff = Image.fromarray(pix_comb_y_diff)
	# # img_diff.show()
	# img_diff.save("/tmp/example_pic_diff.png")
