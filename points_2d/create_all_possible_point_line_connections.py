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
import sys
import time
import traceback

import numpy as np # need installation from pip
import pandas as pd # need installation from pip
import multiprocessing as mp

import matplotlib.pyplot as plt # need installation from pip

from collections import defaultdict
from copy import deepcopy, copy
from decimal import Decimal
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

decimal.getcontext().prec = 50

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

def get_cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
	k1 = ((y1-y3)*(x4-x3) - (x1-x3)*(y4-y3)) / ((x2-x1)*(y4-y3) - (y2-y1)*(x4-x3))

	x5 = x1 + (x2-x1) * k1
	y5 = y1 + (y2-y1) * k1

	return x5, y5


class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y


	def __hash__(self):
		return hash((self.x, self.y))


	def __eq__(self, other):
		return (self.x, self.y) == (other.x, other.y)


	def __str__(self):
		return f"Point(x={self.x}, y={self.y})"


class Line:
	def __init__(self, point_1, point_2):
		self.point_1 = point_1
		self.point_2 = point_2


	def __hash__(self):
		return hash((self.point_1, self.point_2))


	def __eq__(self, other):
		return (self.point_1, self.point_2) == (other.point_1, other.point_2)


	def __str__(self):
		return f"Line(point_1={self.point_1}, point_2={self.point_2})"


def get_cross_point_from_lines(line_1, line_2):
	point_1 = line_1.point_1
	point_2 = line_1.point_2
	point_3 = line_2.point_1
	point_4 = line_2.point_2

	return Point(*get_cross_point(
		x1=point_1.x, y1=point_1.y,
		x2=point_2.x, y2=point_2.y,
		x3=point_3.x, y3=point_3.y,
		x4=point_4.x, y4=point_4.y,
	))


if __name__ == '__main__':
	# x1, y1 = (Decimal('2.0'), Decimal('2.0'))
	# x2, y2 = (Decimal('8.0'), Decimal('5.0'))
	# x3, y3 = (Decimal('3.0'), Decimal('5.0'))
	# x4, y4 = (Decimal('9.0'), Decimal('3.0'))

	x1, y1 = (Decimal('0.0'), Decimal('0.0'))
	x2, y2 = (Decimal('0.0'), Decimal('1.0'))
	x3, y3 = (Decimal('2.0'), Decimal('0.0'))
	x4, y4 = (Decimal('1.0'), Decimal('1.0'))

	# k1 = math.sqrt((x2-x1)**2 + (y2-y1)**2) * ((y1-y3)*(x4-x3) - (x1-x3)*(y4-y3)) / ((x2-x1)*(y4-y3) - (y2-y1)*(x4-x3))

	print(f'x1: {x1}, y1: {y1}')
	print(f'x2: {x2}, y2: {y2}')
	print(f'x3: {x3}, y3: {y3}')
	print(f'x4: {x4}, y4: {y4}')
	# print(f'k1: {k1}')

	x5, y5 = get_cross_point(x1, y1, x2, y2, x3, y3, x4, y4)
	print(f'x5: {x5}, y5: {y5}')

	p1 = (Decimal("2"), Decimal("4"))

	point_1 = Point(x1, y1)
	point_2 = Point(x2, y2)
	point_3 = Point(x3, y3)
	point_4 = Point(x4, y4)

	line_1_2 = Line(point_1, point_2)
	line_3_4 = Line(point_3, point_4)
	line_1_3 = Line(point_1, point_3)
	line_2_4 = Line(point_2, point_4)
	
	point_5 = get_cross_point_from_lines(line_1_2, line_3_4)
	
	line_1_4 = Line(point_1, point_4)
	line_2_3 = Line(point_2, point_3)

	point_6 = get_cross_point_from_lines(line_1_4, line_2_3)

	line_5_6 = Line(point_5, point_6)

	point_7 = get_cross_point_from_lines(line_5_6, line_2_4)
	point_8 = get_cross_point_from_lines(line_5_6, line_1_3)

	print(f'point_1: {point_1}')
	print(f'point_2: {point_2}')
	print(f'point_3: {point_3}')
	print(f'point_4: {point_4}')
	print('--------')
	print(f'line_1_2: {line_1_2}')
	print(f'line_3_4: {line_3_4}')
	print(f'line_1_3: {line_1_3}')
	print(f'line_2_4: {line_2_4}')
	print('--------')
	print(f'point_5: {point_5}')
	print('--------')
	print(f'line_1_4: {line_1_4}')
	print(f'line_2_3: {line_2_3}')
	print('--------')
	print(f'point_6: {point_6}')
	print('--------')
	print(f'line_5_6: {line_5_6}')
	print('--------')
	print(f'point_7: {point_7}')
	print(f'point_8: {point_8}')
