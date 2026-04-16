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

decimal.getcontext().prec = 100

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


class Point:
	idx_counter = 0

	def __init__(self, x, y):
		self.idx = self.__class__.idx_counter + 0
		self.__class__.idx_counter += 1
		self.x = x
		self.y = y


	def __hash__(self):
		return hash(("Point", self.x, self.y))


	def __eq__(self, other):
		if not isinstance(other, Point):
			return False

		return (self.x, self.y) == (other.x, other.y)


	def __str__(self):
		return f"Point(idx={self.idx}, x={self.x}, y={self.y})"


	def __repr__(self):
		return str(self)


	def __mul__(self, other):
		if isinstance(other, Point):
			assert self != other
			return Line(point_1=self, point_2=other)
		elif isinstance(other, int):
			return Point(x=self.x*other, y=self.y*other)
		elif isinstance(other, float):
			return Point(x=self.x*Decimal(other), y=self.y*Decimal(other))

		assert False


	def __add__(self, other):
		assert isinstance(other, Vector)

		return Point(x=self.x+other.x, y=self.y+other.y)


	def __sub__(self, other):
		assert isinstance(other, Point)
		
		return Vector(x=self.x-other.x, y=self.y-other.y)


	def __lt__(self, other):
		assert isinstance(other, Point)

		if self.x < other.x:
			return True
		elif self.x > other.x:
			return False

		if self.y < other.y:
			return True
		elif self.y > other.y:
			return False

		return False


class Vector:
	idx_counter = 0

	def __init__(self, x, y):
		self.idx = self.__class__.idx_counter + 0
		self.__class__.idx_counter += 1
		self.x = x
		self.y = y


	def __hash__(self):
		return hash(("Vector", self.x, self.y))


	def __eq__(self, other):
		if not isinstance(other, Vector):
			return False

		return (self.x, self.y) == (other.x, other.y)


	def __str__(self):
		return f"Vector(idx={self.idx}, x={self.x}, y={self.y})"


	def __repr__(self):
		return str(self)


	def __abs__(self):
		return (self.x**Decimal(2) + self.y**Decimal(2))**Decimal("0.5")


	def __div__(self, other):
		if isinstance(other, Decimal):
			return Vector(x=self.x/other, y=self.y/other)

		assert False


	def __mul__(self, other):
		if isinstance(other, int):
			return Vector(x=self.x*other, y=self.y*other)
		if isinstance(other, float):
			return Vector(x=self.x*Decimal(other), y=self.y*Decimal(other))

		assert False


	def __add__(self, other):
		assert isinstance(other, Vector)
		
		return Vector(x=self.x+other.x, y=self.y+other.y)


	def __sub__(self, other):
		assert isinstance(other, Vector)
		
		return Vector(x=self.x-other.x, y=self.y-other.y)


	def deg_90(self):
		return Vector(x=self.y, y=-self.x)


	def dot(self, other):
		if isinstance(other, Vector):
			return self.x * other.x + self.y * other.y

		assert False


class Line:
	idx_counter = 0

	def __init__(self, point_1, point_2):
		self.idx = self.__class__.idx_counter + 0
		self.__class__.idx_counter += 1
		self.point_1 = point_1
		self.point_2 = point_2


	def __hash__(self):
		return hash(("Line", self.point_1, self.point_2))


	def __eq__(self, other):
		assert isinstance(other, Line)
		return (self.point_1, self.point_2) == (other.point_1, other.point_2)


	def __lt__(self, other):
		assert isinstance(other, Line)

		if self.point_1 < other.point_1:
			return True
		elif self.point_1 > other.point_1:
			return False

		if self.point_2 < other.point_2:
			return True
		elif self.point_2 > other.point_2:
			return False

		return False


	def __str__(self):
		return f"Line(idx={self.idx}, point_1={self.point_1}, point_2={self.point_2})"


	def __repr__(self):
		return str(self)


	def get_vec(self):
		return self.point_2 - self.point_1


	def __abs__(self):
		return abs(self.get_vec())


	def get_cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
		k1 = ((y1-y3)*(x4-x3) - (x1-x3)*(y4-y3)) / ((x2-x1)*(y4-y3) - (y2-y1)*(x4-x3))

		x5 = x1 + (x2-x1) * k1
		y5 = y1 + (y2-y1) * k1

		return Point(x=x5, y=y5)


	def __mul__(self, other):
		if isinstance(other, Line):
			assert self != other

			point_1 = self.point_1
			point_2 = self.point_2
			point_3 = other.point_1
			point_4 = other.point_2

			return Line.get_cross_point(
				x1=point_1.x, y1=point_1.y,
				x2=point_2.x, y2=point_2.y,
				x3=point_3.x, y3=point_3.y,
				x4=point_4.x, y4=point_4.y,
			)
		elif isinstance(other, Point):
			point_1 = self.point_1
			point_2 = self.point_2
			point_3 = other
			point_4 = other + self.get_vec().deg_90()

			return Line.get_cross_point(
				x1=point_1.x, y1=point_1.y,
				x2=point_2.x, y2=point_2.y,
				x3=point_3.x, y3=point_3.y,
				x4=point_4.x, y4=point_4.y,
			)

		assert False


	def __contains__(self, other):
		if isinstance(other, Point):
			point = self * other
			return abs(other - point) < Decimal("1e-40")

		assert False


if __name__ == '__main__':
	x1, y1 = (Decimal('0.0'), Decimal('0.0'))
	x2, y2 = (Decimal('0.0'), Decimal('1.0'))
	x3, y3 = (Decimal('2.0'), Decimal('0.0'))
	x4, y4 = (Decimal('1.0'), Decimal('1.0'))

	point_1 = Point(x1, y1)
	point_2 = Point(x2, y2)
	point_3 = Point(x3, y3)
	point_4 = Point(x4, y4)

	# line_1_2 = point_1 * point_2
	# line_3_4 = point_3 * point_4
	# line_1_3 = point_1 * point_3
	# line_2_4 = point_2 * point_4
	
	# assert point_1 in line_1_2
	# assert point_2 in line_1_2
	# assert not point_3 in line_1_2
	# assert not point_4 in line_1_2

	# point_5 = line_1_2 * line_3_4
	
	# line_1_4 = point_1 * point_4
	# line_2_3 = point_2 * point_3

	# point_6 = line_1_4 * line_2_3

	# line_5_6 = point_5 * point_6

	# point_7 = line_5_6 * line_2_4
	# point_8 = line_5_6 * line_1_3
	

	# s_next_points = {point_1, point_2, point_3, point_4}
	# s_next_lines = set()

	# d_line_to_s_point = {}
	# d_point_to_s_line = {s_next_points.pop(): set()}

	# while s_next_points:
	# 	point_now = s_next_points.pop()

	# 	s_line_of_point_now = set()
	# 	for point, s_line in d_point_to_s_line.items():
	# 		## check if point_now and point are on the same line already or not

	# 		are_on_the_line_already = False
	# 		for s_point in d_line_to_s_point.values():
	# 			if point_now in s_point and point in s_point:
	# 				are_on_the_line_already = True
	# 				break

	# 		if are_on_the_line_already:
	# 			continue

	# 		line_new = Line(point_1=point, point_2=point_now)
	# 		s_line.add(line_new)
	# 		s_line_of_point_now.add(line_new)

	# 		s_next_lines.add(line_new)
	# 		assert not line_new in d_line_to_s_point
	# 		d_line_to_s_point[line_new] = {point, point_now}

	# 	d_point_to_s_line[point_now] = s_line_of_point_now


	l_s_next_points = []
	l_s_next_lines = []

	l_len_s_next_points = []
	l_len_s_next_lines = []

	s_points = {point_1, point_2, point_3, point_4}
	s_lines = set()

	s_t_point = set()
	s_t_line = set()

	d_line_to_s_point = {}
	d_point_to_s_line = {}

	l_points = list(s_points)
	point = l_points[0]

	min_x = point.x
	min_y = point.y
	max_x = point.x
	max_y = point.y

	for point in l_points[1:]:
		x = point.x
		y = point.y

		if min_x > x:
			min_x = x

		if min_y > y:
			min_y = y

		if max_x < x:
			max_x = x

		if max_y < y:
			max_y = y

	print('Stats:')
	print(f'- min_x: {min_x}')
	print(f'- min_y: {min_y}')
	print(f'- max_x: {max_x}')
	print(f'- max_y: {max_y}')

	for iter_num in range(1, 5):
		s_next_points = set()
		s_next_lines = set()

		# print('Calc new Lines:')
		for point_1, point_2 in itertools.combinations(sorted(s_points), 2):
			if (point_1, point_2) in s_t_point or (point_2, point_1) in s_t_point:
				continue

			s_t_point.add((point_1, point_2))

			if point_1 in d_point_to_s_line and point_2 in d_point_to_s_line:
				if d_point_to_s_line[point_1] & d_point_to_s_line[point_2]:
					continue

			line_new = Line(point_1=point_1, point_2=point_2)

			if point_1 not in d_point_to_s_line:
				d_point_to_s_line[point_1] = set()
			if point_2 not in d_point_to_s_line:
				d_point_to_s_line[point_2] = set()

			d_point_to_s_line[point_1].add(line_new)
			d_point_to_s_line[point_2].add(line_new)

			d_line_to_s_point[line_new] = {point_1, point_2}

			s_lines.add(line_new)
			s_next_lines.add(line_new)

			# print(f'point_1: {point_1}')
			# print(f'point_2: {point_2}')
			# print(f'line_new: {line_new}')
			# print("------------")

			if len(s_next_lines) > 30:
				break

		# print('Calc new Points:')
		for line_1, line_2 in itertools.combinations(sorted(s_lines), 2):
			if (line_1, line_2) in s_t_line or (line_2, line_1) in s_t_line:
				continue

			s_t_line.add((line_1, line_2))

			if d_line_to_s_point[line_1] & d_line_to_s_point[line_2]:
				continue

			# TODO: add a check, if the lines are very parallel, otherwise there exists a crossing point
			try:
				vec_1 = line_1.get_vec()
				vec_2 = line_2.get_vec()
				dot_product = vec_1.dot(vec_2) / abs(vec_1) / abs(vec_2)
				# print(f'dot_product: {dot_product}')
				if Decimal("1") - abs(dot_product) < Decimal("1e-50"):
					# print(f'Line line_1 and line_2 are nearly parallel!')
					continue

				point_new = line_1 * line_2
			except:
				continue

			if line_1 not in d_line_to_s_point:
				d_line_to_s_point[line_1] = set()
			if line_2 not in d_line_to_s_point:
				d_line_to_s_point[line_2] = set()

			d_line_to_s_point[line_1].add(point_new)
			d_line_to_s_point[line_2].add(point_new)

			d_point_to_s_line[point_new] = {line_1, line_2}

			s_points.add(point_new)
			s_next_points.add(point_new)

			# print(f'line_1: {line_1}')
			# print(f'line_2: {line_2}')
			# print(f'point_new: {point_new}')
			# print("------------")

			if len(s_next_points) > 200:
				break

		print(f'iter_num: {iter_num}, len(s_next_points): {len(s_next_points)}, len(s_next_lines): {len(s_next_lines)}')
		print(f'- len(s_points): {len(s_points)}, len(s_lines): {len(s_lines)}')
		# l_s_next_points.append(s_next_points)
		# l_s_next_lines.append(s_next_lines)
		l_len_s_next_points.append(len(s_next_points))
		l_len_s_next_lines.append(len(s_next_lines))

		for point in s_points:
			x = point.x
			y = point.y

			if min_x > x:
				min_x = x

			if min_y > y:
				min_y = y

			if max_x < x:
				max_x = x

			if max_y < y:
				max_y = y

		print('Stats:')
		print(f'- min_x: {min_x}')
		print(f'- min_y: {min_y}')
		print(f'- max_x: {max_x}')
		print(f'- max_y: {max_y}')


	iterator = iter(itertools.combinations(sorted(s_points), 2))
	point_1, point_2 = next(iterator)

	min_distance = abs(point_1 - point_2)
	max_distance = min_distance

	for point_1, point_2 in list(iterator):
		distance = abs(point_1 - point_2)

		if min_distance > distance:
			min_distance = distance

		if max_distance < distance:
			max_distance = distance

	print(f'min_distance: {min_distance}')
	print(f'max_distance: {max_distance}')

	# get the most basic infos about min, max values, clustering, most distance, smallest distance of points etc.
