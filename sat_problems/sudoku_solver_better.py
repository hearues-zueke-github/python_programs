#! /usr/bin/env -S /usr/bin/time python3.11 -i

# -*- coding: utf-8 -*-

import os
import random
import sys

from pysat.solvers import Glucose3

from copy import deepcopy

from time import time
from functools import reduce

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

from pprint import pprint
from typing import List

from cnf import CNF

if __name__ == "__main__":
	cnf_obj = CNF()
	
	l_v = cnf_obj.get_new_variables(amount=9*9*9) # 9 rows, 9 columns, 9 numbers

	# add restriction for each cell were only one and only one number is allowed! (1 to 9)
	for row in range(0, 9):
		for column in range(0, 9):
			cnfs_row = CNF.get_tseytin_only_one_true(l_v=[row*9*9+column*9+num+1 for num in range(0, 9)])
			cnf_obj.extend_cnfs(cnfs_row)

	# add restriction for each row for only one and only one same number allowed!
	for row in range(0, 9):
		for num in range(0, 9):
			cnfs_row = CNF.get_tseytin_only_one_true(l_v=[row*9*9+column*9+num+1 for column in range(0, 9)])
			cnf_obj.extend_cnfs(cnfs_row)

	# # add restriction for each column for only one and only one same number allowed!
	# for column in range(0, 9):
	# 	for num in range(0, 9):
	# 		cnfs_row = CNF.get_tseytin_only_one_true(l_v=[row*9*9+column*9+num+1 for row in range(0, 9)])
	# 		cnf_obj.extend_cnfs(cnfs_row)

	# add restriction for each 3x3 square only one and only one same number allowed!
	for row_start in range(0, 9, 3):
		for column_start in range(0, 9, 3):
			for num in range(0, 9):
				cnfs_row = CNF.get_tseytin_only_one_true(l_v=[row*9*9+column*9+num+1 for row in range(row_start, row_start+3) for column in range(column_start, column_start+3)])
				cnf_obj.extend_cnfs(cnfs_row)

	# add a special restriciton for each main diagonal for having only and only one same number allowed!
	# main diagonal
	for num in range(0, 9):
		for idx_1 in range(0, 9):
			cnfs_diag_main = CNF.get_tseytin_only_one_true(l_v=[idx_2*9*9+((idx_1+idx_2)%9)*9+num+1 for idx_2 in range(0, 9)])
			cnf_obj.extend_cnfs(cnfs_diag_main)

	# other diagonal
	for num in range(0, 9):
		for idx_1 in range(0, 9):
			cnfs_diag_other = CNF.get_tseytin_only_one_true(l_v=[idx_2*9*9+((idx_1-idx_2)%9)*9+num+1 for idx_2 in range(0, 9)])
			cnf_obj.extend_cnfs(cnfs_diag_other)

	l_t_row_col_num = [
		(1, 2, 5),
		(2, 3, 8),
		(6, 6, 1),
		(1, 3, 7),
		(2, 2, 2),
		(7, 8, 9),
		(8, 1, 3),
		(9, 1, 2),
		(8, 2, 5),
		(3, 1, 4),
	]

	for row, col, num in l_t_row_col_num:
		cnf_obj.extend_cnfs([((row-1)*9*9+(col-1)*9+num, )])

	models_amount = 10000
	with Glucose3(bootstrap_with=cnf_obj.cnfs) as m:
		print(m.solve())
		print(list(m.get_model()))
		models = [(i, m) for m, i in zip(m.enum_models(), range(0, models_amount))]

	for m in models:
		l = m[1]

		arr = np.zeros((9, 9), dtype=np.int64)
		for var in l:
			if var > 0:
				var -= 1
				y = var//9//9
				x = (var//9)%9
				val = var%9

				if arr[y, x] != 0:
					assert False

				arr[y, x] = val + 1

		# check if arr is a valid sudoku field!

		# check, if each digit is comming 9 times
		assert np.all([np.sum(arr == i) == 9 for i in range(1, 10)])
		
		# # check, if each digit is occuring once per row
		# assert np.all([np.all(np.sum(arr == i, axis=0) == 1) for i in range(1, 10)])
		
		# # check, if each digit is occuring once per column
		# assert np.all([np.all(np.sum(arr == i, axis=1) == 1) for i in range(1, 10)])
		
		# # check, if each digit is occuring once per 3x3 square
		# assert np.all([np.sum(arr[y:y+3, x:x+3] == i) == 1 for i in range(1, 10) for y in range(0, 9, 3) for x in range(0, 9, 3)])

		print(f"arr:\n{arr}")
		print("It is working!!!")

	print(f"len(models): {len(models)}")
