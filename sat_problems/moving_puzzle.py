#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

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
	for game_nr in range(0, 10000):
		print(f"game_nr: {game_nr}")

		cnf_obj = CNF()
		
		rows = 3
		cols = 4

		arr_field_pos = np.empty((rows, cols, 2), dtype=np.int64)
		arr_field_pos[:, :, 0] = np.arange(0, rows).reshape((rows, 1))
		arr_field_pos[:, :, 1] = np.arange(0, cols).reshape((1, cols))

		arr_field_pos_left_to_right = np.roll(arr_field_pos, 1, 1)
		arr_field_pos_right_to_left = np.roll(arr_field_pos, -1, 1)

		arr_field_pos_t = arr_field_pos.transpose(1, 0, 2)
		arr_field_pos_up_to_down = np.roll(arr_field_pos_t, 1, 1)
		arr_field_pos_down_to_up = np.roll(arr_field_pos_t, -1, 1)

		l_arr_row_nr_to_row = (
			[(arr_field_pos, i, arr_field_pos_left_to_right[i]) for i in range(0, rows)]+
			[(arr_field_pos, i, arr_field_pos_right_to_left[i]) for i in range(0, rows)]+
			[(arr_field_pos_t, i, arr_field_pos_up_to_down[i]) for i in range(0, cols)]+
			[(arr_field_pos_t, i, arr_field_pos_down_to_up[i]) for i in range(0, cols)]
		)

		# min max neded turns to solve the puzzle
		# rows=2;cols=2;amount_turns=4
		# rows=2;cols=3;amount_turns=7
		# rows=2;cols=4;amount_turns=8
		# rows=2;cols=5;amount_turns=12

		# rows=3;cols=3;amount_turns=8
		
		# symetrics
		# rows=3;cols=2;amount_turns=7

		amount_turns = 9
		amount_directions = 1+(rows+cols)*2

		amount_nums = rows * cols
		amount_v_row = amount_nums * cols
		amount_v = amount_nums**2

		amount_random_moves = 30
		l_move_random = np.random.randint(0, len(l_arr_row_nr_to_row), (amount_random_moves, ))

		# failed for: rows: 5, cols: 2, amount_turns: 11, amount_random_moves: 30
		# l_move_random = [3, 12, 10, 13, 0, 5, 0, 8, 1, 9, 1, 10, 4, 8, 9, 6, 13, 10, 3, 5, 11, 3, 1, 12, 1, 5, 2, 13, 9, 5]

		arr_field = np.arange(1, amount_nums+1).reshape((rows, cols))

		for move in l_move_random:
			arr_field_pos_temp, i, arr_row_to = l_arr_row_nr_to_row[move]
			arr_row_from = arr_field_pos_temp[i]

			arr_field[tuple(arr_row_to.transpose(1, 0).tolist())] = arr_field[tuple(arr_row_from.transpose(1, 0).tolist())]
		# sys.exit()

		l_l_v_field = [cnf_obj.get_new_variables(amount=amount_v) for _ in range(0, amount_turns + 1)]
		l_d_field = [{(y, x): l_v_field[y*amount_v_row+x*amount_nums:y*amount_v_row+(x+1)*amount_nums] for y in range(0, rows) for x in range(0, cols)} for l_v_field in l_l_v_field]

		l_l_v_dir = [cnf_obj.get_new_variables(amount=amount_directions) for _ in range(0, amount_turns)]
		# l_v_dir_1 = cnf_obj.get_new_variables(amount=1+(rows+cols)*2)

		# set the first field with the default values!
		l_v_field_default = l_l_v_field[0]
		s_non_neg_v = {l_v_field_default[i] for i in range(0, amount_v, amount_nums+1)}
		cnf_obj.extend_cnfs(l_t_v=[(v, ) if v in s_non_neg_v else (-v, ) for v in l_v_field_default])

		# set the goal for the finish field
		l_v_field_last = l_l_v_field[-1]
		cnf_obj.extend_cnfs(l_t_v=[
			(l_v_field_last[row*amount_v_row + col*amount_nums + (arr_field[row, col] - 1)], ) for row in range(0, rows) for col in range(0, cols)
			# (l_v_field_last[2*amount_v_row + 1*amount_nums + (4 - 1)], ),
		])
		# cnf_obj.extend_cnfs(l_t_v=[
		# 	(l_v_field_last[0*amount_v_row + 0*amount_nums + (6 - 1)], ),
		# 	(l_v_field_last[0*amount_v_row + 1*amount_nums + (5 - 1)], ),
		# 	(l_v_field_last[0*amount_v_row + 2*amount_nums + (8 - 1)], ),
		# 	(l_v_field_last[1*amount_v_row + 0*amount_nums + (2 - 1)], ),
		# 	(l_v_field_last[1*amount_v_row + 1*amount_nums + (1 - 1)], ),
		# 	(l_v_field_last[1*amount_v_row + 2*amount_nums + (7 - 1)], ),
		# 	(l_v_field_last[2*amount_v_row + 0*amount_nums + (3 - 1)], ),
		# 	# (l_v_field_last[2*amount_v_row + 1*amount_nums + (4 - 1)], ),
		# ])

		# use only one direction of all the directions
		# cnfs_dir_1 = CNF.get_tseytin_only_one_true(l_v=l_v_dir_1)
		for l_v_dir in l_l_v_dir:
			cnfs_dir = CNF.get_tseytin_only_one_true(l_v=l_v_dir)
			cnf_obj.extend_cnfs(l_t_v=cnfs_dir)


		# all other moves
		for l_v_dir, l_v_field_1, l_v_field_2, d_field_1, d_field_2 in zip(l_l_v_dir, l_l_v_field[:-1], l_l_v_field[1:], l_d_field[:-1], l_d_field[1:]):
			# no move
			v_move = l_v_dir[0]
			cnfs_no_move = []
			for i in range(0, len(l_v_field_1)):
				v1 = l_v_field_1[i]
				v2 = l_v_field_2[i]
				cnfs_no_move.extend([(v1, -v2, -v_move), (-v1, v2, -v_move)])
			cnf_obj.extend_cnfs(l_t_v=cnfs_no_move)
			
			for v_move, (arr_field_pos_temp, row_i, arr_row) in zip(l_v_dir[1:], l_arr_row_nr_to_row):
				cnfs_move = []

				arr_row_prev = arr_field_pos_temp[row_i]
				for arr1, arr2 in zip(arr_row_prev, arr_row):
					t1 = tuple(arr1.tolist())
					t2 = tuple(arr2.tolist())
					for v1, v2 in zip(d_field_1[t1], d_field_2[t2]):
						cnfs_move.extend([(v1, -v2, -v_move), (-v1, v2, -v_move)])

				for j in range(0, arr_field_pos_temp.shape[0]):
					if j == row_i:
						continue

					for arr in arr_field_pos_temp[j]:
						t = tuple(arr.tolist())
						for v1, v2 in zip(d_field_1[t], d_field_2[t]):
							cnfs_move.extend([(v1, -v2, -v_move), (-v1, v2, -v_move)])

				cnf_obj.extend_cnfs(l_t_v=cnfs_move)

		models_amount = 1
		with Glucose3(bootstrap_with=cnf_obj.cnfs) as m:
			is_solvable = m.solve()
			print(f"is_solvable: {is_solvable}")

			try:
				assert is_solvable
			except:
				print(f"rows: {rows}, cols: {cols}, amount_turns: {amount_turns}, amount_random_moves: {amount_random_moves}")
				print(f"l_move_random.tolist(): {l_move_random.tolist()}")
				assert False and "Not solvable!"
			first_model = list(m.get_model())
			print(f"first_model: {first_model}")
			print()

			models = [(i, m) for m, i in zip(m.enum_models(), range(0, models_amount))]

		is_any_any_move_0 = False
		l_l_move = []
		for m in models:
			l = m[1]

			l_arr = [np.zeros((rows, cols), dtype=np.int64) for _ in range(0, amount_turns + 1)]
			for arr, l_v_field in zip(l_arr, l_l_v_field):
				for i in l_v_field:
					var = l[i-1]
					if var > 0:
						var -= 1
						y = (var%amount_v)//amount_v_row
						x = (var%amount_v_row)//amount_nums
						val = var%amount_nums

						if arr[y, x] != 0:
							assert False

						arr[y, x] = val + 1

			l_temp = l[amount_v*(amount_turns+1):]
			l_l_dir = [l_temp[amount_directions*i:amount_directions*(i+1)] for i in range(0, amount_turns)]

			l_move = np.array([np.where(np.array(l_dir) > 0)[0] for l_dir in l_l_dir]).flatten().tolist()

			l_l_move.append(l_move)

			print(f"m_i: {m[0]}")
			# print(f"- l_move: {l_move}")
			# for i, arr in enumerate(l_arr, 1):
			# 	print(f"- i: {i}, arr:\n{arr}")

			is_any_move_0 = any([0 in l_move for l_move in l_l_move])
			if is_any_move_0:
				is_any_any_move_0 = True
			print(f"is_any_move_0: {is_any_move_0}")
			
			print()
		
		print(f"is_any_any_move_0: {is_any_any_move_0}")
