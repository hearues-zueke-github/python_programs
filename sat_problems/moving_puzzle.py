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

def solve_a_mixed_moving_puzzle(amount_turns, arr_field):
	cnf_obj = CNF()

	l_l_v_field = [cnf_obj.get_new_variables(amount=amount_v) for _ in range(0, amount_turns + 1)]
	l_d_field = [{(y, x): l_v_field[y*amount_v_row+x*amount_nums:y*amount_v_row+(x+1)*amount_nums] for y in range(0, rows) for x in range(0, cols)} for l_v_field in l_l_v_field]

	l_l_v_dir = [cnf_obj.get_new_variables(amount=amount_directions) for _ in range(0, amount_turns)]

	# set the last field with the default values!
	l_v_field_default = l_l_v_field[-1]
	s_non_neg_v = {l_v_field_default[i] for i in range(0, amount_v, amount_nums+1)}
	cnf_obj.extend_cnfs(l_t_v=[(v, ) if v in s_non_neg_v else (-v, ) for v in l_v_field_default])

	# set the starting field on the first field
	l_v_field_last = l_l_v_field[0]
	cnf_obj.extend_cnfs(l_t_v=[
		(l_v_field_last[row*amount_v_row + col*amount_nums + (arr_field[row, col] - 1)], ) for row in range(0, rows) for col in range(0, cols)
	])

	# use only one direction of all the directions
	# cnfs_dir_1 = CNF.get_tseytin_only_one_true(l_v=l_v_dir_1)
	for l_v_dir in l_l_v_dir:
		cnfs_dir = CNF.get_tseytin_only_one_true(l_v=l_v_dir)
		cnf_obj.extend_cnfs(l_t_v=cnfs_dir)


	# all other moves (turns)
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

	models_amount = 1 # only one, and only one solution is sufficent engough
	with Glucose3(bootstrap_with=cnf_obj.cnfs) as m:
		is_solvable = m.solve()

		if not is_solvable:
			return None

		models = [(i, m) for m, i in zip(m.enum_models(), range(0, models_amount))]

	m = models[0]
	l = m[1]

	l_temp = l[amount_v*(amount_turns+1):]
	l_l_dir = [l_temp[amount_directions*i:amount_directions*(i+1)] for i in range(0, amount_turns)]

	l_move = np.array([np.where(np.array(l_dir) > 0)[0] for l_dir in l_l_dir]).flatten().tolist()
	return l_move


if __name__ == "__main__":
	rows = 3
	cols = 3

	amount_directions = 1+(rows+cols)*2

	amount_nums = rows * cols
	amount_v_row = amount_nums * cols
	amount_v = amount_nums**2

	arr_field_pos = np.empty((rows, cols, 2), dtype=np.int64)
	arr_field_pos[:, :, 0] = np.arange(0, rows).reshape((rows, 1))
	arr_field_pos[:, :, 1] = np.arange(0, cols).reshape((1, cols))

	arr_field_pos_left_to_right = np.roll(arr_field_pos, -1, 1)
	arr_field_pos_right_to_left = np.roll(arr_field_pos, 1, 1)

	arr_field_pos_t = arr_field_pos.transpose(1, 0, 2)
	arr_field_pos_up_to_down = np.roll(arr_field_pos_t, -1, 1)
	arr_field_pos_down_to_up = np.roll(arr_field_pos_t, 1, 1)

	l_arr_row_nr_to_row = (
		[(arr_field_pos, i, arr_field_pos_left_to_right[i]) for i in range(0, rows)]+
		[(arr_field_pos, i, arr_field_pos_right_to_left[i]) for i in range(0, rows)]+
		[(arr_field_pos_t, i, arr_field_pos_up_to_down[i]) for i in range(0, cols)]+
		[(arr_field_pos_t, i, arr_field_pos_down_to_up[i]) for i in range(0, cols)]
	)

	# min max neded turns to solve the puzzle
	
	# rows=1;cols=1;amount_turns=0
	# rows=1;cols=2;amount_turns=1
	# rows=1;cols=3;amount_turns=1
	# rows=1;cols=4;amount_turns=2
	# rows=1;cols=5;amount_turns=2
	# rows=1;cols=6;amount_turns=3
	
	# rows=2;cols=1;amount_turns=1
	# rows=2;cols=2;amount_turns=4
	# rows=2;cols=3;amount_turns=7
	# rows=2;cols=4;amount_turns=9
	# rows=2;cols=5;amount_turns=12
	# rows=2;cols=6;amount_turns=12

	# rows=3;cols=3;amount_turns=8

	arr_field_prev = np.arange(1, amount_nums+1).reshape((rows, cols))
	arr_field_next = arr_field_prev.copy()
	l_move = []
	amount_turns = 0
	amount_turns_prev = 0
	l_move_tpl = [(move_i, arr_field_pos_temp, i, arr_row_to) for move_i, (arr_field_pos_temp, i, arr_row_to) in enumerate(l_arr_row_nr_to_row, 1)]

	l_move_tpl_part = [l_move_tpl[:rows*2], l_move_tpl[rows*2:]]
	idx_part = 0

	max_round = 10
	round_i = 0
	# is_found_optimal_amount_turns = False
	while True:
		print()
		print(f"l_move: {l_move}")
		print(f"- amount_turns: {amount_turns}")

		# l_working_move = []
		for _ in range(0, 2): # do the 2 different parts
			l_move_tpl_part_used = l_move_tpl_part[idx_part]
			print(f"- idx_part: {idx_part}")
			
			for move_i, arr_field_pos_temp, i, arr_row_to in l_move_tpl_part_used:
				arr_field_next[:] = arr_field_prev # reset the values
				arr_row_from = arr_field_pos_temp[i] # get the arr_row, where the pos is from the values
				arr_field_next[tuple(arr_row_to.transpose(1, 0).tolist())] = arr_field_next[tuple(arr_row_from.transpose(1, 0).tolist())]

				print()
				print(f"-- move_i: {move_i}, arr_field_next:\n{arr_field_next}")

				l_move_solved = solve_a_mixed_moving_puzzle(amount_turns=amount_turns, arr_field=arr_field_next)
				if isinstance(l_move_solved, list):
					print(f"-- l_move_solved: {l_move_solved}")
					if 0 in l_move_solved:
						continue

					# l_working_move.append(move_i)
					continue
						
				amount_turns += 1
				l_move_solved = solve_a_mixed_moving_puzzle(amount_turns=amount_turns, arr_field=arr_field_next)
				print(f"-- amount_turns+=1, l_move_solved: {l_move_solved}")
				assert isinstance(l_move_solved, list)
				assert 0 not in l_move_solved

				arr_field_prev[:] = arr_field_next[:]
				l_move.append(move_i)
				break

			idx_part = (idx_part + 1) % 2

			if amount_turns_prev != amount_turns:
				break

		# if len(l_working_move)+1 == amount_directions:
		# 	break

		# if amount_turns_prev == amount_turns:
		# 	break

		amount_turns_prev = amount_turns

		round_i += 1
		if round_i >= max_round:
			print("-> Breaking, because of max_round!")
			break

	print(f"optimal max min amount_turns: {amount_turns}")
