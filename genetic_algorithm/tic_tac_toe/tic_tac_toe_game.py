#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.11 -i

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
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

from numpy.random import Generator, PCG64

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
load_module_dynamically(**dict(var_glob=var_glob, name='utils_all', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_all.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='SimpleNeuralNetwork', path="../SimpleNeuralNetwork.py"))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_current_datetime_str = utils_all.get_current_datetime_str

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

class TicTacToeBoard:
	def __init__(self, n_field, dim, n_player):
		self.n_field = n_field
		self.dim = dim
		self.n_player = n_player

		self.arr_field = np.zeros((self.n_field, )*self.dim, dtype=np.uint8)

		self.d_player_nr_to_next_player_nr = {i: i+1 for i in range(1, self.n_player)} | {self.n_player: 1}

		# define all the possible groups of clusters with the specific cell
		self.l_cluster_base = [[t_prefix+(i, ) for i in range(0, self.n_field)] for t_prefix in list(itertools.product(*[list(range(0, self.n_field))]*(self.dim-1)))]
		self.l_cell = sorted([t for l_t in self.l_cluster_base for t in l_t])
		self.arr_cluster_base = np.array(self.l_cluster_base, dtype=np.int64)

		self.arr_cluster_normal = np.concatenate([np.roll(self.arr_cluster_base, i, 2) for i in range(0, self.dim)])
		self.arr_idx = np.zeros((2, self.n_field), dtype=np.int64)
		self.arr_idx[0] = np.arange(0, self.n_field)
		self.arr_idx[1] = np.flip(np.arange(0, self.n_field))

		self.l_cluster_diag_idx = [(0, )+t_suffix for t_suffix in list(itertools.product(*[list(range(0, 2))]*(self.dim-1)))]
		self.arr_cluster_diag = self.arr_idx[np.array(self.l_cluster_diag_idx)].transpose(0, 2, 1)

		self.arr_cluster = np.concatenate([self.arr_cluster_normal, self.arr_cluster_diag])
		
		self.s_all_cell = set(itertools.product(*[[i for i in range(0, self.n_field)]]*self.dim))

		self.l_t_cluster_cell = [tuple(sorted([tuple(row.tolist()) for row in arr])) for arr in self.arr_cluster]
		self.d_cell_to_t_cluster = {cell: [] for cell in self.s_all_cell}

		for t_cluster_cell in self.l_t_cluster_cell:
			for cell in t_cluster_cell:
				self.d_cell_to_t_cluster[cell].append(t_cluster_cell)
		
		# define the attributes for the class
		self.s_empty_cell = set()
		self.s_used_cell = set()
		self.d_player_nr_to_l_used_cell = {i: [] for i in range(1, self.n_player+1)}

		self.reset_values()


	def reset_values(self):
		self.curr_player_nr = self.n_player

		self.arr_field[:] = 0
		
		self.s_empty_cell.clear()
		self.s_empty_cell.update(self.s_all_cell)
		
		self.s_used_cell.clear()
		
		for k in self.d_player_nr_to_l_used_cell:
			self.d_player_nr_to_l_used_cell[k] = []


	def get_next_player_nr(self):
		self.curr_player_nr = self.d_player_nr_to_next_player_nr[self.curr_player_nr]
		return self.curr_player_nr


	def __str__(self):
		return f"TicTacToeBoard(n_field={self.n_field}, dim={self.dim})\narr_field:\n{self.arr_field}"


if __name__ == '__main__':
	print("Hello World!")

	# take one random empty cell and place the players symbol in the cell

	n_field = 3
	dim = 2
	n_player = 2
	
	arr_player_won = np.zeros((n_player+1, ), dtype=np.int64)

	dt_str = "2022-11-27 06:00:45.797843"
	# dt_str = get_current_datetime_str()
	l_seed = list(dt_str.encode("utf-8"))
	rnd = Generator(bit_generator=PCG64(seed=l_seed))
	
	print(f"dt_str: {dt_str}")
	
	input_nodes = n_field**dim*n_player

	elite_games = 3
	take_best_player = 5
	amount_player = 20

	mix_rate = 0.60
	change_factor = 0.2725
	random_rate = 0.25

	def create_df_nn(amount_player):
		l_column = [
			"nr",
			"nn",
			"won", "loose",
			"won_against", "loose_against",
			"won_moves", "loose_moves",
			"won_pos_nr", "loose_pos_nr",
		]
		d_data = {column: [] for column in l_column}

		for nr in range(0, amount_player):
			d_data["nr"].append(nr)
			nn = SimpleNeuralNetwork.SimpleNeuralNetwork(
				input_nodes=input_nodes,
				hidden_nodes=[input_nodes*2, input_nodes*2],
				output_nodes=n_field**dim,
				rnd=rnd,
			)
			d_data["nn"].append(nn)
			d_data["won"].append(0)
			d_data["loose"].append(0)
			d_data["won_against"].append([])
			d_data["loose_against"].append([])
			d_data["won_moves"].append([])
			d_data["loose_moves"].append([])
			d_data["won_pos_nr"].append([])
			d_data["loose_pos_nr"].append([])
		
		df_nn = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

		return df_nn


	def reset_df_nn_stats(df_nn):
		df_nn["won"] = pd.Series(data=[0 for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["loose"] = pd.Series(data=[0 for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["won_against"] = pd.Series(data=[[] for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["loose_against"] = pd.Series(data=[[] for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["won_moves"] = pd.Series(data=[[] for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["loose_moves"] = pd.Series(data=[[] for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["won_pos_nr"] = pd.Series(data=[[] for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)
		df_nn["loose_pos_nr"] = pd.Series(data=[[] for _ in range(0, df_nn.shape[0])], dtype=object, index=df_nn.index)

	df_nn = create_df_nn(amount_player=amount_player)
	reset_df_nn_stats(df_nn=df_nn)

	tictactoe_board = TicTacToeBoard(n_field=n_field, dim=dim, n_player=n_player)
	
	s_empty_cell = tictactoe_board.s_empty_cell
	s_used_cell = tictactoe_board.s_used_cell
	d_player_nr_to_l_used_cell = tictactoe_board.d_player_nr_to_l_used_cell
	
	arr_field = tictactoe_board.arr_field

	l_player_player_won = []

	for round_i, t_idx in enumerate(itertools.permutations(range(0, amount_player), n_player), 1):
		assert len(t_idx) == n_player

		d_row_nn = {
			player_nr: df_nn.loc[idx] for player_nr, idx in enumerate(t_idx, 1)
		}

		if round_i % 100 == 0:
			print(f"round_i: {round_i}")

		tictactoe_board.reset_values()
		is_player_winning = False
		player_nr_won = 0

		move_nr = 0
		while tictactoe_board.s_empty_cell:
			move_nr += 1

			curr_player_nr = tictactoe_board.get_next_player_nr()

			X = np.zeros((1, input_nodes), dtype=np.float64) - 1.

			# TODO: make this more efficient
			arr_mult = n_field**np.flip(np.arange(0, dim))
			for cell in tictactoe_board.l_cell:
				# player_nr = arr_field[cell] # for <=python3.10
				player_nr = arr_field[*cell]

				if player_nr == 0:
					continue

				idx = np.sum(cell*arr_mult)*n_player

				X[0, idx+(player_nr-curr_player_nr)%n_player] = 1.

			nn_player = d_row_nn[curr_player_nr]['nn']

			Y = nn_player.calc_Y(X=X)

			arr_argsort = np.argsort(-Y[0])

			l_cell = tictactoe_board.l_cell
			for idx in arr_argsort:
				cell_coord = l_cell[idx]
				if cell_coord in s_empty_cell:
					break

			s_empty_cell.remove(cell_coord)
			s_used_cell.add(cell_coord)
			d_player_nr_to_l_used_cell[curr_player_nr].append(cell_coord)

			# arr_field[cell_coord] = curr_player_nr # for <=python3.10
			arr_field[*cell_coord] = curr_player_nr

			arr_pos = np.array(tictactoe_board.d_cell_to_t_cluster[cell_coord]).reshape((-1, dim)).transpose()

			# arr_clusters = arr_field[tuple(arr_pos.tolist())].reshape((-1, tictactoe_board.n_field)) # for <=python3.10
			arr_clusters = arr_field[*arr_pos].reshape((-1, tictactoe_board.n_field))

			if np.any(np.all(arr_clusters == curr_player_nr, axis=1)):
				is_player_winning = True
				player_nr_won = curr_player_nr
				break


		if player_nr_won != 0:
			nr_won = d_row_nn[player_nr_won]['nr']
			s_loose_player_nr = set([d_row_nn[player_nr]['nr'] for player_nr in range(1, n_player+1)]) - set([nr_won])
			for player_nr in range(1, n_player+1):
				row = d_row_nn[player_nr]
				if player_nr == player_nr_won:
					row['won'] += 1
					row['won_against'].append(s_loose_player_nr)
					row['won_moves'].append(move_nr)
					row['won_pos_nr'].append(player_nr)
				else:
					row['loose'] += 1
					row['loose_against'].append(nr_won)
					row['loose_moves'].append(move_nr)
					row['loose_pos_nr'].append(player_nr)

		arr_player_won[player_nr_won] += 1
		l_player_player_won.append((t_idx, player_nr_won))

	df_nn.sort_values(by=['won', 'loose'], ascending=[False, True], inplace=True)
	df_nn.reset_index(drop=True, inplace=True)

	print(f"df_nn[df_nn.columns[:4].values]:\n{df_nn[df_nn.columns[:4].values]}")
	# print(f"df_nn:\n{df_nn}")

	# pd.concat((
	# 	df_nn,
	# 	df_nn['won_moves'].apply(lambda x: np.mean(x)).rename('won_moves_mean'),
	# 	df_nn['loose_moves'].apply(lambda x: np.mean(x)).rename('loose_moves_mean'),
	# ), axis=1)

	func_vec_list = np.vectorize(lambda x: list(), otypes=[list])
	arr_win_table = func_vec_list(np.zeros((amount_player, amount_player)))
	arr_loose_table = func_vec_list(np.zeros((amount_player, amount_player)))
	for row_idx in range(0, amount_player):
		row = df_nn.loc[row_idx]
		nr = row['nr']

		l_s_won_against = row['won_against']
		l_won_pos_nr = row['won_pos_nr']
		l_won_moves = row['won_moves']
		for s_won_against, won_pos_nr, won_moves in zip(l_s_won_against, l_won_pos_nr, l_won_moves):
			for nr_loose in s_won_against:
				arr_win_table[nr, nr_loose].append((won_pos_nr, won_moves))

		l_loose_against = row['loose_against']
		l_loose_pos_nr = row['loose_pos_nr']
		l_loose_moves = row['loose_moves']
		for nr_won, loose_pos_nr, loose_moves in zip(l_loose_against, l_loose_pos_nr, l_loose_moves):
			arr_loose_table[nr, nr_won].append((loose_pos_nr, loose_moves))
