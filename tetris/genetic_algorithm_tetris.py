#! /usr/bin/python3.10

# pip installed libraries
import dill
import gzip
import keyboard
import os
import requests
import sh
import string
import subprocess
import sys
import time
import tty
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from copy import deepcopy
from io import StringIO
from memory_tempfile import MemoryTempfile
from multiprocessing import Pool
from PIL import Image
from typing import Dict

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

HOME_DIR = os.path.expanduser("~")
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_serialization', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_serialization.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

save_pkl_obj = utils_serialization.save_pkl_obj
load_pkl_obj = utils_serialization.load_pkl_obj

import utils_tetris

from utils_multiprocessing_parallel import MultiprocessingParallelManager

from tetris_game import TetrisGame

TETRIS_TEMP_DIR = os.path.join(TEMP_DIR, "tetris_temp")
mkdirs(TETRIS_TEMP_DIR)

def simple_tetris_genetic():
	class TetrisGameSimpleGenetic():
		def __init__(self, field_h, field_w, l_seed_piece, l_seed_weight):
			self.field_h = field_h
			self.field_w = field_w

			self.d_points_per_line_clear = {0: 0, 1: 1, 2: 3, 3: 6, 4: 10}
			self.d_points_per_line_clear = {k: v * field_w for k, v in d_points_per_line_clear.items()}

			self.l_seed_piece = l_seed_piece
			self.l_seed_weight = l_seed_weight
			self.is_filling_with_tetris_piece_nr = False

			self.rnd_piece = Generator(bit_generator=PCG64(seed=self.l_seed_piece))
			self.init_values()

			self.amount_weights = len(self.get_board_info())
			self.rnd_weight = Generator(bit_generator=PCG64(seed=self.l_seed_weight))
			self.init_weights()

		def init_values(self):
			# self.rnd_piece = Generator(bit_generator=PCG64(seed=self.l_seed_piece))

			self.arr_field = np.zeros((self.field_h, self.field_w), dtype=np.uint8)
			self.arr_field_prev = self.arr_field.copy()

			self.piece_nr_idx_next = self.rnd_piece.integers(0, 7, (1, ))[0]
			self.points = 0
			self.pieces_placed = 0
			self.lines_removed = 0
			self.lines_removed_prev = 0
			self.lines_type_removed = [0] * 4 # single, double, 3 or 4 (tetris) lines removed

			self.l_history_piece_nr = []
			self.l_history_orient_nr = []
			self.l_history_x = []

		def init_weights(self):
			# self.weights = (self.rnd_weight.random((self.amount_weights, )) - 0.5) * 2 # set the random values to -1...1
			# l_weight = [-0.6087311570649244, -0.7231046390395275, -0.6283360496747151, 0.6233584265588956, 0.07397920434668731, -0.06909243357294653, 0.32880264697813705, -0.7295563134368505, -0.4661053830242516]
			# l_weight = [-0.6087311570649244, -1.2703285585928936, -0.34030283287563623, 0.963738141508188, 1.047164817212276, -0.009315652625639936, 0.005625896168517497, -0.8924090882799527, -1.5288700830210513]
			# l_weight = [-0.6087311570649244, -1.688288365695647, -0.4941584429807942, 0.963738141508188, 1.047164817212276, -0.009315652625639936, 0.21486864077918172, -0.9373115111683088, -1.8260661257100996]
			# l_weight = [-0.6087311570649244, -1.688288365695647, -0.4941584429807942, 0.963738141508188, 1.047164817212276, -0.009315652625639936, 0.21486864077918172, -1.0794162964226148, -1.8260661257100996]
			l_weight = [-0.6087311570649244, -1.688288365695647, -0.4941584429807942, 0.963738141508188, 1.047164817212276, -0.009315652625639936, 0.21486864077918172, -1.0794162964226148, -1.431784774386101]
			self.weights = np.array(l_weight, dtype=np.float64)

		def set_prev_field(self):
			self.arr_field[:] = self.arr_field_prev

		def update_prev_field(self):
			self.arr_field_prev[:] = self.arr_field

		def define_next_piece(self):
			self.piece_nr_idx_next = self.rnd_piece.integers(0, 7, (1, ))[0]

		def update_stats(self, add_lines_removed):
			self.lines_removed_prev = self.lines_removed
			self.lines_removed += add_lines_removed
			if add_lines_removed > 0:
				self.lines_type_removed[add_lines_removed - 1] += 1
			self.points += self.d_points_per_line_clear[add_lines_removed]
			self.pieces_placed += 1

		def print_stats(self):
			print(f"self.lines_removed_prev: {self.lines_removed_prev}, self.lines_removed: {self.lines_removed}, self.lines_type_removed: {self.lines_type_removed}, self.points: {self.points}, self.pieces_placed: {self.pieces_placed}")

		def crossover_and_mutate(self, weights_1, weights_2, mix_rate, random_rate, change_factor):
			arr_is_used = (self.rnd_weight.random(self.weights.shape) <= mix_rate)
			if np.any(arr_is_used):
				t_idx_1 = np.where(arr_is_used)
				t_idx_2 = np.where(~arr_is_used)
				self.weights[t_idx_1] = weights_1[t_idx_1]
				self.weights[t_idx_2] = weights_2[t_idx_2]

			arr_is_used = (self.rnd_weight.random(self.weights.shape) <= random_rate)
			if np.any(arr_is_used):
				t_idx = np.where(arr_is_used)
				self.weights[t_idx] += (self.rnd_weight.random((np.sum(arr_is_used), ))-0.5)*2*change_factor

		def place_next_piece_at_random(self):
			piece_nr_idx = self.piece_nr_idx_next
			self.define_next_piece()

			orientation = self.rnd_piece.integers(0, 4, (1, ))[0]
			arr_yx = arr_tetris_pieces_rotate[piece_nr_idx][orientation]
			
			max_pos_x = self.field_w - np.max(arr_yx[1])
			x = self.rnd_piece.integers(0, max_pos_x, (1, ))[0]

			self.place_next_piece_with_arr_yx(arr_yx=arr_yx, x=x)

		def place_next_piece_with_arr_yx(self, arr_yx, x):
			assert np.max(arr_yx[1]) + x < self.field_w

			# piece_nr_idx = self.piece_nr_idx_next
			# self.piece_nr_idx_next = self.rnd_piece.integers(0, 7, (1, ))[0]

			# if not use_user_vals:
			# 	orientation = self.rnd_piece.integers(0, 4, (1, ))[0]

			# arr_x = np.zeros((self.nn.l_node_amount[0], ), dtype=np.float64)
			# arr_x[:] = -1
			# arr_x[piece_nr_idx] = 1
			# arr_x[7+self.piece_nr_idx_next] = 1

			# arr_field_flat = self.arr_field.reshape((-1, )).copy()
			# arr_idx_piece = (arr_field_flat > 0)
			# arr_field_flat[arr_idx_piece] = 1
			# arr_field_flat[~arr_idx_piece] = -1
			# arr_x[7*2:] = arr_field_flat
			# arr_y = self.nn.calc_feed_forward(X=arr_x.reshape((1, -1)))[0]

			# orientation = np.argmax(arr_y[self.field_w:])
			# arr_yx = arr_tetris_pieces_rotate[piece_nr_idx][orientation]

			# max_pos_x = self.field_w - np.max(arr_yx[1])
			max_y = np.max(arr_yx[0])

			# if not use_user_vals:
			# 	x = self.rnd_piece.integers(0, max_pos_x, (1, ))[0]
			# x = np.argmax(arr_y[:max_pos_x])
			y_prev = self.field_h - max_y - 1

			y = y_prev - 1

			is_first_piece_placeable = False
			while True:
				if np.any(self.arr_field[arr_yx[0]+y, arr_yx[1]+x] != 0):
					break

				y_prev = y
				y -= 1
				if y < 0:
					break

				is_first_piece_placeable = True

			self.arr_field_prev[:] = self.arr_field

			if self.is_filling_with_tetris_piece_nr:
				self.arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = piece_nr_idx + 1
			else:
				self.arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = 1

			add_lines_removed = 0
			# delete all rows, which are full!
			arr_idx_full = np.all(self.arr_field != 0, axis=1)
			if np.any(arr_idx_full):
				add_lines_removed = np.sum(arr_idx_full)
				
				# self.l_lines_cleared_l_arr_x[add_lines_removed - 1].append(arr_x)
				# self.l_lines_cleared_l_arr_y[add_lines_removed - 1].append(arr_y)
				self.arr_field[:self.field_h-add_lines_removed] = self.arr_field[~arr_idx_full]
				self.arr_field[self.field_h-add_lines_removed:] = 0

			return add_lines_removed, is_first_piece_placeable

		def play_the_game(self, max_pieces):
			for piece_nr in range(1, max_pieces+1):
				next_piece_nr = self.piece_nr_idx_next
				df_piece_positions_part = df_piece_positions.loc[df_piece_positions['piece_nr'].values == next_piece_nr]
				
				# print(f"piece_nr: {piece_nr}")
				l_fitness = []
				l_add_lines_removed = []
				l_is_first_piece_placeable = []
				l_idx = []
				for idx, row in df_piece_positions_part.iterrows():
					self.set_prev_field()
					add_lines_removed, is_first_piece_placeable = self.place_next_piece_with_arr_yx(arr_yx=row['arr_yx_base'], x=row['x'])
					tpl = self.get_board_info()
					fitness = np.sum(self.weights * tpl)
					l_fitness.append(fitness)
					l_add_lines_removed.append(add_lines_removed)
					l_is_first_piece_placeable.append(is_first_piece_placeable)
					l_idx.append(idx)

				# idx_best_fitness = self.rnd_piece.integers(0, len(df_piece_positions_part), (1, ))[0]
				# idx_best_fitness = np.argmin(l_fitness)
				idx_best_fitness = np.argmax(l_fitness)
				
				if not l_is_first_piece_placeable[idx_best_fitness]:
					break

				idx = l_idx[idx_best_fitness]
				row = df_piece_positions_part.loc[idx]
				self.set_prev_field()
				add_lines_removed, is_first_piece_placeable = self.place_next_piece_with_arr_yx(arr_yx=row['arr_yx_base'], x=row['x'])
				self.update_prev_field()
				assert add_lines_removed == l_add_lines_removed[idx_best_fitness]
				self.update_stats(add_lines_removed=add_lines_removed)

				self.l_history_piece_nr.append(row['piece_nr'])
				self.l_history_orient_nr.append(row['orient_nr'])
				self.l_history_x.append(row['x'])

				self.define_next_piece()

		def get_board_info(self):
			"""
			area: a numpy matrix representation of the board
			tetris: game wrapper
			s_lines: the starting number of cleared lines
			"""
			# Columns heights
			# area = self.arr_field
			area = np.flip(self.arr_field, axis=0)

			peaks = self.get_peaks(area)
			highest_peak = np.max(peaks)

			# Aggregated height
			agg_height = np.sum(peaks)

			holes = self.get_holes(peaks, area)
			n_holes = np.sum(holes)
			n_cols_with_holes = np.count_nonzero(np.array(holes) > 0)

			row_transitions = self.get_row_transition(area, highest_peak)
			col_transitions = self.get_col_transition(area, peaks)
			bumpiness = self.get_bumpiness(peaks)
			num_pits = np.count_nonzero(np.count_nonzero(area, axis=0) == 0)

			wells = self.get_wells(peaks)
			max_wells = np.max(wells)

			cleared = (self.lines_removed - self.lines_removed_prev) * 8

			return agg_height, n_holes, bumpiness, cleared, num_pits, max_wells, \
				n_cols_with_holes, row_transitions, col_transitions
		  
		 
		def get_peaks(self, area):
			peaks = np.array([], dtype=np.int16)
			for col in range(area.shape[1]):
				if 1 in area[:, col]:
					p = area.shape[0] - np.argmax(area[:, col], axis=0)
					peaks = np.append(peaks, p)
				else:
					peaks = np.append(peaks, 0)
			return peaks
		  
		 
		def get_row_transition(self, area, highest_peak):
			sum = 0
			# From highest peak to bottom
			for row in range(int(area.shape[0] - highest_peak), area.shape[0]):
				for col in range(1, area.shape[1]):
					if area[row, col] != area[row, col - 1]:
						sum += 1
			return sum


		def get_col_transition(self, area, peaks):
			sum = 0
			for col in range(area.shape[1]):
				if peaks[col] <= 1:
					continue
				for row in range(int(area.shape[0] - peaks[col]), area.shape[0] - 1):
					if area[row, col] != area[row + 1, col]:
						sum += 1
			return sum


		def get_bumpiness(self, peaks):
			s = 0
			for i in range(self.field_w-1):
				s += np.abs(peaks[i] - peaks[i + 1])
			return s


		def get_holes(self, peaks, area):
			# Count from peaks to bottom
			holes = []
			for col in range(area.shape[1]):
				start = -peaks[col]
				# If there's no holes i.e. no blocks on that column
				if start == 0:
					holes.append(0)
				else:
					holes.append(np.count_nonzero(area[int(start):, col] == 0))
			return holes


		def get_wells(self, peaks):
			wells = []
			for i in range(len(peaks)):
				if i == 0:
					w = peaks[1] - peaks[0]
					w = w if w > 0 else 0
					wells.append(w)
				elif i == len(peaks) - 1:
					w = peaks[-2] - peaks[-1]
					w = w if w > 0 else 0
					wells.append(w)
				else:
					w1 = peaks[i - 1] - peaks[i]
					w2 = peaks[i + 1] - peaks[i]
					w1 = w1 if w1 > 0 else 0
					w2 = w2 if w2 > 0 else 0
					w = w1 if w1 >= w2 else w2
					wells.append(w)
			return wells

	# arr_field = np.zeros((field_h, field_w), dtype=np.uint8)
	# arr_field_prev = arr_field.copy()

	# arr_field[0] = np.random.randint(0, 2, (arr_field.shape[1], ))
	# arr_field_prev[:] = arr_field

	# l_arr_x = np.zeros((n_pieces, n_pieces*2+field_h*field_w), dtype=np.float64)
	# l_arr_y = np.zeros((n_pieces, field_w+4), dtype=np.float64)

	# l_best_piece_placement = []
	# for piece_name in l_piece_name:
	# 	df_piece_positions_part = df_piece_positions.loc[df_piece_positions['piece_name'].values == piece_name]
	# 	l_fitness = []

	# 	break

	# tetris_game = TetrisGameSimpleGenetic(field_h=field_h, field_w=field_w, l_seed_piece=[0, 0, 1], l_seed_weight=[0, 0, 1])

	# next_piece_nr = tetris_game.piece_nr_idx_next
	# df_piece_positions_part = df_piece_positions.loc[df_piece_positions['piece_nr'].values == next_piece_nr]

	# tetris_game.play_the_game(max_pieces=max_pieces)

	take_best_games = 5
	max_game_nr = 50

	# take_best_games = 2
	# max_game_nr = 10

	l_column = ['tetris_game', 'game_nr', 'points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed']
	df_games = pd.DataFrame(data=np.empty((max_game_nr, len(l_column))).tolist(), columns=l_column, dtype=object)

	# arr_tetris_game = np.array([None]*max_game_nr, dtype=object)
	max_pieces = 1000
	for game_nr in range(0, max_game_nr):
		row = df_games.loc[game_nr]

		l_seed_prefix = np.frombuffer(time.time_ns().to_bytes(8, 'little'), dtype=np.uint32).tolist()

		l_seed_piece = l_seed_prefix + [0, 1, game_nr]
		l_seed_weight = l_seed_prefix + [0, 1, game_nr]
		tetris_game = TetrisGameSimpleGenetic(field_h=field_h, field_w=field_w, l_seed_piece=l_seed_piece, l_seed_weight=l_seed_weight)
		row['tetris_game'] = tetris_game

		# continue

	# pool = Pool(processes=10)

	def do_tetris_game(tetris_game):
		tetris_game.play_the_game(max_pieces=max_pieces)

	# def do_tetris_game(game_nr):
	# 	row = df_games.loc[game_nr]
	# 	tetris_game = row['tetris_game']
	# 	tetris_game.play_the_game(max_pieces=max_pieces)

	# for game_nr in range(0, max_game_nr):
	# 	pool.apply_async(do_tetris_game, (df_games.loc[game_nr]['tetris_game'], ))

	# # pool.starmap(do_tetris_game, [df_games.loc[game_nr]['tetris_game'] for game_nr in range(0, max_game_nr)])
	# # pool.starmap(do_tetris_game, [game_nr for game_nr in range(0, max_game_nr)])

	# for game_nr in range(0, max_game_nr):
	# 	row = df_games.loc[game_nr]

	# 	print(f"game_nr: {game_nr:4}, points: {tetris_game.points}, lines_type_removed_reverse: {tuple(tetris_game.lines_type_removed[::-1])}, pieces_placed: {tetris_game.pieces_placed}, lines_removed: {tetris_game.lines_removed}")

	# 	row['game_nr'] = game_nr
	# 	row['points'] = tetris_game.points
	# 	row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
	# 	row['lines_removed'] = tetris_game.lines_removed
	# 	row['pieces_placed'] = tetris_game.pieces_placed
	# 	# row = df_games.loc[game_nr]
	
	for game_nr in range(0, max_game_nr):
		row = df_games.loc[game_nr]
		tetris_game = row['tetris_game']

		tetris_game.play_the_game(max_pieces=max_pieces)

		print(f"game_nr: {game_nr:4}, points: {tetris_game.points}, lines_type_removed_reverse: {tuple(tetris_game.lines_type_removed[::-1])}, pieces_placed: {tetris_game.pieces_placed}, lines_removed: {tetris_game.lines_removed}")

		row['game_nr'] = game_nr
		row['points'] = tetris_game.points
		row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
		row['lines_removed'] = tetris_game.lines_removed
		row['pieces_placed'] = tetris_game.pieces_placed

	df_games.sort_values(by=['points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed'], inplace=True, ascending=False)
	df_games.reset_index(drop=True, inplace=True)
	print(f"df_games init:\n{df_games.loc[:take_best_games*2-1]}")

	# tetris_game = df_games.loc[0]['tetris_game']

	# tetris_game.init_values()

	# add_lines_removed, is_first_piece_placeable = tetris_game.place_next_piece_with_arr_yx(arr_yx=arr_tetris_pieces_rotate[0][0], x=0)
	# tetris_game.update_prev_field()
	# tetris_game.update_stats(add_lines_removed=add_lines_removed)
	# print(f"add_lines_removed: {add_lines_removed}")
	# tetris_game.print_stats()

	# add_lines_removed, is_first_piece_placeable = tetris_game.place_next_piece_with_arr_yx(arr_yx=arr_tetris_pieces_rotate[0][0], x=2)
	# tetris_game.update_prev_field()
	# tetris_game.update_stats(add_lines_removed=add_lines_removed)
	# print(f"add_lines_removed: {add_lines_removed}")
	# tetris_game.print_stats()

	# add_lines_removed, is_first_piece_placeable = tetris_game.place_next_piece_with_arr_yx(arr_yx=arr_tetris_pieces_rotate[0][0], x=4)
	# tetris_game.update_prev_field()
	# tetris_game.update_stats(add_lines_removed=add_lines_removed)
	# print(f"add_lines_removed: {add_lines_removed}")
	# tetris_game.print_stats()

	# print(f"arr_field:\n{np.flip(tetris_game.arr_field, axis=0)}")

	# sys.exit()

	mix_rate = 0.50
	change_factor = 0.45
	random_rate = 0.01

	def do_tetris_game_with_init(tetris_game):
		tetris_game.init_values()
		tetris_game.play_the_game(max_pieces=max_pieces)

	for i_round in range(1, 30+1):
		for game_nr_current in range(take_best_games, max_game_nr):
			arr_idx_game = np.random.permutation(np.arange(0, take_best_games))[:2]
			tetris_game_1 = df_games.loc[arr_idx_game[0]]['tetris_game']
			tetris_game_2 = df_games.loc[arr_idx_game[1]]['tetris_game']

			row = df_games.loc[game_nr_current]
			tetris_game = row['tetris_game']

			tetris_game.crossover_and_mutate(weights_1=tetris_game_1.weights, weights_2=tetris_game_2.weights, mix_rate=mix_rate, random_rate=random_rate, change_factor=change_factor)

			# play again
			tetris_game.init_values()
			tetris_game.play_the_game(max_pieces=max_pieces)


			row['points'] = tetris_game.points
			# row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed)
			row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
			row['lines_removed'] = tetris_game.lines_removed
			row['pieces_placed'] = tetris_game.pieces_placed

		# for game_nr in range(take_best_games, max_game_nr):
		# 	pool.apply_async(do_tetris_game_with_init, (df_games.loc[game_nr]['tetris_game'], ))

		# for game_nr in range(take_best_games, max_game_nr):
			# row = df_games.loc[game_nr]
			# row['points'] = tetris_game.points
			# row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
			# row['lines_removed'] = tetris_game.lines_removed
			# row['pieces_placed'] = tetris_game.pieces_placed

		df_games.sort_values(by=['points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed'], inplace=True, ascending=False)
		df_games.reset_index(drop=True, inplace=True)
		print(f"df_games round {i_round:5}:\n{df_games.loc[:take_best_games*2-1]}")

	# for piece_nr in range(1, max_pieces+1):
	# 	print(f"piece_nr: {piece_nr}")
	# 	l_fitness = []
	# 	l_add_lines_removed = []
	# 	l_is_first_piece_placeable = []
	# 	for i, row in df_piece_positions_part.iterrows():
	# 		tetris_game.set_prev_field()
	# 		add_lines_removed, is_first_piece_placeable = tetris_game.place_next_piece_with_arr_yx(arr_yx=row['arr_yx_base'], x=row['x'])
	# 		tpl = tetris_game.get_board_info()
	# 		fitness = np.sum(tetris_game.weights * tpl)
	# 		l_fitness.append(fitness)
	# 		l_add_lines_removed.append(add_lines_removed)
	# 		l_is_first_piece_placeable.append(is_first_piece_placeable)

	# 	idx_best_fitness = np.argmax(l_fitness)
		
	# 	if not l_is_first_piece_placeable[idx_best_fitness]:
	# 		break

	# 	row = df_piece_positions_part.iloc[idx_best_fitness]
	# 	tetris_game.set_prev_field()
	# 	add_lines_removed, is_first_piece_placeable = tetris_game.place_next_piece_with_arr_yx(arr_yx=row['arr_yx_base'], x=row['x'])
	# 	tetris_game.update_prev_field()
	# 	assert add_lines_removed == l_add_lines_removed[idx_best_fitness]
	# 	tetris_game.update_stats(add_lines_removed=add_lines_removed)

	# 	tetris_game.define_next_piece()

	# tetris_game.piece_nr_idx_next = 4
	# tetris_game.place_next_piece(orientation=0, x=0, use_user_vals=True)
	# tetris_game.piece_nr_idx_next = 4
	# tetris_game.place_next_piece(orientation=0, x=2, use_user_vals=True)
	# tetris_game.set_prev_field()
	# # tetris_game.piece_nr_idx_next = 4
	# # tetris_game.place_next_piece(orientation=0, x=2, use_user_vals=True)
	# # tetris_game.piece_nr_idx_next = 4
	# # tetris_game.place_next_piece(orientation=1, x=0, use_user_vals=True)
	# tpl = tetris_game.get_board_info()

	# print(f"np.flip(tetris_game.arr_field, axis=0): {np.flip(tetris_game.arr_field, axis=0)}")
	# print(f"tpl: {tpl}")


class ManyTetrisGame():

	def __init__(self, dir_path, many_game_nr, h, w, l_hidden_neurons=[], l_seed_main=[0, 0], l_seed_prefix=[0, 0], d_params={}):
		self.dir_path = dir_path

		self.tetris_gamefield_images_dir = os.path.join(self.dir_path, "tetris_gamefield_images")
		mkdirs(self.tetris_gamefield_images_dir)
		
		self.many_game_nr = many_game_nr

		self.field_h = h
		self.field_w = w

		self.d_params = deepcopy(d_params)

		self.l_hidden_neurons = deepcopy(l_hidden_neurons)
		self.l_hidden_neurons_str = '_'.join(map(str, self.l_hidden_neurons))

		self.l_seed_main = deepcopy(l_seed_main)
		self.l_seed_prefix = deepcopy(l_seed_prefix)

		self.seed_orig = np.array(self.l_seed_main, dtype=np.uint32)
		self.seed = self.seed_orig.copy()
		self.rnd = Generator(bit_generator=PCG64(seed=self.seed))

		self.l_l_column_sort = [
			# ['lines_type_removed_reverse', 'arr_removing_lines_points', 'lines_removed', 'lines_removed_points', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['max_line_at_once_remove', 'max_line_at_once_remove_times', 'lines_type_removed_reverse', 'piece_cell_rest_points', 'lines_removed_points', 'arr_removing_lines_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			['lines_removed_points', 'lines_type_removed_reverse', 'pieces_placed', 'piece_cell_rest_points', 'lines_removed', 'points', 'arr_removing_lines', 'arr_removing_lines_points'],
			# ['max_line_at_once_remove', 'max_line_at_once_remove_times', 'arr_removing_lines_points', 'lines_type_removed_reverse', 'piece_cell_rest_points', 'lines_removed_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['lines_type_removed_reverse', 'piece_cell_rest_points', 'lines_removed_points', 'arr_removing_lines_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['lines_removed_points', 'lines_type_removed_reverse', 'arr_removing_lines_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['arr_removing_lines_points', 'lines_type_removed_reverse', 'lines_removed', 'lines_removed_points', 'pieces_placed', 'points', 'arr_removing_lines'],
			
			# ['lines_type_removed_reverse', 'arr_removing_lines_points', 'lines_removed_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['lines_type_removed_reverse', 'arr_removing_lines_points', 'lines_removed_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['lines_removed_points', 'arr_removing_lines_points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['lines_type_removed_reverse', 'arr_removing_lines_points', 'lines_removed', 'lines_removed_points', 'pieces_placed', 'points', 'arr_removing_lines'],
			# ['lines_removed', 'arr_removing_lines_points', 'lines_type_removed_reverse', 'lines_removed_points', 'pieces_placed', 'points', 'arr_removing_lines'],
			
			# ['arr_removing_lines', 'lines_removed', 'lines_type_removed_reverse', 'lines_removed_points', 'pieces_placed', 'points'],
			# ['points', 'lines_type_removed_reverse', 'lines_removed_points', 'lines_removed', 'pieces_placed', 'arr_removing_lines'],
			# ['pieces_placed', 'lines_type_removed_reverse', 'lines_removed_points', 'lines_removed', 'points', 'arr_removing_lines'],
		]
		self.l_column = ['tetris_game', 'game_nr'] + self.l_l_column_sort[0]
		self.df_games = pd.DataFrame(data=np.empty((self.d_params['max_game_nr'], len(self.l_column))).tolist(), columns=self.l_column, dtype=object)

		self.suffix_file_path = f'many_game_nr_{self.many_game_nr:02}_field_w_{self.field_w}_field_h_{self.field_h}_hidden_neurons_{self.l_hidden_neurons_str}'
		self.file_path_arr_bw_top = os.path.join(self.dir_path, f'arr_bw_top_{self.suffix_file_path}.pkl')
	
	@staticmethod
	def update_info_in_row(row, tetris_game):
		row['points'] = tetris_game.points
		row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
		row['lines_removed'] = tetris_game.lines_removed
		row['lines_removed_points'] = tetris_game.lines_removed_points
		row['pieces_placed'] = tetris_game.pieces_placed
		row['arr_removing_lines'] = tuple(tetris_game.arr_removing_lines.tolist())
		row['arr_removing_lines_points'] = np.sum(tetris_game.arr_removing_lines * np.cumsum(np.arange(1, tetris_game.arr_removing_lines.shape[0]+1))[::-1])
		row['piece_cell_rest_points'] = tetris_game.piece_cell_rest_points
		i_argmax = np.argmax(tetris_game.lines_type_removed)
		row['max_line_at_once_remove'] = i_argmax + 1
		row['max_line_at_once_remove_times'] = tetris_game.lines_type_removed[i_argmax]
		# row['arr_removing_lines_points'] = np.sum(tetris_game.arr_removing_lines * np.arange(1, tetris_game.arr_removing_lines.shape[0]+1)[::-1])

	def init_tetris_game_in_df_games(self):
		if os.path.exists(self.file_path_arr_bw_top):
			arr_bw_top = load_pkl_obj(self.file_path_arr_bw_top)
		else:
			arr_bw_top = None

		max_game_nr = self.d_params['max_game_nr']

		for game_nr in range(0, max_game_nr):
			row = self.df_games.loc[game_nr]

			tetris_game = TetrisGame(
				game_nr=game_nr,
				h=self.field_h,
				w=self.field_w,
				n_pieces=n_pieces,
				arr_tetris_pieces_rotate=arr_tetris_pieces_rotate,
				dir_path_picture=os.path.join(self.tetris_gamefield_images_dir, f"game_{game_nr:03}"),
				l_hidden_neurons=self.l_hidden_neurons,
				# l_seed_main=self.seed,
				l_seed_main=self.l_seed_main + [0, game_nr],
				# l_seed=self.l_seed_prefix,
				l_seed=self.l_seed_prefix + [0, game_nr],
			)
			row['tetris_game'] = tetris_game
			row['game_nr'] = tetris_game.game_nr

			if arr_bw_top is not None:
				if game_nr < arr_bw_top.shape[0]:
					tetris_game.nn.arr_bw = arr_bw_top[game_nr]

		self.i_round_total = 0

		self.l_round = []

		self.l_points_best = []
		self.l_points_mean_top = []
		self.l_points_mean_all = []

		self.l_lines_removed_best = []
		self.l_lines_removed_mean_top = []
		self.l_lines_removed_mean_all = []

	def play_many_tetris_game(self):
		max_game_nr = self.d_params['max_game_nr']
		elite_games = self.d_params['elite_games']
		take_best_games = self.d_params['take_best_games']
		mix_rate = self.d_params['mix_rate']
		change_factor = self.d_params['change_factor']
		random_rate = self.d_params['random_rate']
		loop_kth_same_seed = self.d_params['loop_kth_same_seed']
		max_round = self.d_params['max_round']
		max_pieces = self.d_params['max_pieces']

		# self.seed[0] += 1
		i_round = 0
		for game_nr in range(0, max_game_nr):
			row = self.df_games.loc[game_nr]
			tetris_game = row['tetris_game']

			# tetris_game.seed[:] = self.seed
			tetris_game.increment_seed()
			tetris_game.play_the_game(max_pieces=max_pieces)

			# print(f"game_nr: {game_nr:4}, tetris_game.pieces_placed: {tetris_game.pieces_placed}, tetris_game.lines_removed: {tetris_game.lines_removed}")

			ManyTetrisGame.update_info_in_row(row=row, tetris_game=tetris_game)

		l_column_sort = self.l_l_column_sort[0]
		self.df_games.sort_values(by=l_column_sort, inplace=True, ascending=False)
		self.df_games.reset_index(drop=True, inplace=True)
		# print(f"self.df_games init:\n{self.df_games}")

		self.l_round.append(self.i_round_total)
		self.i_round_total += 1

		self.l_points_best.append(self.df_games['points'].values[0])
		self.l_points_mean_top.append(np.mean(self.df_games['points'].values[:take_best_games]))
		self.l_points_mean_all.append(np.mean(self.df_games['points'].values))

		self.l_lines_removed_best.append(self.df_games['lines_removed'].values[0])
		self.l_lines_removed_mean_top.append(np.mean(self.df_games['lines_removed'].values[:take_best_games]))
		self.l_lines_removed_mean_all.append(np.mean(self.df_games['lines_removed'].values))

		for i_round in range(1, max_round+1):
			# self.seed[0] += 1
			random_rate_next = random_rate
			# random_rate_next = random_rate * np.exp(-i_round * stretch_factor) * (1 - random_rate_max) + random_rate_max
			# print()
			# print(f"random_rate_next: {random_rate_next}")
			# print(f"l_hidden_neurons: {l_hidden_neurons}")
			# print(f"self.seed: {self.seed}")
			
			if i_round % loop_kth_same_seed == 0: # let the elits play every k-th game too!
				for game_nr_current in range(elite_games, take_best_games):
					row = self.df_games.loc[game_nr_current]
					tetris_game = row['tetris_game']

					# play again
					# tetris_game.seed[:] = self.seed
					tetris_game.increment_seed()
					tetris_game.play_the_game(max_pieces=max_pieces)

					ManyTetrisGame.update_info_in_row(row=row, tetris_game=tetris_game)

			for game_nr_current in range(take_best_games, max_game_nr):
				row = self.df_games.loc[game_nr_current]
				tetris_game = row['tetris_game']

				arr_idx_game = self.rnd.permutation(np.arange(0, take_best_games))[:2]
				tetris_game_1 = self.df_games.loc[arr_idx_game[0]]['tetris_game']
				tetris_game_2 = self.df_games.loc[arr_idx_game[1]]['tetris_game']
				
				tetris_game.nn.crossover_and_mutate(arr_bw_1=tetris_game_1.nn.arr_bw, arr_bw_2=tetris_game_2.nn.arr_bw, mix_rate=mix_rate, random_rate=random_rate_next, change_factor=change_factor)

				# play again
				# tetris_game.seed[:] = self.seed
				tetris_game.increment_seed()
				tetris_game.play_the_game(max_pieces=max_pieces)

				ManyTetrisGame.update_info_in_row(row=row, tetris_game=tetris_game)

			l_column_sort = self.l_l_column_sort[i_round % len(self.l_l_column_sort)]
			# print(f"l_column_sort: {l_column_sort}")
			self.df_games.sort_values(by=l_column_sort, inplace=True, ascending=False)
			self.df_games.reset_index(drop=True, inplace=True)
			# print(f"self.df_games round {i_round:5}:\n{self.df_games.loc[:take_best_games*4-1]}")

			self.l_round.append(self.i_round_total)
			self.i_round_total += 1

			self.l_points_best.append(self.df_games['points'].values[0])
			self.l_points_mean_top.append(np.mean(self.df_games['points'].values[:take_best_games]))
			self.l_points_mean_all.append(np.mean(self.df_games['points'].values))

			self.l_lines_removed_best.append(self.df_games['lines_removed'].values[0])
			self.l_lines_removed_mean_top.append(np.mean(self.df_games['lines_removed'].values[:take_best_games]))
			self.l_lines_removed_mean_all.append(np.mean(self.df_games['lines_removed'].values))

		# plt.figure()

		# p1, = plt.plot(self.l_round, self.l_points_best, marker='', linestyle='-', linewidth=0.5, label='best')
		# p2, = plt.plot(self.l_round, self.l_points_mean_top, marker='', linestyle='-', linewidth=0.5, label='mean_top')
		# p3, = plt.plot(self.l_round, self.l_points_mean_all, marker='', linestyle='-', linewidth=0.5, label='mean_all')

		# plt.legend(handles=[p1, p2, p3])

		# plt.title("Points of each generation")

		# plt.xlabel("round/generation")
		# plt.ylabel("points")

		# # plt.show()
		# plt.savefig(os.path.join(self.dir_path, f'plot_points_{self.suffix_file_path}.png'), dpi=300)

		# arr_tetris_game = self.df_games['tetris_game'].values
		# arr_bw_top = np.array([tg.nn.arr_bw for tg in arr_tetris_game[:take_best_games*2]], dtype=object)
		# # arr_bw_top = np.array([tg.nn.arr_bw for tg in arr_tetris_game[:max_game_nr]], dtype=object)

		# save_pkl_obj(arr_bw_top, self.file_path_arr_bw_top)

	def init(self):
		self.init_tetris_game_in_df_games()

	def next(self, max_round):
		self.d_params['max_round'] = max_round
		self.play_many_tetris_game()

		df_part = self.df_games.iloc[:self.d_params['take_best_games']]
		# l_column = df_part.columns.tolist()
		# l_column.pop(l_column.index('tetris_game'))
		# df_part = df[l_column]

		return (df_part.copy(), )

	def update(self, l_arr_bw):
		max_num = max(len(l_arr_bw), self.d_params['take_best_games'])
		for _, tetris_game, arr_bw in zip(range(0, max_num), self.df_games['tetris_game'].values, l_arr_bw):
			for bw, bw_new in zip(tetris_game.nn.arr_bw, arr_bw):
				bw[:] = bw_new

	def save(self):
		plt.figure()

		p1, = plt.plot(self.l_round, self.l_points_best, marker='', linestyle='-', linewidth=0.5, label='best')
		p2, = plt.plot(self.l_round, self.l_points_mean_top, marker='', linestyle='-', linewidth=0.5, label='mean_top')
		p3, = plt.plot(self.l_round, self.l_points_mean_all, marker='', linestyle='-', linewidth=0.5, label='mean_all')

		plt.legend(handles=[p1, p2, p3])

		plt.title("Points of each generation")

		plt.xlabel("round/generation")
		plt.ylabel("points")

		# plt.show()
		plt.savefig(os.path.join(self.dir_path, f'plot_points_{self.suffix_file_path}.png'), dpi=300)


		plt.figure()

		p1, = plt.plot(self.l_round, self.l_lines_removed_best, marker='', linestyle='-', linewidth=0.5, label='best')
		p2, = plt.plot(self.l_round, self.l_lines_removed_mean_top, marker='', linestyle='-', linewidth=0.5, label='mean_top')
		p3, = plt.plot(self.l_round, self.l_lines_removed_mean_all, marker='', linestyle='-', linewidth=0.5, label='mean_all')

		plt.legend(handles=[p1, p2, p3])

		plt.title("Lines removed of each generation")

		plt.xlabel("round/generation")
		plt.ylabel("lines_removed")

		# plt.show()
		plt.savefig(os.path.join(self.dir_path, f'plot_lines_removed_{self.suffix_file_path}.png'), dpi=300)


		arr_tetris_game = self.df_games['tetris_game'].values
		arr_bw_top = np.array([tg.nn.arr_bw for tg in arr_tetris_game[:take_best_games*2]], dtype=object)
		# arr_bw_top = np.array([tg.nn.arr_bw for tg in arr_tetris_game[:max_game_nr]], dtype=object)

		save_pkl_obj(arr_bw_top, self.file_path_arr_bw_top)


class AccumulateManyTetrisGame():
	
	def __init__(self, field_h, field_w, l_hidden_neurons_str, min_amount, max_amount, l_seed, amount_best_arr_bw):
		self.field_h = field_h
		self.field_w = field_w
		self.l_hidden_neurons_str = l_hidden_neurons_str

		assert(min_amount >= 1)
		assert(min_amount <= max_amount)

		self.min_amount = min_amount
		self.max_amount = max_amount
		
		self.l_seed = deepcopy(l_seed)
		self.seed = np.array(l_seed, dtype=np.uint32)
		self.rnd = Generator(bit_generator=PCG64(seed=self.seed))

		self.amount_best_arr_bw = amount_best_arr_bw

		self.print_prefix = f"field_h: {self.field_h:2}, field_w: {self.field_w:2}, l_hidden_neurons_str: {self.l_hidden_neurons_str}"


	def get_next_amount(self):
		return self.rnd.integers(self.min_amount, self.max_amount + 1)


	def init(self):
		self.l_df_game = []
		self.l_best_arr_bw = []
		self.df_game = None


	def accumulate(self, df_game):
		self.l_df_game.append(df_game)


	def update(self):
		self.df_game = pd.concat(self.l_df_game)
		# l_column_sort = ['arr_removing_lines', 'lines_removed', 'lines_type_removed_reverse', 'lines_removed_points', 'pieces_placed', 'points']
		# l_column_sort = ['lines_removed', 'arr_removing_lines', 'lines_type_removed_reverse', 'lines_removed_points', 'pieces_placed', 'points']
		l_column_sort = ['lines_removed_points', 'lines_removed', 'piece_cell_rest_points', 'pieces_placed', 'points', 'arr_removing_lines', 'arr_removing_lines_points']
		# l_column_sort = ['arr_removing_lines_points', 'piece_cell_rest_points', 'lines_removed_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines']
		# l_column_sort = ['max_line_at_once_remove', 'max_line_at_once_remove_times', 'arr_removing_lines_points', 'piece_cell_rest_points', 'lines_removed_points', 'lines_removed', 'pieces_placed', 'points', 'arr_removing_lines']
		self.df_game.sort_values(by=l_column_sort, inplace=True, ascending=False)
		self.df_game.reset_index(drop=True, inplace=True)

		self.l_best_arr_bw = [tetris_game.nn.arr_bw for tetris_game in self.df_game['tetris_game'].values[:self.amount_best_arr_bw]]


	def next(self):
		return self.l_best_arr_bw


	def current(self):
		return self.df_game.iloc[:self.amount_best_arr_bw]


if __name__ == '__main__':
	print("Hello World!")

	# acc_many_tetris_game = AccumulateManyTetrisGame(**{'min_amount': 2, 'max_amount': 4, 'l_seed': [0, 1], 'amount_best_arr_bw': 7})

	# sys.exit()

	# TODO: create a simple network for differnet tetris pieces

	# TODO: create a function for creating different examples for the training set for the neuronal network

	# mean_from_n_games = 7 # TODO: maybe not needed?!?!

	field_w = 6
	field_h = 10

	d_pieces = utils_tetris.prepare_pieces(field_w=field_w)

	arr_tetris_pieces_rotate = d_pieces['arr_tetris_pieces_rotate']
	df_piece_positions = d_pieces['df_piece_positions']
	arr_tetris_pieces_base = d_pieces['arr_tetris_pieces_base']

	n_pieces = arr_tetris_pieces_base.shape[0]

	# simple_tetris_genetic()

	# arr_max_x = np.max(arr_tetris_pieces_rotate[:, :, 1], axis=2)

	# sys.exit()

	cpu_amount = int(sys.argv[1])

	elite_games = 3
	take_best_games = 5
	max_game_nr = 60

	mix_rate = 0.60
	change_factor = 0.2725
	random_rate = 0.25

	loop_kth_same_seed = 1
	max_round = int(sys.argv[2])
	max_pieces = 35

	d_params = dict(
		elite_games=elite_games,
		take_best_games=take_best_games,
		max_game_nr=max_game_nr,
		mix_rate=mix_rate,
		change_factor=change_factor,
		random_rate=random_rate,
		loop_kth_same_seed=loop_kth_same_seed,
		max_round=max_round,
		max_pieces=max_pieces,
	)

	# stretch_factor = 0.002
	# random_rate_max = 0.01

	l_hidden_neurons = list(map(int, sys.argv[3].split(',')))
	l_hidden_neurons_str = '_'.join(map(str, l_hidden_neurons))
	# l_seed_main = [0, 0, 1]

	max_many_tetris_games_iter = int(sys.argv[4])

	# l_seed_main = np.frombuffer(time.time_ns().to_bytes(8, 'little'), dtype=np.uint32).tolist()
	# l_seed_prefix = np.frombuffer(time.time_ns().to_bytes(8, 'little'), dtype=np.uint32).tolist()

	# many_tetris_game_0 = ManyTetrisGame(
	# 	many_game_nr=0,
	# 	h=field_h,
	# 	w=field_w,
	# 	l_hidden_neurons=l_hidden_neurons,
	# 	l_seed_main=l_seed_main,
	# 	l_seed_prefix=l_seed_prefix,
	# 	d_params=d_params,
	# )
	# # many_tetris_game_0.init_tetris_game_in_df_games()
	# # many_tetris_game_0.play_many_tetris_game()
	# many_tetris_game_0.init()
	# many_tetris_game_0.next(10)
	# many_tetris_game_0.save()

	# sys.exit()

	# cpu_amount = 6
	mult_proc_parallel_manager = MultiprocessingParallelManager(cpu_amount=cpu_amount)
	worker_amount = mult_proc_parallel_manager.worker_amount

	dir_path = os.path.join(TETRIS_TEMP_DIR, f'h_{field_h:02}_w_{field_w:02}_l_hidden_neurons_{l_hidden_neurons_str}')
	mkdirs(dir_path)

	l_kwargs = []
	for many_game_nr in range(0, worker_amount):
		l_seed_main = np.frombuffer(time.time_ns().to_bytes(8, 'little'), dtype=np.uint32).tolist()
		l_seed_prefix = np.frombuffer(time.time_ns().to_bytes(8, 'little'), dtype=np.uint32).tolist()

		kwargs = dict(
			dir_path=dir_path,
			many_game_nr=many_game_nr,
			h=field_h,
			w=field_w,
			l_hidden_neurons=l_hidden_neurons,
			l_seed_main=l_seed_main,
			l_seed_prefix=l_seed_prefix,
			d_params=d_params,
		)

		l_kwargs.append(kwargs)

	kwargs_accumulate = {
		'field_h': field_h,
		'field_w': field_w,
		'l_hidden_neurons_str': l_hidden_neurons_str,
		'min_amount': worker_amount,
		'max_amount': worker_amount,
		'l_seed': [0, 1],
		'amount_best_arr_bw': take_best_games*2,
	}
	# kwargs_accumulate = {'min_amount': 2, 'max_amount': worker_amount, 'l_seed': [0, 1], 'amount_best_arr_bw': 7}
	mult_proc_parallel_manager.init(iter_class=ManyTetrisGame, l_kwargs=l_kwargs, accumulate_class=AccumulateManyTetrisGame, kwargs_accumulate=kwargs_accumulate)

	l_args_next = [(max_round, ) for _ in range(0, (max_many_tetris_games_iter//worker_amount)*worker_amount)]
	df_games = mult_proc_parallel_manager.next(l_args_next=l_args_next, args_acc_next=())
	mult_proc_parallel_manager.save(l_args_iter_save=[() for _ in range(0, worker_amount)])

	del mult_proc_parallel_manager

	# l_seed_main[0] += 1
	# many_tetris_game_1 = ManyTetrisGame(
	# 	many_game_nr=1,
	# 	h=field_h,
	# 	w=field_w,
	# 	l_hidden_neurons=l_hidden_neurons,
	# 	l_seed_main=l_seed_main,
	# 	l_seed_prefix=l_seed_prefix,
	# 	d_params=d_params,
	# )
	# many_tetris_game_1.init_tetris_game_in_df_games()
	# many_tetris_game_1.play_many_tetris_game()

	# l_seed_main[0] += 1
	# many_tetris_game_2 = ManyTetrisGame(
	# 	many_game_nr=2,
	# 	h=field_h,
	# 	w=field_w,
	# 	l_hidden_neurons=l_hidden_neurons,
	# 	l_seed_main=l_seed_main,
	# 	l_seed_prefix=l_seed_prefix,
	# 	d_params=d_params,
	# )
	# many_tetris_game_2.init_tetris_game_in_df_games()
	# many_tetris_game_2.play_many_tetris_game()

	# print(f"many_tetris_game_0.df_games.iloc[:5]:\n{many_tetris_game_0.df_games.iloc[:5]}")
	# print(f"many_tetris_game_1.df_games.iloc[:5]:\n{many_tetris_game_1.df_games.iloc[:5]}")
	# print(f"many_tetris_game_2.df_games.iloc[:5]:\n{many_tetris_game_2.df_games.iloc[:5]}")
