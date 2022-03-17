#! /usr/bin/python3.10

# pip installed libraries
import dill
import glob
import gzip
import os
import requests
import sh
import string
import subprocess
import sys
import time
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd


import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from io import StringIO
from memory_tempfile import MemoryTempfile

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

from PIL import Image
from typing import Dict

from neuronal_network import NeuralNetwork

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

TETRIS_TEMP_DIR = os.path.join(TEMP_DIR, "tetris_temp")
TETRIS_GAMEFIELD_IMAGES_DIR = os.path.join(TETRIS_TEMP_DIR, "tetris_gamefield_images")
mkdirs(TETRIS_GAMEFIELD_IMAGES_DIR)

if __name__ == '__main__':
	print("Hello World!")
	# TODO: create a simple network for differnet tetris pieces

	# list of y, x position of the base orientation
	# all points are in the upper right quartet, so all y and x values are >=0
	arr_tetris_pieces_base = np.array([
		[[0, 0], [0, 1], [1, 0], [1, 1]], # O
		[[0, 0], [0, 1], [0, 2], [0, 3]], # I
		[[0, 0], [0, 1], [1, 1], [1, 2]], # S
		[[1, 0], [1, 1], [0, 1], [0, 2]], # Z
		[[0, 0], [0, 1], [0, 2], [1, 1]], # T
		[[0, 0], [0, 1], [0, 2], [1, 0]], # J
		[[0, 0], [0, 1], [0, 2], [1, 2]], # L
	], dtype=np.int32).transpose(0, 2, 1)

	l_tetris_pieces_rotate = []

	def rotate_piece(arr_yx):
		arr_yx = arr_yx.copy()
		arr_yx = np.flip(arr_yx, axis=0)
		arr_yx[0] *= -1
		arr_yx[0] = arr_yx[0] - np.min(arr_yx[0])
		return arr_yx

	for arr_yx in arr_tetris_pieces_base:
		l_piece_rotate = [arr_yx]
		for _ in range(0, 3):
			arr_yx = rotate_piece(arr_yx=arr_yx)
			l_piece_rotate.append(arr_yx.copy())

		l_tetris_pieces_rotate.append(l_piece_rotate)
	
	arr_tetris_pieces_rotate = np.array(l_tetris_pieces_rotate)
	# print(f"arr_tetris_pieces_rotate:\n{arr_tetris_pieces_rotate}")

	pix_byte = np.zeros((7*4, 4*4), dtype=np.uint8)
	for piece_nr, arr_piece_rotate in enumerate(arr_tetris_pieces_rotate, 1):
		y0 = piece_nr*4 - 1
		for col, arr_yx in enumerate(arr_piece_rotate, 0):
			x0 = col*4

			pix_byte[y0 - arr_yx[0], x0 + arr_yx[1]] = piece_nr
			# print(f"piece_nr: {piece_nr}, col: {col}, arr_yx.T.tolist(): {arr_yx.T.tolist()}")

	n_pieces = arr_tetris_pieces_base.shape[0]
	field_w = 5
	field_h = 15

	l_column = ['piece_nr', 'piece_name', 'orient_nr', 'x', 'arr_yx_base', 'arr_yx']
	d_data = {column: [] for column in l_column}

	l_piece_name = ['O', 'I', 'S', 'Z', 'T', 'J', 'L']
	for piece_nr, (piece_name, arr_tetris_piece_rotate) in enumerate(zip(
		l_piece_name,
		arr_tetris_pieces_rotate
	), 0):
		for orient_nr, arr_yx_base in enumerate(arr_tetris_piece_rotate, 0):
			max_x = np.max(arr_yx[1])
			for x in range(0, field_w - max_x - 1):
				d_data['piece_nr'].append(piece_nr)
				d_data['piece_name'].append(piece_name)
				d_data['orient_nr'].append(orient_nr)
				d_data['x'].append(x)
				d_data['arr_yx_base'].append(arr_yx_base)
				d_data['arr_yx'].append(arr_yx_base + ((0, ), (x, )))

	df_piece_positions = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

	# TODO: create a function for creating different examples for the training set for the neuronal network

	# arr_max_x = np.max(arr_tetris_pieces_rotate[:, :, 1], axis=2)

	class TetrisGameSimpleGenetic():
		def __init__(self, field_h, field_w, seed_piece, seed_weight):
			self.field_h = field_h
			self.field_w = field_w

			self.d_points_per_line_clear = {0: 0, 1: 1, 2: 3, 3: 6, 4: 10}
			self.seed_piece = seed_piece
			self.seed_weight = seed_weight
			self.is_filling_with_tetris_piece_nr = False

			self.init_values()

			self.amount_weights = len(self.get_board_info())
			self.rnd_weight = Generator(bit_generator=PCG64(seed=self.seed_weight))
			self.init_weights()

		def init_values(self):
			self.rnd_piece = Generator(bit_generator=PCG64(seed=self.seed_piece))

			self.arr_field = np.zeros((self.field_h, self.field_w), dtype=np.uint8)
			self.arr_field_prev = self.arr_field.copy()

			self.piece_nr_idx_next = self.rnd_piece.integers(0, 7, (1, ))[0]
			self.points = 0
			self.pieces_placed = 0
			self.lines_removed = 0
			self.lines_removed_prev = 0
			self.lines_type_removed = [0] * 4 # single, double, 3 or 4 (tetris) lines removed

		def init_weights(self):
			self.weights = (self.rnd_weight.random((self.amount_weights, )) - 0.5) * 2 # set the random values to -1...1

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
			next_piece_nr = self.piece_nr_idx_next
			df_piece_positions_part = df_piece_positions.loc[df_piece_positions['piece_nr'].values == next_piece_nr]

			max_pieces = 100
			for piece_nr in range(1, max_pieces+1):
				print(f"piece_nr: {piece_nr}")
				l_fitness = []
				l_add_lines_removed = []
				l_is_first_piece_placeable = []
				for i, row in df_piece_positions_part.iterrows():
					self.set_prev_field()
					add_lines_removed, is_first_piece_placeable = self.place_next_piece_with_arr_yx(arr_yx=row['arr_yx_base'], x=row['x'])
					tpl = self.get_board_info()
					fitness = np.sum(self.weights * tpl)
					l_fitness.append(fitness)
					l_add_lines_removed.append(add_lines_removed)
					l_is_first_piece_placeable.append(is_first_piece_placeable)

				idx_best_fitness = np.argmax(l_fitness)
				
				if not l_is_first_piece_placeable[idx_best_fitness]:
					break

				row = df_piece_positions_part.iloc[idx_best_fitness]
				self.set_prev_field()
				add_lines_removed, is_first_piece_placeable = self.place_next_piece_with_arr_yx(arr_yx=row['arr_yx_base'], x=row['x'])
				self.update_prev_field()
				assert add_lines_removed == l_add_lines_removed[idx_best_fitness]
				self.update_stats(add_lines_removed=add_lines_removed)

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

	tetris_game = TetrisGameSimpleGenetic(field_h=field_h, field_w=field_w, seed_piece=[0, 0, 1], seed_weight=[0, 0, 1])

	next_piece_nr = tetris_game.piece_nr_idx_next
	df_piece_positions_part = df_piece_positions.loc[df_piece_positions['piece_nr'].values == next_piece_nr]

	max_pieces = 1000
	tetris_game.play_the_game(max_pieces=max_pieces)

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

	sys.exit()

	arr_color = np.array([
		[0x00, 0x00, 0x00],
		[0x00, 0x00, 0xFF],
		[0x00, 0xFF, 0x00],
		[0x00, 0xFF, 0xFF],
		[0xFF, 0x00, 0x00],
		[0xFF, 0x00, 0xFF],
		[0xFF, 0xFF, 0x00],
		[0xFF, 0xFF, 0xFF],
		[0x80, 0x80, 0x80], # frame color for the gamefield
		[0x60, 0x90, 0x20], # frame color for the next piece
	], dtype=np.uint8)

	class TetrisGame():
		def __init__(self, h, w, dir_path, l_seed=[0, 0]):
			# every single found
			# self.l_lines_cleared_l_arr_x = [[] for _ in range(0, arr_tetris_pieces_rotate.shape[1])]
			# self.l_lines_cleared_l_arr_y = [[] for _ in range(0, arr_tetris_pieces_rotate.shape[1])]

			self.dir_path = dir_path

			self.factor_resize_img = 30
			self.field_h = h
			self.field_w = w

			# the first colors are the colors of the pieces
			self.arr_color = np.array([
				[0x00, 0x00, 0x00], # background color
				[0x00, 0x00, 0xFF],
				[0x00, 0xFF, 0x00],
				[0x00, 0xFF, 0xFF],
				[0xFF, 0x00, 0x00],
				[0xFF, 0x00, 0xFF],
				[0xFF, 0xFF, 0x00],
				[0xFF, 0xFF, 0xFF],
				[0x80, 0x80, 0x80], # frame color for the gamefield
				[0x60, 0x90, 0x20], # frame color for the next piece
			], dtype=np.uint8)

			self.arr_pix = np.zeros((self.field_h + 2, self.field_w + 2 + 6), dtype=np.uint8)
			self.draw_frames()

			self.d_points_per_line_clear = {1: 1, 2: 3, 3: 6, 4: 10}

			self.seed = np.array(l_seed, dtype=np.uint32)

			# TODO: make a bag of pieces and choose one of them for the next piece
			self.init_starting_values()

			self.nn = NeuralNetwork()
			# self.nn.init_arr_bw(l_node_amount=[7+7+h*w, 20, w+4])
			self.nn.init_arr_bw(l_node_amount=[7+7+h*w, 100, 100, w+4])
			# self.nn.init_arr_bw(l_node_amount=[7+7+h*w, 500, w+4])
			# self.nn.init_arr_bw(l_node_amount=[7+h*w, 500, 100, w+4])
			# self.nn.init_arr_bw(l_node_amount=[7+h*w, 500, 200, 100, w+4])

		def init_starting_values(self):
			self.rnd = Generator(bit_generator=PCG64(seed=self.seed))
			self.arr_field = np.zeros((self.field_h, self.field_w), dtype=np.uint8)
			
			self.img_nr = 0
			self.piece_nr_idx_next = self.rnd.integers(0, 7, (1, ))[0]
			self.points = 0
			self.pieces_placed = 0
			self.lines_removed = 0
			self.lines_type_removed = [0] * 4 # single, double, 3 or 4 (tetros) lines removed

		def draw_frames(self):
			self.arr_pix[0:self.field_h+2, 0] = 8
			self.arr_pix[0:self.field_h+2, self.field_w+1] = 8
			self.arr_pix[0, 0:self.field_w+2] = 8
			self.arr_pix[self.field_h+1, 0:self.field_w+2] = 8

			self.arr_pix[8, self.field_w+2:self.field_w+2+6] = 9
			self.arr_pix[8+5, self.field_w+2:self.field_w+2+6] = 9
			self.arr_pix[8:8+5, self.field_w+2] = 9
			self.arr_pix[8:8+5, self.field_w+2+5] = 9

		def draw_next_piece(self):
			arr_yx = arr_tetris_pieces_rotate[self.piece_nr_idx_next][0]
			self.arr_pix[8+1:8+1+4, self.field_w+2+1:self.field_w+2+1+4] = 0
			self.arr_pix[8+4-arr_yx[0], arr_yx[1]+self.field_w+2+1] = self.piece_nr_idx_next + 1

		def save_image(self):
			self.arr_pix[1:1+self.field_h, 1:1+self.field_w] = np.flip(self.arr_field, axis=0)
			img = Image.fromarray(self.arr_color[self.arr_pix])
			size = img.size
			img = img.resize((size[0]*self.factor_resize_img, size[1]*self.factor_resize_img), resample=Image.NEAREST)
			img.save(os.path.join(self.dir_path, f'img_{self.img_nr:04}.png'))
			self.img_nr += 1

		def play_the_game(self, max_pieces=100):
			self.init_starting_values()

			for iter_nr in range(0, max_pieces):
				is_first_piece_placeable = self.place_next_piece()
				# tetris_game.draw_next_piece()
				# tetris_game.save_image()

				if not is_first_piece_placeable:
					break

		def play_the_game_with_images_saving(self, max_pieces=100):
			mkdirs(self.dir_path)
			files = glob.glob(os.path.join(self.dir_path, '*.png'))
			if len(files) > 0:
				sh.rm(files)

			self.init_starting_values()

			tetris_game.draw_next_piece()
			tetris_game.save_image()
			for iter_nr in range(0, max_pieces):
				is_first_piece_placeable = self.place_next_piece()
				tetris_game.draw_next_piece()
				tetris_game.save_image()

				if not is_first_piece_placeable:
					break

		def place_next_piece(self, orientation=0, x=0):
			piece_nr_idx = self.piece_nr_idx_next
			self.piece_nr_idx_next = self.rnd.integers(0, 7, (1, ))[0]

			# orientation = self.rnd.integers(0, 4, (1, ))[0]

			arr_x = np.zeros((self.nn.l_node_amount[0], ), dtype=np.float64)
			arr_x[:] = -1
			arr_x[piece_nr_idx] = 1
			arr_x[7+self.piece_nr_idx_next] = 1

			arr_field_flat = self.arr_field.reshape((-1, )).copy()
			arr_idx_piece = (arr_field_flat > 0)
			arr_field_flat[arr_idx_piece] = 1
			arr_field_flat[~arr_idx_piece] = -1
			arr_x[7*2:] = arr_field_flat
			arr_y = self.nn.calc_feed_forward(X=arr_x.reshape((1, -1)))[0]

			orientation = np.argmax(arr_y[self.field_w:])
			arr_yx = arr_tetris_pieces_rotate[piece_nr_idx][orientation]

			max_pos_x = self.field_w - np.max(arr_yx[1])
			max_y = np.max(arr_yx[0])

			# x = self.rnd.integers(0, max_pos_x, (1, ))[0]
			x = np.argmax(arr_y[:max_pos_x])
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

			self.arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = piece_nr_idx + 1

			# delete all rows, which are full!
			arr_idx_full = np.all(self.arr_field != 0, axis=1)
			if np.any(arr_idx_full):
				add_lines_removed = np.sum(arr_idx_full)
				
				self.points += self.d_points_per_line_clear[add_lines_removed]

				# self.l_lines_cleared_l_arr_x[add_lines_removed - 1].append(arr_x)
				# self.l_lines_cleared_l_arr_y[add_lines_removed - 1].append(arr_y)

				self.arr_field[:self.field_h-add_lines_removed] = self.arr_field[~arr_idx_full]
				self.arr_field[self.field_h-add_lines_removed:] = 0
				self.lines_removed += add_lines_removed
				self.lines_type_removed[add_lines_removed - 1] += 1

			self.pieces_placed += 1

			return is_first_piece_placeable

	# field_w = 10
	# field_h = 25

	# # do a simple predefined game!
	# game_nr = 100
	# tetris_game = TetrisGame(h=field_h, w=field_w, dir_path=os.path.join(TETRIS_GAMEFIELD_IMAGES_DIR, f"game_{game_nr:03}"), l_seed=[0, 4])

	# tetris_game.piece_nr_idx_next = 4
	# tetris_game.draw_next_piece()
	# tetris_game.save_image()

	# l_orient_x_next_piece = [
	# 	# (0, 0, 0),
	# 	# (0, 2, 0),
	# 	# (0, 4, 0),
		
	# 	# (0, 0, 0),
	# 	# (0, 2, 0),
	# 	# (0, 4, 0),
		
	# 	# (0, 0, 0),
	# 	# (0, 2, 0),
	# 	# (0, 4, 0),

	# 	# (0, 6, 0),
	# 	# (0, 6, 0),
	# 	# (0, 6, 0),
	# 	(0, 0, 4),
	# 	(0, 3, 4),
	# 	(0, 6, 4),

	# 	(0, 0, 4),
	# 	(0, 3, 4),
	# 	(0, 6, 4),

	# 	(3, 8, 4),
	# 	(3, 5, 4),
	# 	(3, 2, 4),
	# 	(1, 0, 4),
		
	# 	(3, 4, 4),
	# 	(3, 7, 4),
	# 	(2, 1, 4),
	# ]

	# for orient, x, next_piece in l_orient_x_next_piece:
	# 	tetris_game.place_next_piece(orientation=orient, x=x)
	# 	tetris_game.piece_nr_idx_next = next_piece
	# 	tetris_game.draw_next_piece()
	# 	tetris_game.save_image()

	# sys.exit()

	take_best_games = 10
	max_game_nr = 100

	l_column = ['tetris_game', 'points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed']
	df_games = pd.DataFrame(data=np.empty((max_game_nr, len(l_column))).tolist(), columns=l_column, dtype=object)

	# arr_tetris_game = np.array([None]*max_game_nr, dtype=object)
	max_pieces = 1000
	for game_nr in range(0, max_game_nr):
		row = df_games.loc[game_nr]
		tetris_game = TetrisGame(h=field_h, w=field_w, dir_path=os.path.join(TETRIS_GAMEFIELD_IMAGES_DIR, f"game_{game_nr:03}"), l_seed=[0, game_nr])
		row['tetris_game'] = tetris_game
		# arr_tetris_game[game_nr] = tetris_game
		
		# tetris_game.draw_next_piece()
		# tetris_game.save_image()

		tetris_game.play_the_game(max_pieces=max_pieces)

		# for iter_nr in range(0, max_pieces):
		# 	is_first_piece_placeable = tetris_game.place_next_piece()
		# 	# tetris_game.draw_next_piece()
		# 	# tetris_game.save_image()

		# 	if not is_first_piece_placeable:
		# 		break

		print(f"game_nr: {game_nr:4}, tetris_game.pieces_placed: {tetris_game.pieces_placed}, tetris_game.lines_removed: {tetris_game.lines_removed}")

		row['points'] = tetris_game.points
		# row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed)
		row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
		row['lines_removed'] = tetris_game.lines_removed
		row['pieces_placed'] = tetris_game.pieces_placed

	df_games.sort_values(by=['points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed'], inplace=True, ascending=False)
	df_games.reset_index(drop=True, inplace=True)
	print(f"df_games init:\n{df_games}")

	mix_rate = 0.70
	change_factor = 0.10
	random_rate = 10.00


	for i_round in range(1, 40000+1):
		for game_nr_current in range(take_best_games, max_game_nr):
			arr_idx_game = np.random.permutation(np.arange(0, take_best_games))[:2]
			tetris_game_1 = df_games.loc[arr_idx_game[0]]['tetris_game']
			tetris_game_2 = df_games.loc[arr_idx_game[1]]['tetris_game']

			row = df_games.loc[game_nr_current]
			tetris_game = row['tetris_game']

			# tetris_game.nn.mix_arr_bw(tetris_game_1.nn.arr_bw, mix_rate=mix_rate_1)
			# tetris_game.nn.mix_arr_bw(tetris_game_2.nn.arr_bw, mix_rate=mix_rate_2)
			# tetris_game.nn.mutate_arr_bw(random_rate=random_rate, change_rate=change_rate)

			tetris_game.nn.crossover_and_mutate(arr_bw_1=tetris_game_1.nn.arr_bw, arr_bw_2=tetris_game_2.nn.arr_bw, mix_rate=mix_rate, random_rate=random_rate, change_factor=change_factor)

			# play again
			tetris_game.play_the_game(max_pieces=max_pieces)

			row['points'] = tetris_game.points
			# row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed)
			row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
			row['lines_removed'] = tetris_game.lines_removed
			row['pieces_placed'] = tetris_game.pieces_placed

		df_games.sort_values(by=['points', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed'], inplace=True, ascending=False)
		df_games.reset_index(drop=True, inplace=True)
		print(f"df_games round {i_round:5}:\n{df_games.loc[:take_best_games*2-1]}")

	# l_arr_x_1 = []
	# l_arr_y_1 = []
	# for tetris_game in df_games['tetris_game']:
	# 	l_arr_x_1.extend(tetris_game.l_lines_cleared_l_arr_x[0])
	# 	l_arr_y_1.extend(tetris_game.l_lines_cleared_l_arr_y[0])

	# df_arr_x_y_1 = pd.DataFrame(data={'arr_x_1': l_arr_x_1, 'arr_y_1': l_arr_y_1}, columns=['arr_x_1', 'arr_y_1'])
