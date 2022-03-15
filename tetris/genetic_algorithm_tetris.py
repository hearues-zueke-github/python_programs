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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

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
	print(f"arr_tetris_pieces_rotate:\n{arr_tetris_pieces_rotate}")

	pix_byte = np.zeros((7*4, 4*4), dtype=np.uint8)
	for piece_nr, arr_piece_rotate in enumerate(arr_tetris_pieces_rotate, 1):
		y0 = piece_nr*4 - 1
		for col, arr_yx in enumerate(arr_piece_rotate, 0):
			x0 = col*4

			pix_byte[y0 - arr_yx[0], x0 + arr_yx[1]] = piece_nr
			print(f"piece_nr: {piece_nr}, col: {col}, arr_yx.T.tolist(): {arr_yx.T.tolist()}")

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
			self.img_nr = 0
			self.dir_path = dir_path

			mkdirs(self.dir_path)
			files = glob.glob(os.path.join(self.dir_path, '*.png'))
			if len(files) > 0:
				sh.rm(files)

			self.factor_resize_img = 30
			self.field_h = h
			self.field_w = w

			self.arr_field = np.zeros((self.field_h, self.field_w))

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

			self.seed = np.array(l_seed, dtype=np.uint32)
			self.rnd = Generator(bit_generator=PCG64(seed=self.seed))

			# TODO: make a bag of pieces and choose one of them for the next piece
			self.piece_nr_idx_next = self.rnd.integers(0, 7, (1, ))[0]
			self.pieces_placed = 0
			self.lines_removed = 0
			self.lines_type_removed = [0] * 4 # single, double, 3 or 4 (tetros) lines removed

			self.nn = NeuralNetwork()
			self.nn.init_l_bw(l_node_amount=[7+h*w, 100, 50, w+4])

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

		def place_next_piece(self, orientation=0, x=0):
			piece_nr_idx = self.piece_nr_idx_next
			self.piece_nr_idx_next = self.rnd.integers(0, 7, (1, ))[0]

			# orientation = self.rnd.integers(0, 4, (1, ))[0]

			arr_x = np.zeros((self.nn.l_node_amount[0], ), dtype=np.float64)
			arr_x[:] = -1
			arr_x[piece_nr_idx] = 1

			arr_field_flat = self.arr_field.reshape((-1, )).copy()
			arr_idx_piece = (arr_field_flat > 0)
			arr_field_flat[arr_idx_piece] = 1
			arr_field_flat[~arr_idx_piece] = -1
			arr_x[7:] = arr_field_flat
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
				self.arr_field[:self.field_h-add_lines_removed] = self.arr_field[~arr_idx_full]
				self.arr_field[self.field_h-add_lines_removed:] = 0
				self.lines_removed += add_lines_removed
				self.lines_type_removed[add_lines_removed - 1] += 1

			self.pieces_placed += 1

			return is_first_piece_placeable

	n_pieces = arr_tetris_pieces_base.shape[0]
	field_w = 10
	field_h = 25

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

	take_best_games = 5
	max_game_nr = 50

	l_column = ['tetris_game', 'lines_type_removed_reverse', 'lines_removed', 'pieces_placed']
	df_games = pd.DataFrame(data=np.empty((max_game_nr, len(l_column))).tolist(), columns=l_column, dtype=object)

	# arr_tetris_game = np.array([None]*max_game_nr, dtype=object)
	max_pieces = 100
	for game_nr in range(0, max_game_nr):
		row = df_games.loc[game_nr]
		tetris_game = TetrisGame(h=field_h, w=field_w, dir_path=os.path.join(TETRIS_GAMEFIELD_IMAGES_DIR, f"game_{game_nr:03}"), l_seed=[0, game_nr])
		row['tetris_game'] = tetris_game
		# arr_tetris_game[game_nr] = tetris_game
		
		# tetris_game.draw_next_piece()
		# tetris_game.save_image()

		for iter_nr in range(1, max_pieces+1):
			is_first_piece_placeable = tetris_game.place_next_piece()
			# tetris_game.draw_next_piece()
			# tetris_game.save_image()

			if not is_first_piece_placeable:
				break

		print(f"game_nr: {game_nr:4}, tetris_game.pieces_placed: {tetris_game.pieces_placed}, tetris_game.lines_removed: {tetris_game.lines_removed}")

		row['lines_type_removed_reverse'] = tuple(tetris_game.lines_type_removed[::-1])
		row['lines_removed'] = tetris_game.lines_removed
		row['pieces_placed'] = tetris_game.pieces_placed

	df_games.sort_values(by=['lines_type_removed_reverse', 'lines_removed', 'pieces_placed'], inplace=True, ascending=False)
	df_games.reset_index(drop=True, inplace=True)

	print(f"df_games:\n{df_games}")



	# arr_pix[1:1+field_h, 1:1+field_w] = np.flip(arr_field, axis=0)
	# img = Image.fromarray(arr_color[arr_pix])
	# size = img.size
	# img = img.resize((size[0]*self.factor_resize_img, size[1]*self.factor_resize_img), resample=Image.NEAREST)
	# img.save(os.path.join(self.dir_path, f'img_{0:04}.png'))

	# for iter_nr in range(1, 100+1):
	# 	print(f"iter_nr: {iter_nr}")

	# 	piece_nr_idx = piece_nr_idx_next
	# 	piece_nr = piece_nr_idx + 1
	# 	orientation = rnd.integers(0, 4, (1, ))[0]

	# 	arr_yx = arr_tetris_pieces_rotate[piece_nr_idx][orientation]

	# 	max_pos_x = field_w - np.max(arr_yx[1])
	# 	max_y = np.max(arr_yx[0])

	# 	x = rnd.integers(0, max_pos_x, (1, ))[0]
	# 	y_prev = field_h - max_y - 1

	# 	y = y_prev - 1

	# 	while True:
	# 		if np.any(arr_field[arr_yx[0]+y, arr_yx[1]+x] != 0):
	# 			break

	# 		y_prev = y
	# 		y -= 1
	# 		if y < 0:
	# 			break

	# 	arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = piece_nr

	# 	# add the next piece here!
	# 	piece_nr_idx_next = rnd.integers(0, 7, (1, ))[0]
	# 	arr_yx = arr_tetris_pieces_rotate[piece_nr_idx_next][0]
	# 	piece_nr_next = piece_nr_idx_next + 1
	# 	arr_pix[8+1:8+1+4, field_w+2+1:field_w+2+1+4] = 0
	# 	arr_pix[8+4-arr_yx[0], arr_yx[1]+field_w+2+1] = piece_nr_next

	# 	arr_pix[1:1+field_h, 1:1+field_w] = np.flip(arr_field, axis=0)
	# 	img = Image.fromarray(arr_color[arr_pix])
	# 	factor = 30
	# 	size = img.size
	# 	img = img.resize((size[0]*factor, size[1]*factor), resample=Image.NEAREST)
	# 	img.save(os.path.join(TETRIS_GAMEFIELD_IMAGES_DIR, f'img_{iter_nr:04}.png'))
