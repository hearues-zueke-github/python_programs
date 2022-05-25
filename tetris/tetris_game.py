import glob
import os
import sys

import numpy as np

from PIL import Image

from numpy.random import Generator, PCG64

from neuronal_network import NeuralNetwork

HOME_DIR = os.path.expanduser("~")
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))

mkdirs = utils.mkdirs

class TetrisGame():

	def __init__(self, game_nr, h, w, n_pieces, arr_tetris_pieces_rotate, dir_path_picture, l_hidden_neurons=[], l_seed_main=[0, 0], l_seed=[0, 0]):
		# every single found
		# self.l_lines_cleared_l_arr_x = [[] for _ in range(0, arr_tetris_pieces_rotate.shape[1])]
		# self.l_lines_cleared_l_arr_y = [[] for _ in range(0, arr_tetris_pieces_rotate.shape[1])]

		self.dir_path_picture = dir_path_picture

		self.game_nr = game_nr

		self.factor_resize_img = 30
		self.field_h = h
		self.field_w = w
		self.n_pieces = n_pieces
		self.arr_tetris_pieces_rotate = arr_tetris_pieces_rotate

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
		self.y_next_piece = 3
		self.draw_frames()

		self.d_points_per_line_clear = {i: v for i, v in enumerate(np.cumsum(np.cumsum(np.arange(1, 4))), 1)}
		# self.d_points_per_line_clear = {1: 1, 2: 3, 3: 6, 4: 10}
		# self.d_points_per_line_clear = {k: v * field_w for k, v in self.d_points_per_line_clear.items()}

		self.seed_main = np.array(l_seed_main, dtype=np.uint32)

		self.seed_orig = np.array(l_seed, dtype=np.uint32)
		self.seed = self.seed_orig.copy()

		self.arr_field_cell_multiply = np.zeros((self.field_h, self.field_w), dtype=np.int64)
		self.arr_field_cell_multiply += np.arange(1, self.field_w+1)
		self.arr_field_cell_multiply += np.arange(1, self.field_h+1).reshape((-1, 1))

		# TODO: make a bag of pieces and choose one of them for the next piece
		self.init_starting_values()

		self.nn = NeuralNetwork(l_seed=self.seed_main)
		# self.nn.init_arr_bw(l_node_amount=[self.n_pieces*(1+self.amount_next_piece)+9*(self.field_w-1)] + l_hidden_neurons + [self.field_w+4])
		self.nn.init_arr_bw(l_node_amount=[self.n_pieces*(1+self.amount_next_piece)+h*w] + l_hidden_neurons + [w+4])

		# self.nn.init_arr_bw(l_node_amount=[self.n_pieces+self.n_pieces+h*w, 100, 100, w+4])

	def init_tetris_bag(self):
		self.arr_piece_bag = np.arange(0, self.n_pieces)
		self.bag_index = 0
		self.arr_piece_bag[:] = self.rnd.permutation(self.arr_piece_bag)

	def get_next_tetris_piece(self):
		piece_nr_idx = self.arr_piece_bag[self.bag_index]

		self.bag_index += 1
		if self.bag_index >= self.arr_piece_bag.shape[0]:
			self.arr_piece_bag[:] = self.rnd.permutation(self.arr_piece_bag)
			self.bag_index = 0

		return piece_nr_idx

	def increment_seed(self):
		self.seed[:] += 1

	def init_starting_values(self):
		self.rnd = Generator(bit_generator=PCG64(seed=self.seed))
		self.arr_field = np.zeros((self.field_h, self.field_w), dtype=np.uint8)

		self.init_tetris_bag()
		
		self.img_nr = 0
		self.amount_next_piece = 2
		self.arr_piece_nr_idx_next = np.array([self.get_next_tetris_piece() for _ in range(0, self.amount_next_piece)])
		# self.arr_piece_nr_idx_next = self.rnd.integers(0, self.n_pieces, (self.amount_next_piece, ))
		self.lines_removed_points = 0
		self.points = 0
		self.pieces_placed = 0
		self.lines_removed = 0
		self.lines_type_removed = [0] * 4 # single, double, 3 or 4 (tetros) lines removed
		self.arr_used_piece_idx = np.zeros((self.n_pieces, ), dtype=np.int64)
		self.arr_removing_lines = np.zeros((self.field_h, ), dtype=np.int64)

		self.piece_cell_rest_points = 0

	def draw_frames(self):
		self.arr_pix[0:self.field_h+2, 0] = 8
		self.arr_pix[0:self.field_h+2, self.field_w+1] = 8
		self.arr_pix[0, 0:self.field_w+2] = 8
		self.arr_pix[self.field_h+1, 0:self.field_w+2] = 8

		self.arr_pix[self.y_next_piece, self.field_w+2:self.field_w+2+6] = 9
		self.arr_pix[self.y_next_piece+5, self.field_w+2:self.field_w+2+6] = 9
		self.arr_pix[self.y_next_piece:self.y_next_piece+5, self.field_w+2] = 9
		self.arr_pix[self.y_next_piece:self.y_next_piece+5, self.field_w+2+5] = 9

	def draw_next_piece(self):
		arr_yx = self.arr_tetris_pieces_rotate[self.arr_piece_nr_idx_next[0]][0]
		self.arr_pix[self.y_next_piece+1:self.y_next_piece+1+4, self.field_w+2+1:self.field_w+2+1+4] = 0
		self.arr_pix[self.y_next_piece+4-arr_yx[0], arr_yx[1]+self.field_w+2+1] = self.arr_piece_nr_idx_next[0] + 1

	def save_image(self):
		self.arr_pix[1:1+self.field_h, 1:1+self.field_w] = np.flip(self.arr_field, axis=0)
		img = Image.fromarray(self.arr_color[self.arr_pix])
		size = img.size
		img = img.resize((size[0]*self.factor_resize_img, size[1]*self.factor_resize_img), resample=Image.NEAREST)
		img.save(os.path.join(self.dir_path_picture, f'img_{self.img_nr:04}.png'))
		self.img_nr += 1

	def play_the_game(self, max_pieces=100):
		self.init_starting_values()

		for iter_nr in range(0, max_pieces):
			is_piece_placeable = self.place_next_piece()
			# tetris_game.draw_next_piece()
			# tetris_game.save_image()

			if not is_piece_placeable:
				break
		
		self.piece_cell_rest_points = -np.sum(self.arr_field_cell_multiply[self.arr_field != 0])

	def play_the_game_with_images_saving(self, max_pieces=100):
		mkdirs(self.dir_path_picture)
		files = glob.glob(os.path.join(self.dir_path_picture, '*.png'))
		if len(files) > 0:
			sh.rm(files)

		self.init_starting_values()

		self.draw_next_piece()
		self.save_image()
		for iter_nr in range(0, max_pieces):
			is_piece_placeable = self.place_next_piece()
			self.draw_next_piece()
			self.save_image()

			if not is_piece_placeable:
				break

		self.piece_cell_rest_points = -np.sum(self.arr_field_cell_multiply[self.arr_field != 0])

	def place_next_piece(self, orientation=0, x=0):
		piece_nr_idx = self.arr_piece_nr_idx_next[0]
		self.arr_piece_nr_idx_next[:-1] = self.arr_piece_nr_idx_next[1:]
		# self.arr_piece_nr_idx_next[0] = self.arr_piece_nr_idx_next[1]
		self.arr_piece_nr_idx_next[-1] = self.get_next_tetris_piece() # TODO: make a function for getting the next piece from the bag!
		# self.arr_piece_nr_idx_next[-1] = self.rnd.integers(0, 7, (1, ))[0] # TODO: make a function for getting the next piece from the bag!

		# orientation = self.rnd.integers(0, 4, (1, ))[0]

		arr_x = np.zeros((self.nn.l_node_amount[0], ), dtype=np.float64)
		arr_x[:] = -1
		arr_x[7*0+piece_nr_idx] = 1
		for i, v_p in enumerate(self.arr_piece_nr_idx_next, 1):
			arr_x[self.n_pieces*i+v_p] = 1

		arr_height = np.argmin((np.flip(np.vstack(((1, )*self.arr_field.shape[1], self.arr_field)).T, 1) == 0) - 1, 1)
		arr_height_diff = np.diff(arr_height)
		arr_height_diff_sign = np.sign(arr_height_diff)
		arr_height_diff_abs = np.abs(arr_height_diff)

		arr_field_flat = self.arr_field.reshape((-1, )).copy()
		arr_idx_piece = (arr_field_flat > 0)
		arr_field_flat[arr_idx_piece] = 1
		arr_field_flat[~arr_idx_piece] = -1
		arr_x[self.n_pieces*(1+self.amount_next_piece):] = arr_field_flat

		# max_val_diff = 4
		# arr_idx = arr_height_diff_abs > max_val_diff
		# arr_height_diff_prep = arr_height_diff.copy()
		# if np.any(arr_idx):
		# 	arr_height_diff_prep[arr_idx] = arr_height_diff_sign[arr_idx] * max_val_diff

		# arr_x[self.n_pieces*(1+self.amount_next_piece) + (arr_height_diff_prep+max_val_diff)+np.arange(0, arr_height_diff.shape[0])*9] = 1
		
		arr_y = self.nn.calc_feed_forward(X=arr_x.reshape((1, -1)))[0]

		orientation = np.argmax(arr_y[self.field_w:])

		self.arr_used_piece_idx[piece_nr_idx] += 1

		arr_yx = self.arr_tetris_pieces_rotate[piece_nr_idx][orientation]

		max_pos_x = self.field_w - np.max(arr_yx[1])
		max_y = np.max(arr_yx[0])

		# x = self.rnd.integers(0, max_pos_x, (1, ))[0]
		x = np.argmax(arr_y[:max_pos_x])
		y_prev = self.field_h - max_y - 1

		y = y_prev - 1
		y_start = y

		is_piece_placeable = False
		while True:
			if np.any(self.arr_field[arr_yx[0]+y, arr_yx[1]+x] != 0):
				break

			y_prev = y
			y -= 1
			if y < 0:
				break

			is_piece_placeable = True

		# self.arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = 255
		self.arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = piece_nr_idx + 1

		self.points += (y_start - y)

		# delete all rows, which are full!
		arr_idx_full = np.all(self.arr_field != 0, axis=1)
		if np.any(arr_idx_full):
			add_lines_removed = np.sum(arr_idx_full)
			
			self.lines_removed_points += self.d_points_per_line_clear[add_lines_removed]
			self.points += self.d_points_per_line_clear[add_lines_removed] * 100

			# self.l_lines_cleared_l_arr_x[add_lines_removed - 1].append(arr_x)
			# self.l_lines_cleared_l_arr_y[add_lines_removed - 1].append(arr_y)

			# self.arr_removing_lines[arr_idx_full] += np.sum(self.arr_field[arr_idx_full] == 255, axis=0)
			self.arr_removing_lines[arr_idx_full] += 1
			# self.arr_field[arr_yx[0]+y_prev, arr_yx[1]+x] = piece_nr_idx + 1

			self.arr_field[:self.field_h-add_lines_removed] = self.arr_field[~arr_idx_full]
			self.arr_field[self.field_h-add_lines_removed:] = 0
			self.lines_removed += add_lines_removed
			self.lines_type_removed[add_lines_removed - 1] += 1


		self.pieces_placed += 1

		return is_piece_placeable
