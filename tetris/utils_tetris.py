import numpy as np
import pandas as pd

from typing import Dict, Any

def prepare_pieces(field_w: int) -> Dict[str, Any]:
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

	l_column = ['piece_nr', 'piece_name', 'orient_nr', 'x', 'arr_yx_base', 'arr_yx']
	d_data = {column: [] for column in l_column}

	l_piece_name = ['O', 'I', 'S', 'Z', 'T', 'J', 'L']
	for piece_nr, (piece_name, arr_tetris_piece_rotate) in enumerate(zip(
		l_piece_name,
		arr_tetris_pieces_rotate
	), 0):
		for orient_nr, arr_yx_base in enumerate(arr_tetris_piece_rotate, 0):
			max_x = np.max(arr_yx_base[1])
			for x in range(0, field_w - max_x):
				d_data['piece_nr'].append(piece_nr)
				d_data['piece_name'].append(piece_name)
				d_data['orient_nr'].append(orient_nr)
				d_data['x'].append(x)
				d_data['arr_yx_base'].append(arr_yx_base)
				d_data['arr_yx'].append(arr_yx_base + ((0, ), (x, )))

	df_piece_positions = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

	d_pieces = dict(
		arr_tetris_pieces_rotate=arr_tetris_pieces_rotate,
		df_piece_positions=df_piece_positions,
		arr_tetris_pieces_base=arr_tetris_pieces_base,
	)

	return d_pieces
