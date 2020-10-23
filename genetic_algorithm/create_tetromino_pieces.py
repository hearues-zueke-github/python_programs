#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

import platform
print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

if __name__ == "__main__":
    l_piece_pos = [((0, 0), )] # base case

    def get_all_rotation_piece_pos(piece_pos):
        l_piece_pos_rotations = [piece_pos]
        for _ in range(0, 3):
            arr = np.array(piece_pos)
            arr = np.flip(arr, axis=1)
            arr[:, 1] = np.max(arr[:, 1])-arr[:, 1]
            piece_pos_rot = tuple(sorted(tuple(t for t in l) for l in arr.tolist()))
            piece_pos = piece_pos_rot
            l_piece_pos_rotations.append(piece_pos)

        return tuple(sorted(set(l_piece_pos_rotations)))
    
    piece_pos = ((0, 0), (0, 1), (1, 1), (1, 2)) # S piece
    t_piece_pos = get_all_rotation_piece_pos(piece_pos)
    
    print("piece_pos: {}".format(piece_pos))
    print("t_piece_pos: {}".format(t_piece_pos))

    d_l_piece_pos = {1: l_piece_pos}
    d_l_piece_pos_rotate_group = {1: [tuple(l_piece_pos)]}

    l_amount_rotated_pieces = [1]
    l_amount_unique_pieces = [1]

    for i in range(2, 7):
        l_pieces_pos_new = []
        for piece_pos in l_piece_pos:
            s_new_pos = set()
            for y, x in piece_pos:
                t = (y-1, x)
                if t not in s_new_pos and t not in piece_pos:
                    s_new_pos.add(t)
                t = (y+1, x)
                if t not in s_new_pos and t not in piece_pos:
                    s_new_pos.add(t)
                t = (y, x-1)
                if t not in s_new_pos and t not in piece_pos:
                    s_new_pos.add(t)
                t = (y, x+1)
                if t not in s_new_pos and t not in piece_pos:
                    s_new_pos.add(t)

            l_combined_pieces = []
            for new_pos in s_new_pos:
                l_combined_pieces.append(piece_pos+(new_pos, ))

            arr = np.array(l_combined_pieces)
            arr -= np.min(arr, axis=1).reshape((-1, 1, 2))

            l_combined_pieces_moved = [tuple(sorted(tuple(t for t in l) for l in l2)) for l2 in arr.tolist()]
            s_combined_pieces = set(l_combined_pieces_moved)

            l_pieces_pos_new.extend(s_combined_pieces)

        l_pieces_pos_new_unique = sorted(set(l_pieces_pos_new))
        l_piece_pos = l_pieces_pos_new_unique
        print("i: {}".format(i))
        print("len(l_piece_pos): {}".format(len(l_piece_pos)))

        # create different shape groups! need to rotate each shape btw.
        l_piece_pos_rotate_group = sorted(list(set(get_all_rotation_piece_pos(piece_pos) for piece_pos in l_piece_pos)), key=lambda x: (len(x), x))
        print("len(l_piece_pos_rotate_group): {}".format(len(l_piece_pos_rotate_group)))

        l_amount_rotated_pieces.append(len(l_piece_pos))
        l_amount_unique_pieces.append(len(l_piece_pos_rotate_group))

        d_l_piece_pos[i] = l_piece_pos
        d_l_piece_pos_rotate_group[i] = l_piece_pos_rotate_group

    print("l_amount_rotated_pieces: {}".format(l_amount_rotated_pieces))
    print("l_amount_unique_pieces: {}".format(l_amount_unique_pieces))


    def convert_l_piece_pos_rotate_group_to_list_int(l_piece_pos_rotate_group):
        num_blocks = len(l_piece_pos_rotate_group[0][0])
        num_groups = len(l_piece_pos_rotate_group)
        l_group_len = [len(l) for l in l_piece_pos_rotate_group]
        l_pos = [v for t1 in l_piece_pos_rotate_group for t2 in t1 for t3 in t2 for v in t3]
        l = [num_blocks, num_groups]+l_group_len+l_pos
        return l


    PATH_FOLDER_TETRIS_DATA = PATH_ROOT_DIR+'tetris_data/'
    if not os.path.exists(PATH_FOLDER_TETRIS_DATA):
        os.makedirs(PATH_FOLDER_TETRIS_DATA)

    for num_blocks in range(1, 7):
        l_piece_pos_rotate_group = d_l_piece_pos_rotate_group[num_blocks]
        l = convert_l_piece_pos_rotate_group_to_list_int(l_piece_pos_rotate_group)
        print("num_blocks: {}, len(l): {}".format(num_blocks, len(l)))
        
        bytes_content = bytes(l)
        with open(PATH_FOLDER_TETRIS_DATA+'tetris_pieces_block_amount_{}.trpcs'.format(num_blocks), 'wb') as f:
            f.write(bytes_content)
