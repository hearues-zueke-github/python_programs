#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import random
import sys

from pysat.solvers import Glucose3

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

# find all possible new positions! (8-neighborhood)
def get_free_cells(s_used_cells):
    s_free_cells = set()
    for y, x in s_used_cells:
        new_cell = (y+0, x+1)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y+0, x-1)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y+1, x+0)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y-1, x+0)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y+1, x+1)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y+1, x-1)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y-1, x+1)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
        new_cell = (y-1, x-1)
        if not new_cell in s_used_cells and not new_cell in s_free_cells:
            s_free_cells.add(new_cell)
    return s_free_cells


# get the amount of max length per cell for a set of cells!
def get_amount_of_lengths(cell, s_cells):
    y, x = cell

    i1_1 = 1
    while True:
        next_cell = (y, x+i1_1)
        if not next_cell in s_cells:
            break
        i1_1 += 1
    i1_2 = 1
    while True:
        next_cell = (y, x-i1_2)
        if not next_cell in s_cells:
            break
        i1_2 += 1

    i2_1 = 1
    while True:
        next_cell = (y+i2_1, x+i2_1)
        if not next_cell in s_cells:
            break
        i2_1 += 1
    i2_2 = 1
    while True:
        next_cell = (y-i2_2, x-i2_2)
        if not next_cell in s_cells:
            break
        i2_2 += 1

    i3_1 = 1
    while True:
        next_cell = (y+i3_1, x)
        if not next_cell in s_cells:
            break
        i3_1 += 1
    i3_2 = 1
    while True:
        next_cell = (y-i3_2, x)
        if not next_cell in s_cells:
            break
        i3_2 += 1

    i4_1 = 1
    while True:
        next_cell = (y+i4_1, x-i4_1)
        if not next_cell in s_cells:
            break
        i4_1 += 1
    i4_2 = 1
    while True:
        next_cell = (y-i4_2, x+i4_2)
        if not next_cell in s_cells:
            break
        i4_2 += 1

    return (i1_1+i1_2-1, i2_1+i2_2-1, i3_1+i3_2-1, i4_1+i4_2-1)


def get_set_of_best_cells(s_used_cells, s_x_cells):
        s_free_cells = get_free_cells(s_used_cells)
        max_length = 0
        max_amount = 0
        s_best_cells = set()
        for cell in s_free_cells:
            t = get_amount_of_lengths(cell, s_x_cells)
            u, c = np.unique(t, return_counts=True)
            i_max = np.argmax(u)
            best_length = u[i_max]
            best_amount = c[i_max]

            if best_length>max_length:
                max_length = best_length
                max_amount = best_amount
                s_best_cells = set([cell])
                continue

            if best_length==max_length and best_amount>max_amount:
                max_amount = best_amount
                s_best_cells = set([cell])
                continue

            if best_length!=max_length or best_amount!=max_amount:
                continue

            s_best_cells.add(cell)
        return s_best_cells, max_length, max_amount

s1_used = set([(0, 0), (0, 1), (1, 0)])
s1_x = set([(0, 0), (1, 0)])
s_best_cells, max_length, max_amount = get_set_of_best_cells(s1_used, s1_x)
assert s_best_cells=={(2, 0), (-1, 0)}
assert max_length==3
assert max_amount==1

s1_used = set([(0, 0), (0, 1), (1, 0), (1, 1), (-1, 0)])
s1_x = set([(0, 0), (1, 0), (1, 1)])
s_best_cells, max_length, max_amount = get_set_of_best_cells(s1_used, s1_x)
assert s_best_cells=={(2, 0), (-1, -1), (1, -1), (1, 2), (2, 2)}
assert max_length==3
assert max_amount==1


def find_best_next_move_for_x_and_o(l_stats, s_used_cells, s_x_cells, s_o_cells, max_depth=3, l_used_o_x_cells=[], l_length_amount=[]):
    assert s_o_cells&s_x_cells==set()
    assert s_used_cells==s_o_cells|s_x_cells
    assert all([len(l)%2==0 for l in l_used_o_x_cells])

    if max_depth<=0:
        # print("- s_x_cells: {}".format(s_x_cells))
        # print("- s_o_cells: {}".format(s_o_cells))
        # d_counts = {}
        # t_complete = ()
        # for x_cell in s_x_cells:
        #     t = get_amount_of_lengths(x_cell, s_x_cells)
        #     t_complete += t
        # arr_u, arr_c = np.unique(t, return_counts=True)
        # d_counts = {}
        # for u, c in zip(arr_u, arr_c):
        #     if not u in d_counts:
        #         d_counts[u] = 0
        #     d_counts[u] += c
        # print("- d_counts: {}".format(d_counts))
        # l_stats.append((l_used_o_x_cells, tuple(((key, d_counts[key]) for key in sorted(d_counts.keys(), reverse=True)))))
        # l_stats.append((l_used_o_x_cells, d_counts))
        l_stats.append((l_used_o_x_cells, l_length_amount))
        return

    s_o_best_cells, max_length_o, max_amount_o = get_set_of_best_cells(s_used_cells, s_x_cells)
    if len(l_used_o_x_cells)==0:
        print("len(s_o_best_cells): {}".format(len(s_o_best_cells)))
    # s_o_best_cells, max_length_o, max_amount_o = get_set_of_best_cells(s_used_cells, s_x_cells)
    for i_o_cell, o_cell in enumerate(s_o_best_cells, 0):
        # s_used_cells_new = s_used_cells|set([o_cell])
        # s_o_cells_new = s_o_cells|set([o_cell])
        s_used_cells.add(o_cell)
        s_o_cells.add(o_cell)
        s_x_best_cells, max_length_x, max_amount_x = get_set_of_best_cells(s_used_cells, s_x_cells)
        # print("- i_o_cell: {}, o_cell: {}: len(s_x_best_cells): {}".format(i_o_cell, o_cell, len(s_x_best_cells)))
        # s_x_best_cells, max_length_x, max_amount_x = get_set_of_best_cells(s_used_cells_new, s_x_cells)
        # s_x_free_cells = get_free_cells(s_used_cells_new)
        for x_cell in s_x_best_cells:
            # s_used_cells_new_2 = s_used_cells_new|set([x_cell])
            # s_x_cells_new = s_x_cells|set([x_cell])
            s_used_cells.add(x_cell)
            s_x_cells.add(x_cell)

            # print("o_cell: {}, x_cell: {}".format(o_cell, x_cell))

            find_best_next_move_for_x_and_o(
                l_stats,
                s_used_cells,
                s_x_cells,
                s_o_cells,
                # s_used_cells_new_2,
                # s_x_cells_new,
                # s_o_cells_new,
                max_depth=max_depth-1,
                l_used_o_x_cells=l_used_o_x_cells+[(o_cell, x_cell)],
                l_length_amount=l_length_amount+[(max_length_o, max_amount_o), (max_length_x, max_amount_x)]
            )

            s_used_cells.remove(x_cell)
            s_x_cells.remove(x_cell)
        s_used_cells.remove(o_cell)
        s_o_cells.remove(o_cell)


if __name__=='__main__':

    # PATH_DIR_OBJECTS = PATH_ROOT_DIR+'objects/langtons_ant_multiple_machines/'
    # if not os.path.exists(PATH_DIR_OBJECTS):
    #     os.makedirs(PATH_DIR_OBJECTS)

    argv = sys.argv
    str_moves = argv[1]
    max_iterations = int(argv[2])
    
    print("str_moves: {}, max_iterations: {}".format(str_moves, max_iterations))

    PATH_DIR_PICTURES = '/tmp/ram/'+'pictures/langtons_ant_multiple_machines_iters_{}/'.format(max_iterations)
    # PATH_DIR_PICTURES = PATH_ROOT_DIR+'pictures/langtons_ant_multiple_machines_iters_{}/'.format(max_iterations)
    if not os.path.exists(PATH_DIR_PICTURES):
        os.makedirs(PATH_DIR_PICTURES)

    # (y, x)
    d_field = {(0, 0): 0}
    d_field_count = {(0, 0): 1}
    l_moves = []
    l_directions = []
    l_states = []
    l_tpls = []
    cell_now = (0, 0)
    direction = 'U'

    """
    directions:
    0,UP,NORTH,N
    1,RIGHT,EAST,E
    2,DOWN,SOUTH,S
    3,LEFT,WEST,W
    """

    # (direction, state) -> (delta y, delta x, new direction, new_state), where the delta values are:
    # (1, 0), (-1, 0), (0, 1), (0, -1)
    # 4 neighborhood
    # d_next_state = {0: 1, 1: 0}
    POSSIBLE_DIRECTIONS = ['U', 'R', 'D', 'L']
    d_dir_turn_next_pos_dir_move = {
        ('U', 'R'): (0, 1, 'R', 'R'), ('R', 'R'): (-1, 0, 'D', 'R'), ('D', 'R'): (0, -1, 'L', 'R'), ('L', 'R'): (1, 0, 'U', 'R'),
        ('U', 'L'): (0, -1, 'L', 'L'), ('R', 'L'): (1, 0, 'U', 'L'), ('D', 'L'): (0, 1, 'R', 'L'), ('L', 'L'): (-1, 0, 'D', 'L'),
    }
    
    # str_moves = 'LLLRRRLRLRLRLLR'
    # max_iterations = 2000000
    
    l_moves = list(str_moves)
    # l_moves = list('LRRRRRLLR')

    l_states_nums = np.arange(0, len(l_moves))
    l_states_nums_next = np.roll(l_states_nums, -1)
    d_state_next_state_turn = {state_now: (state_next, move) for move, state_now, state_next in zip(l_moves, l_states_nums, l_states_nums_next)}
    
    d_next_position_dir_move_state = {}
    for state, (next_state, next_turn) in d_state_next_state_turn.items():
        for direction in POSSIBLE_DIRECTIONS:
            d_next_position_dir_move_state[(direction, state)] = d_dir_turn_next_pos_dir_move[(direction, next_turn)]+(next_state, )
    # d_next_position_dir_state_2 = {
    #     (0, 0): (0, 1, 1, 1), (1, 0): (-1, 0, 2, 1), (2, 0): (0, -1, 3, 1), (3, 0): (1, 0, 0, 1),
    #     (0, 1): (0, -1, 3, 0), (1, 1): (1, 0, 0, 0), (2, 1): (0, 1, 1, 0), (3, 1): (-1, 0, 2, 0),
    # }
    # assert d_next_position_dir_state_2==d_next_position_dir_move_state

    for i_step in range(1, max_iterations+1):
        state_now = d_field[cell_now]
        dy, dx, new_dir, move, new_state = d_next_position_dir_move_state[(direction, state_now)]
        cell_new = (cell_now[0]+dy, cell_now[1]+dx)
        
        y, x = cell_new
        if y<-300 or y>300 or x<-300 or x>300:
            break

        # print("cell_new: {}".format(cell_new))
        d_field[cell_now] = new_state
        # d_field[cell_now] = d_next_state[state_now]
        if not cell_new in d_field:
            d_field[cell_new] = 0
            d_field_count[cell_new] = 0

        l_moves.append(move)
        l_directions.append(new_dir)
        l_states.append(new_state)
        l_tpls.append((move, new_dir, new_state))

        direction = new_dir
        cell_now = cell_new
        d_field_count[cell_new] += 1

    # print("d_field: {}".format(d_field))


    # find the needed size for the picture!
    min_y = 0
    min_x = 0
    max_y = 0
    max_x = 0

    for y, x in d_field.keys():
        if y<min_y:
            min_y = y
        if x<min_x:
            min_x = x
        if y>max_y:
            max_y = y
        if x>max_x:
            max_x = x

    # print("min_y: {}".format(min_y))
    # print("min_x: {}".format(min_x))
    # print("max_y: {}".format(max_y))
    # print("max_x: {}".format(max_x))

    # convert s_x_cells and s_o_cells to list first!

    size_w = max_x-min_x+1
    size_h = max_y-min_y+1

    pix = np.zeros((size_h, size_w, 3), dtype=np.uint8)
    pix[:] = (128, 128, 128)

    arr_colors = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [255, 255, 255],

        [128, 128, 128],
        [128, 128, 255],
        [128, 255, 128],
        [255, 128, 128],
        [128, 255, 255],
        [255, 128, 255],
        [255, 255, 128],
        [255, 255, 255],

        [64, 64, 64],
        [64, 64, 255],
        [64, 255, 64],
        [255, 64, 64],
        [64, 255, 255],
        [255, 64, 255],
        [255, 255, 64],
        [255, 255, 255],

        [64, 64, 64],
        [64, 64, 128],
        [64, 128, 64],
        [128, 64, 64],
        [64, 128, 128],
        [128, 64, 128],
        [128, 128, 64],
        [128, 128, 128],
    ], dtype=np.uint8)

    # for y, x in s_x_cells:
    #     pix[y-min_y, x-min_x] = (255, 128, 0)
    for (y, x), state in d_field.items():
        pix[y-min_y, x-min_x] = arr_colors[state]

    img = Image.fromarray(pix)
    img.save(PATH_DIR_PICTURES+'langtons_ant_1x1_pixels_states_{}_iterations_{}.png'.format(str_moves, max_iterations)) 
    

    pix_count = np.zeros((size_h, size_w, 3), dtype=np.uint8)
    pix_count[:] = (64, 128, 192)

    max_count = 0
    for count in d_field_count.values():
        if max_count<count:
            max_count = count
    # print("max_count: {}".format(max_count))

    arr_colors = np.zeros((max_count+1, 3), dtype=np.uint8)
    for i in range(0, max_count+1):
        arr_colors[i] = int(i*255/max_count)

    for (y, x), count in d_field_count.items():
        pix_count[y-min_y, x-min_x] = arr_colors[count]

    img_count = Image.fromarray(pix_count)
    img_count.save(PATH_DIR_PICTURES+'langtons_ant_1x1_pixels_count_{}_iterations_{}.png'.format(str_moves, max_iterations)) 

    sys.exit(0)

    max_len_repeat = len(l_directions)//2
    len_repeat = 1
    is_found_repeat = False
    while len_repeat<max_len_repeat:
        l_dir_repeat = l_tpls[-len_repeat:]
        if l_dir_repeat==l_tpls[-2*len_repeat:-len_repeat]:
            is_found_repeat = True
            break
        len_repeat += 1

    if is_found_repeat:
        print("l_dir_repeat: {}".format(l_dir_repeat))
        print("len_repeat: {}".format(len_repeat))
    else:
        print("No repeat found!")

    # print("l_len_repeat: {}".format(l_len_repeat))
