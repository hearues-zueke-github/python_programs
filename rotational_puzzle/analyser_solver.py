#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import math
import mmap
import os
import re
import sys
import time
import traceback

import itertools
import multiprocessing

from multiprocessing import Process, Pipe
# from multiprocessing import shared_memory # in python3.8 available!

from dotmap import DotMap

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

from copy import copy, deepcopy

import utils

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

def get_rot_functions():
    # times 2 is the part for the rot field of the size 3x3
    dys1 = np.array([0]*2+[1]*2)
    dxs1 = np.array([1]*2+[0]*2)

    dys = np.hstack((dys1, -dys1))
    dxs = np.hstack((dxs1, -dxs1))

    ys = np.cumsum(dys)
    xs = np.cumsum(dxs)

    ys_rot_right = [np.roll(ys, i) for i in range(1, 5)]
    xs_rot_right = [np.roll(xs, i) for i in range(1, 5)]
    ys_rot_left = [np.roll(ys, -i) for i in range(1, 5)]
    xs_rot_left = [np.roll(xs, -i) for i in range(1, 5)]

    ys_cc = np.roll(ys, -1)
    xs_cc = np.roll(xs, -1)

    def rot_field_right(f, x, y, r):
        f[ys+y-1, xs+x-1] = f[ys_rot_right[r-1]+y-1, xs_rot_right[r-1]+x-1]
    def rot_field_left(f, x, y, r):
        f[ys+y-1, xs+x-1] = f[ys_rot_left[r-1]+y-1, xs_rot_left[r-1]+x-1]

    def f_rot(f, x, y, r, c):
        if c:
            f[ys+y-1, xs+x-1] = f[ys_rot_right[r-1]+y-1, xs_rot_right[r-1]+x-1]
        else:
            f[ys+y-1, xs+x-1] = f[ys_rot_left[r-1]+y-1, xs_rot_left[r-1]+x-1]

    return ys, xs, rot_field_right, rot_field_left, f_rot


def get_do_moves():
    _, _, _, _, f_rot = get_rot_functions()
    def do_moves(field, moves):
        for move in moves:
            f_rot(*((field, )+move))
    return do_moves

do_moves = get_do_moves()

def get_moving_dict_field(w, h):
    ys, xs, _, _, _ = get_rot_functions()

    pos_rotation_centers = set([(x, y) for y in range(1, h-1) for x in range(1, w-1)])
    print("pos_rotation_centers:\n{}".format(pos_rotation_centers))
    pos_cell_centers = set([(x, y) for y in range(0, h) for x in range(0, w)])
    print("pos_cell_centers: {}".format(pos_cell_centers))

    rotation_cell_dict = {}
    for rot_center in pos_rotation_centers:
        x, y = rot_center
        rotation_cell_dict[(x, y)] = [
            {p: pl for p, pl in zip(list(zip(xs+x-1, ys+y-1)), list(zip(np.roll(xs+x-1, 1), np.roll(ys+y-1, 1))))},
            {p: pr for p, pr in zip(list(zip(xs+x-1, ys+y-1)), list(zip(np.roll(xs+x-1, -1), np.roll(ys+y-1, -1))))}
        ]

    possible_rotation_centers = {}
    for cell_center in np.array(list(pos_cell_centers))[np.random.permutation(np.arange(0, len(pos_cell_centers)))]:
        x, y = cell_center
        pos_rot_centers = set(list(zip(xs+x-1, ys+y-1)))
        possible_rotation_centers[(x, y)] = list(pos_rotation_centers & pos_rot_centers)

    cells_4_neighbors = {}
    for cell_center in np.array(list(pos_cell_centers))[np.random.permutation(np.arange(0, len(pos_cell_centers)))]:
        x, y = cell_center
        neighbors_4 = []
        if x > 0:
            neighbors_4.append((x-1, y))
        if x < w-1:
            neighbors_4.append((x+1, y))
        if y > 0:
            neighbors_4.append((x, y-1))
        if y < h-1:
            neighbors_4.append((x, y+1))

        cells_4_neighbors[(x, y)] = neighbors_4

    def build_best_moves(x, y, moving_dict):
        moving_dict = deepcopy(moving_dict)

        centers_cell = [(x, y)]
        while len(centers_cell) > 0:
            next_centers_cell = []
            possible_moves = {}
            for center_cell in centers_cell:
                rots_centers = possible_rotation_centers[center_cell]
                neighs_4 = cells_4_neighbors[center_cell]
                # print("center_cell: {}".format(center_cell))
                # print("rots_centers: {}".format(rots_centers))
                # print("neighs_4: {}".format(neighs_4))
                neighs_4_rest = deepcopy(neighs_4)
                for cell in neighs_4:
                    # print("cell: {}".format(cell))
                    if cell in moving_dict:
                        neighs_4_rest.pop(neighs_4_rest.index(cell))
                # print("neighs_4_rest: {}".format(neighs_4_rest))

                def check_if_all_cells_moveable(rot_moves):
                    for center_cell in rot_moves.keys():
                        if center_cell in moving_dict and moving_dict[center_cell] == None:
                            return False
                    return True

                for rot_center in rots_centers:
                    rot_moves = rotation_cell_dict[rot_center]
                    for neigh_cell in neighs_4_rest:
                        if rot_moves[0][center_cell] == neigh_cell and not neigh_cell in moving_dict and check_if_all_cells_moveable(rot_moves[0]):
                            moving_dict[neigh_cell] = []
                            moving_dict[center_cell].append((neigh_cell, rot_center+(1, False)))
                            
                            possible_moves[center_cell] = moving_dict[center_cell]
                        elif rot_moves[1][center_cell] == neigh_cell and not neigh_cell in moving_dict and check_if_all_cells_moveable(rot_moves[0]):
                            moving_dict[neigh_cell] = []
                            moving_dict[center_cell].append((neigh_cell, rot_center+(1, True)))
                            
                            possible_moves[center_cell] = moving_dict[center_cell]
                next_centers_cell.extend(neighs_4_rest)
            # print("possible_moves:\n{}".format(possible_moves))
            # print("moving_dict: {}".format(moving_dict))
            # print("next_centers_cell:\n{}".format(next_centers_cell))

            centers_cell = next_centers_cell

        return moving_dict


    def parse_moving_dict(x, y, moving_dict):
        abs_moving_dict = {}
        centers_cell = [(x, y)]
        while len(centers_cell) > 0:
            next_centers_cell = []
            for center_cell in centers_cell:
                lst_moves = moving_dict[center_cell]
                # print("lst_moves: {}".format(lst_moves))
                for neigh_cell, move in lst_moves:
                    next_move = move[:3]+(move[3]==False, )
                    next_centers_cell.append(neigh_cell)

                    if not center_cell in abs_moving_dict:
                        abs_moving_dict[neigh_cell] = [next_move]
                    else:
                        prev_moves = deepcopy(abs_moving_dict[center_cell])
                        # check if the last move is at the same position and same rotation!
                        # add the rotations up!
                        if prev_moves[0][:2] == next_move[:2]:
                            t = prev_moves[0]
                            prev_moves[0] = (t[0], t[1], t[2]+1, t[3])
                            abs_moving_dict[neigh_cell] = prev_moves
                        else:
                            abs_moving_dict[neigh_cell] = [next_move]+abs_moving_dict[center_cell]
            # print("next_centers_cell: {}".format(next_centers_cell))
            centers_cell = next_centers_cell
        # print("abs_moving_dict: {}".format(abs_moving_dict))
        return abs_moving_dict


    def get_absolute_moves_per_center_cell(x, y, moving_dict):
        moving_dict = build_best_moves(x, y, moving_dict)
        absolute_moving_dict = parse_moving_dict(x, y, moving_dict)

        return absolute_moving_dict


    def do_create_moving_dict_field(w, h):
        moving_dict_field = {}
        
        for start_y in range(h-1, 2, -1):
            for start_x in range(w-1, 1, -1):
                print("start_x: {}".format(start_x))
                moving_dict_start = {(x, start_y): None for x in range(start_x, w)}
                moving_dict_start_below = {(x, y): None for y in range(start_y+1, h) for x in range(0, w)}
                moving_dict_start = {**moving_dict_start, **moving_dict_start_below}
                moving_dict_start[(start_x, start_y)] = []
                absolute_moving_dict = get_absolute_moves_per_center_cell(start_x, start_y, moving_dict_start)

                moving_dict_field[start_y*w+start_x] = absolute_moving_dict

            moving_dict_start = {(x, y): None for y in range(start_y, h) for x in range(0, w)}
            moving_dict_start[(0, start_y-3)] = []
            absolute_moving_dict_0 = get_absolute_moves_per_center_cell(0, start_y-3, moving_dict_start)

            moves_for_0_0_y = [
                (1, start_y-1, 1, True),
                (1, start_y-2, 2, True),
                (1, start_y-1, 1, True),
                (1, start_y-2, 2, False),
                (1, start_y-1, 2, False),
            ]
            moves_for_1_0_y = [
                (1, start_y-1, 2, True),
                (1, start_y-2, 2, False),
                (1, start_y-1, 2, False),
            ]

            absolute_moving_dict = deepcopy(absolute_moving_dict_0)
            for k, v in list(absolute_moving_dict.items()):
                absolute_moving_dict[k] = v+moves_for_1_0_y
            absolute_moving_dict[(0, start_y-3)] = moves_for_1_0_y
            absolute_moving_dict[(0, start_y)] = moves_for_0_0_y

            moving_dict_field[start_y*w+1] = absolute_moving_dict

            moves_for_0_y = [
                (1, start_y-1, 1, True),
                (1, start_y-2, 2, False),
                (1, start_y-1, 1, False),
            ]

            absolute_moving_dict = deepcopy(absolute_moving_dict_0)
            for k, v in list(absolute_moving_dict.items()):
                absolute_moving_dict[k] = v+moves_for_0_y
            absolute_moving_dict[(0, start_y-3)] = moves_for_0_y

            moving_dict_field[start_y*w+0] = absolute_moving_dict


        for start_x in range(w-1, 2, -1):
            print("For first 3 rows: start_x: {}".format(start_x))
            start_y = 2
            moving_dict_start = {(start_x, 2): []}
            moving_dict_start_right = {(x, y): None for x in range(start_x+1, w) for y in range(0, 3)}
            moving_dict_start_below = {(x, y): None for y in range(3, h) for x in range(0, w)}
            moving_dict_start = {**moving_dict_start, **moving_dict_start_right, **moving_dict_start_below}
            # moving_dict_start[(start_x, start_y)] = []
            # moving_dict_start_orig = deepcopy(moving_dict_start)
            # print("doing start_x: {}, start_y: {}".format(start_x, start_y))
            absolute_moving_dict = get_absolute_moves_per_center_cell(start_x, start_y, moving_dict_start)

            moving_dict_field[2*w+start_x] = absolute_moving_dict

            # moving_dict_start_below = {(x, y): [] for y in range(start_y, h) for x in range(0, w)}
            # moving_dict_start = deepcopy(moving_dict_start_orig)
            moving_dict_start[(start_x, 1)] = None
            moving_dict_start[(start_x, 0)] = None
            moving_dict_start[(start_x-3, 0)] = []
            # print("doing start_x: {}, start_y: {}".format(start_x, start_y-1))
            absolute_moving_dict_0 = get_absolute_moves_per_center_cell(start_x-3, 0, moving_dict_start)

            moves_for_0_0_x = [
                (start_x-1, 1, 1, False),
                (start_x-2, 1, 2, False),
                (start_x-1, 1, 1, False),
                (start_x-2, 1, 2, True),
                (start_x-1, 1, 2, True),
            ]
            moves_for_0_1_x = [
                (start_x-1, 1, 2, False),
                (start_x-2, 1, 2, True),
                (start_x-1, 1, 2, True),
            ]

            absolute_moving_dict = deepcopy(absolute_moving_dict_0)
            for k, v in list(absolute_moving_dict.items()):
                absolute_moving_dict[k] = v+moves_for_0_1_x
            absolute_moving_dict[(start_x-3, 0)] = moves_for_0_1_x
            absolute_moving_dict[(start_x, 0)] = moves_for_0_0_x

            if start_x == 3: # special case for the position (1, 1)
                absolute_moving_dict[(1, 1)] = [
                    (2, 1, 1, False),
                    (1, 1, 1, True),
                    (2, 1, 1, False),
                    (1, 1, 4, True),
                    (2, 1, 2, True),
                ]

            moving_dict_field[1*w+start_x] = absolute_moving_dict

            moves_for_0_x = [
                (start_x-1, 1, 1, False),
                (start_x-2, 1, 2, True),
                (start_x-1, 1, 1, True),
            ]

            absolute_moving_dict = deepcopy(absolute_moving_dict_0)
            for k, v in list(absolute_moving_dict.items()):
                absolute_moving_dict[k] = v+moves_for_0_x
            absolute_moving_dict[(start_x-3, 0)] = moves_for_0_x

            if start_x == 3: # special case for the position (1, 1)
                absolute_moving_dict[(1, 1)] = [
                    (2, 1, 1, False),
                    (1, 1, 3, False),
                    (2, 1, 1, True),
                ]

            moving_dict_field[0*w+start_x] = absolute_moving_dict

        return moving_dict_field


    return do_create_moving_dict_field(w, h)


def get_random_moves(n):
    arr_xs = np.random.randint(1, field_size[1]-1, (n, )).tolist()
    arr_ys = np.random.randint(1, field_size[0]-1, (n, )).tolist()
    arr_rs = np.random.randint(1, 5, (n, )).tolist()
    arr_rot_side = (np.random.randint(0, 2, (n, ))==0).tolist()

    moves = list(zip(arr_xs, arr_ys, arr_rs, arr_rot_side))
    return moves


def get_inverted_moves(moves):
    moves_inverted = []
    for move in moves[::-1]:
        moves_inverted.append((move[0], move[1], move[2], (move[3]==False)+0))
    return moves_inverted


def get_inverted_rotations_moves(moves):
    moves_inv_rot = []
    for move in moves:
        moves_inv_rot.append((move[0], move[1], move[2], (move[3]==False)+0))
    return moves_inv_rot


def get_transpose_moves(moves):
    moves_transpose = []
    for move in moves:
        moves_transpose.append((move[1], move[0], move[2], (move[3]==False)+0))
    return moves_transpose

def convert_moves_to_str(moves):
    s = ""

    if len(moves) > 0:
        m = moves[0]
        s += '{}{}{}{}'.format(m[0], m[1], m[2], 1 if m[3] else 0)
        for m in moves[1:]:
            s += ',{}{}{}{}'.format(m[0], m[1], m[2], 1 if m[3] else 0)

    return s


def test_rotation_of_fields():
    field_size = (5, 6) # (y, x)
    max_num = np.multiply.reduce(field_size)
    num_field = np.arange(0, max_num).reshape((field_size))
    print("num_field:\n{}".format(num_field))

    ys, xs, rot_r, rot_l, f_rot = get_rot_functions()

    n  = 10
    moves = get_random_moves(n)
    moves_inv = get_inverted_moves(moves)

    print("len(moves):\n{}".format(len(moves)))
    print("len(moves_inv):\n{}".format(len(moves_inv)))
    
    for move in moves:
        f_rot(*((num_field, )+move))
    print("after doing moves on num_field:\n{}".format(num_field))

    for move in moves_inv:
        f_rot(*((num_field, )+move))
    print("after doing moves_inv on num_field:\n{}".format(num_field))


def test_rotation_of_fields_nr_2():
    w, h = 6, 6
    field_size = (h, w) # (y, x)
    max_num = np.multiply.reduce(field_size)
    num_field = np.arange(0, max_num).reshape((field_size))
    print("num_field:\n{}".format(num_field))
    _, _, _, _, f_rot = get_rot_functions()

    moving_dict_field = get_moving_dict_field(w, h)

    print("num_field:\n{}".format(num_field))

    absolute_moving_dict = moving_dict_field[1*w+3]
    for move in absolute_moving_dict[(0, 0)]:
        f_rot(*((num_field, )+move))

    print("after doing moves for (0, 0): num_field:\n{}".format(num_field))


def create_moving_table_json_file(file_path, w, h):
    f = open(file_path, "w")

    moving_dict_field = get_moving_dict_field(w, h)

    def write_moves(moves):
        for move in moves[:-1]:
            f.write('"{move}",'.format(move=move))
            # f.write('        "{move}",\n'.format(move=move))
        f.write('"{move}"'.format(move=moves[-1]))
        # f.write('        "{move}"\n'.format(move=moves[-1]))

    def write_centers_cell(centers_cell):
        centers_cell_keys = list(centers_cell.keys())
        for center_cell in centers_cell_keys[:-1]:
            f.write('"{center_cell}":['.format(center_cell=center_cell))
            # f.write('      "{center_cell}": [\n'.format(center_cell=center_cell))
            write_moves(centers_cell[center_cell])
            f.write('],')
            # f.write('      ],\n')
        center_cell = centers_cell_keys[-1]
        f.write('"{center_cell}":['.format(center_cell=center_cell))
        # f.write('      "{center_cell}": [\n'.format(center_cell=center_cell))
        write_moves(centers_cell[center_cell])
        f.write(']')
        # f.write('      ]\n')

    def write_moving_dict_field(moving_dict_field):
        moving_dict_field_keys = list(moving_dict_field.keys())
        for cell_num in moving_dict_field_keys[:-1]:
            f.write('"{cell_num}":{{'.format(cell_num=cell_num))
            # f.write('    "{cell_num}": {{\n'.format(cell_num=cell_num))
            write_centers_cell(moving_dict_field[cell_num])
            f.write('},')
            # f.write('    },\n')
        cell_num = moving_dict_field_keys[-1]
        f.write('"{cell_num}":{{'.format(cell_num=cell_num))
        # f.write('    "{cell_num}": {{\n'.format(cell_num=cell_num))
        write_centers_cell(moving_dict_field[cell_num])
        f.write('}')
        # f.write('  }\n')

    f.write('{')
    # f.write('{\n')
    f.write('"({w},{h})":{{'.format(w=w, h=h))
    # f.write('  "({w}, {h})": {{\n'.format(w=w, h=h))
    
    write_moving_dict_field(moving_dict_field)
    
    f.write('}')
    # f.write('}\n')

    f.close()


def convert_moves_str_to_moves(moves_str):
    moves_1 = [moves_str[4*i:4*(i+1)] for i in range(0, len(moves_str)//4)]
    return [(int(m[0]), int(m[1]), int(m[2]), int(m[3])==1) for m in moves_1]


def get_combine_moves(moves1, moves2): # for a 3x3 rotation field!
    moves = moves1+moves2
    if len(moves) == 0:
        return []
    move_prev = moves[0]

    combined_moves = []

    for move in moves[1:]:
        if move_prev[0]==move[0] and move_prev[1]==move[1]:
            rot1 = move_prev[2]
            rot2 = move[2]
            clock1 = move_prev[3]
            clock2 = move[3]
            if clock1 != clock2:
                if rot1 > rot2:
                    rot = rot1-rot2
                    clock = clock1
                elif rot1 < rot2:
                    rot = rot2-rot1
                    clock = clock2
                else:
                    rot = 0
                    clock = False
            else:
                rot = rot1+rot2
                clock = clock1
                if rot > 4:
                    rot = (-rot) % 4
                    clock = not clock
                if rot == 4:
                    clock = False

            combine_move = (move[0], move[1], rot, clock)
            move_prev = combine_move
        else:
            if move_prev[2] > 0:
                combined_moves.append(move_prev)
            move_prev = move

    if move_prev[2] > 0:
        combined_moves.append(move_prev)

    return combined_moves


def find_solutions_for_3x3_block():
    _, _, _, _, f_rot = get_rot_functions()

    file_name = 'all_moves_3x3_field_str.pkl'
    if os.path.exists(file_name) and False:
        with open('all_moves_3x3_field_str.pkl', 'rb') as f:
            all_moves_3x3_field_str = dill.load(f)

        print("len(all_moves_3x3_field_str): {}".format(len(all_moves_3x3_field_str)))
        print("all_moves_3x3_field_str[10]: {}".format(all_moves_3x3_field_str[1]))

        arr = np.array(all_moves_3x3_field_str)
        arr = np.array([x.replace(",", "") for x in arr])
        lens = np.array(list(map(lambda x: (len(x)-9)//4, all_moves_3x3_field_str)))
        arr_max = arr[lens==np.max(lens)]
        print("np.max(lens): {}".format(np.max(lens)))
        print("arr_max: {}".format(arr_max))

        arr = arr[np.argsort(lens)]
        globals()['arr'] = arr

        new_moves_3x3_field = []
        count_same_length = 0
        count_shorter_length = 0
        sum_diff_length = 0
        diff_lengths = []
        for index, row in enumerate(arr, 0):
            print("index: {}".format(index))
            row = row.replace(",", "")
            field_3x3, moves_str = row[:9], row[9:]
            moves_old = convert_moves_str_to_moves(moves_str)
            
            orig_len = len(moves_old)
            combined_moves = get_combine_moves([], moves_old)
            if len(combined_moves) != len(moves_old):
                combined_moves_2 = get_combine_moves([], combined_moves)
                while len(combined_moves_2) != len(combined_moves):
                    combined_moves = combined_moves_2
                    combined_moves_2 = get_combine_moves([], combined_moves)
            new_len = len(combined_moves)

            diff_lengths.append(orig_len-new_len)

            moves_str_new = convert_moves_to_str(combined_moves).replace(",", "")
            new_moves_3x3_field.append(field_3x3+moves_str_new)

            if len(moves_old) == len(combined_moves):
                count_same_length += 1
            else:
                count_shorter_length += 1
                sum_diff_length += len(moves_old)-len(combined_moves)

        print("count_same_length: {}".format(count_same_length))
        print("count_shorter_length: {}".format(count_shorter_length))
        print("sum_diff_length: {}".format(sum_diff_length))
        globals()['new_moves_3x3_field'] = new_moves_3x3_field

        arr = np.array(new_moves_3x3_field)

        lens = np.array(list(map(len, arr))).astype(np.uint64)
        ints = np.array([int(x[:9]) for x in arr]).astype(np.uint64)
        
        arr_2 = np.vstack((lens, ints)).T.reshape((-1, )).view("u8,u8")

        arr = arr[np.argsort(arr_2)]

        f = open("all_moves_3x3_field_text_3.txt", "w")
        for moves_str in arr:
            f.write(moves_str+"\n")
        f.close()

        return

        # do the check, if the moves are correct or not!
        field_size = (4, 4) # (y, x)
        max_num = np.multiply.reduce(field_size)
        num_field = np.arange(0, max_num).reshape((field_size))

        num_field[:] = -1
        num_field[0, :3] = np.arange(0, 3)
        num_field[1, :3] = np.arange(3, 6)
        num_field[2, :3] = np.arange(6, 9)

        print("num_field:\n{}".format(num_field))

        sorted_numbers = np.arange(0, 9)

        for index, row in enumerate(arr, 0):
            if index % 10000 == 0:
                print("index: {}".format(index))
            field_3x3, moves_str = row[:9], row[9:]
            field_3x3_int = list(map(int, list(field_3x3)))
            moves = convert_moves_str_to_moves(moves_str)

            num_field_ = num_field.copy()
            num_field_[0, :3] = field_3x3_int[0:3]
            num_field_[1, :3] = field_3x3_int[3:6]
            num_field_[2, :3] = field_3x3_int[6:9]

            do_moves(num_field_, moves)
            moved_numbers = num_field_[:3, :3].reshape((-1, )).copy()


            combined_moves = get_combine_moves([], moves)
            if len(combined_moves) != len(moves_old):
                combined_moves_2 = get_combine_moves([], combined_moves)
                while len(combined_moves_2) != len(combined_moves):
                    combined_moves = combined_moves_2
                    combined_moves_2 = get_combine_moves([], combined_moves)

            num_field_ = num_field.copy()
            num_field_[0, :3] = field_3x3_int[0:3]
            num_field_[1, :3] = field_3x3_int[3:6]
            num_field_[2, :3] = field_3x3_int[6:9]

            do_moves(num_field_, combined_moves)
            moved_numbers_2 = num_field_[:3, :3].reshape((-1, )).copy()

            try:
                assert np.sum(sorted_numbers != moved_numbers) == 0
                assert np.sum(sorted_numbers != moved_numbers_2) == 0
            except:
                print("moves:\n{}".format(moves))
                print("combined_moves:\n{}".format(combined_moves))
                return

        return

    field_size = (4, 4) # (y, x)
    max_num = np.multiply.reduce(field_size)
    num_field = np.arange(0, max_num).reshape((field_size))
    print("num_field:\n{}".format(num_field))

    neutral_moves_part_1 = [
        [(1,1,1,1)], [(1,1,2,1)], [(1,1,3,1)], [(1,1,4,1)],
        [(1,1,1,0)], [(1,1,2,0)], [(1,1,3,0)],
    ]

    import utils_03 as utils
    neutral_moves_read_part = utils.neutral_moves

    n_moves_tm = []
    # n_moves_tm = [get_transpose_moves(moves) for moves in neutral_moves_read_part]

    neutral_moves = neutral_moves_part_1+neutral_moves_read_part+n_moves_tm

    neutral_moves = list(filter(lambda x: x[0][:2]!=(1, 1) and x[-1][:2]!=(1, 1), neutral_moves))

    print("len(neutral_moves): {}".format(len(neutral_moves)))

    # print("neutral_moves_part_4: {}".format(neutral_moves_part_4))

    neutral_moves_str = [convert_moves_to_str(moves).replace(",", "") for moves in neutral_moves]

    print("neutral_moves_str: {}".format(neutral_moves_str))

    print("len(neutral_moves_str): {}".format(len(neutral_moves_str)))
    print("len(set(neutral_moves_str)): {}".format(len(set(neutral_moves_str))))

    globals()['neutral_moves_str'] = neutral_moves_str


    num_field[:] = -1
    num_field[0, :3] = np.arange(0, 3)
    num_field[1, :3] = np.arange(3, 6)
    num_field[2, :3] = np.arange(6, 9)
    
    def get_3x3_str(field):
        return "".join(map(str, field[:3, :3].reshape((-1, ))))

    # TODO: remove duplicated moves from neutral_moves!
    # first, generate the moving field results!
    idxs = np.arange(0, len(neutral_moves))
    moving_field_results = []
    for i, moves in enumerate(neutral_moves, 0):
        num_field_ = num_field.copy()
        do_moves(num_field_, moves)
        num_str = get_3x3_str(num_field_)
        # print("i: {}, num_str: {}".format(i, num_str))
        assert len(num_str) == 9
        moving_field_results.append((int(num_str), len(moves), i))
    globals()['moving_field_results'] = moving_field_results

    sort_lst = sorted(moving_field_results, key=lambda x: (x[0], x[1], x[2]))

    idxs_sort, lens_sort, true_idxs_sort = list(zip(*sort_lst))

    idxs_sort = np.array(idxs_sort)
    lens_sort = np.array(lens_sort)
    true_idxs_sort = np.array(true_idxs_sort)
    do_not_remove_idxs = np.where(idxs_sort[1:]!=idxs_sort[:-1])[0]+1
    found_idxs_orig = np.where(idxs_sort[1:]==idxs_sort[:-1])[0]+1

    found_idxs = np.sort(np.unique(np.vstack((found_idxs_orig, found_idxs_orig-1)).reshape((-1, ))))

    idxs_sort = idxs_sort[found_idxs]
    lens_sort = lens_sort[found_idxs]

    globals()['do_not_remove_idxs'] = do_not_remove_idxs
    globals()['found_idxs_orig'] = found_idxs_orig
    globals()['found_idxs'] = found_idxs
    globals()['true_idxs_sort'] = true_idxs_sort
    globals()['neutral_moves'] = neutral_moves

    idxs_to_remove = np.sort(true_idxs_sort[found_idxs_orig])[::-1]

    for idx in idxs_to_remove:
        neutral_moves.pop(idx)
    print("new shorter len(neutral_moves): {}".format(len(neutral_moves)))

    # sys.exit(-5)

    prev_moves_3x3_field = {"".join(map(str, num_field[:3, :3].reshape((-1, )))): ""}
    # prev_moves_3x3_field = {"".join(map(str, num_field[:3, :3].reshape((-1, )))): []}
    # prev_moves_3x3_field = {tuple(num_field[:3, :3].reshape((-1, ))): []}
    all_moves_3x3_field = deepcopy(prev_moves_3x3_field)

    # print("prev_moves_3x3_field: {}".format(prev_moves_3x3_field))
    # sys.exit(-1)

    max_length = math.factorial(9)
    for i in range(0, 1):
        new_moves_3x3_field = {}
        for item_number, (field_3x3, prev_moves) in enumerate(prev_moves_3x3_field.items()):
            print("i: {}, item_number: {}, len(all_moves_3x3_field): {}".format(i, item_number, len(all_moves_3x3_field)))
            num_field_ = num_field.copy()
            num_field_[:3, :3] = np.array(list(field_3x3)).reshape((3, 3))
            for next_moves in neutral_moves:
                num_field__ = num_field_.copy()

                do_moves(num_field__, next_moves)
                new_field_3x3 = "".join(map(str, num_field__[:3, :3].reshape((-1, ))))
                assert len(new_field_3x3) == 9
                # new_field_3x3 = tuple(num_field__[:3, :3].reshape((-1, )))
                moves_str = convert_moves_to_str(get_inverted_moves(next_moves)).replace(",", "")+prev_moves
                if not new_field_3x3 in all_moves_3x3_field:
                    new_moves_3x3_field[new_field_3x3] = moves_str
                    all_moves_3x3_field[new_field_3x3] = moves_str
                elif len(moves_str) < len(all_moves_3x3_field[new_field_3x3]):
                    if new_field_3x3 in new_moves_3x3_field:
                        new_moves_3x3_field[new_field_3x3] = moves_str
                    all_moves_3x3_field[new_field_3x3] = moves_str

        print("i: {}, len(new_moves_3x3_field): {}".format(i, len(new_moves_3x3_field)))

        prev_moves_3x3_field = new_moves_3x3_field
        # all_moves_3x3_field = {**all_moves_3x3_field, **new_moves_3x3_field}

        if len(all_moves_3x3_field) >= max_length:
            break

    print("")
    print("len(all_moves_3x3_field): {}".format(len(all_moves_3x3_field)))

    all_moves_3x3_field_str = [key+val for key, val in all_moves_3x3_field.items()]
    with open('all_moves_3x3_field_str.pkl', 'wb') as f:
        dill.dump(all_moves_3x3_field_str, f)

    globals()['all_moves_3x3_field'] = all_moves_3x3_field

    sum_lengths = np.sum([len(l) for l in all_moves_3x3_field.values()]) // 4
    print("sum_lengths: {}".format(sum_lengths))


def count_rest_neutral_rotations(moves):
    moves = deepcopy(moves)

    amount_clockwise_moves = {
        (0, 0): 0, (0, 1): 0,
        (1, 0): 7, (1, 1): 1,
        (2, 0): 6, (2, 1): 2,
        (3, 0): 5, (3, 1): 3,
        (4, 0): 4, (4, 1): 4,
    }

    # 1st convert all rotations to absolute clockwise rotations!
    new_moves = []
    for x, y, r, c in moves:
        new_moves.append((x, y, amount_clockwise_moves[(r, c)]))
    moves = new_moves

    # 2nd find all positions and rotations and add up all rotations % 8
    positions_rotations = {}
    for x, y, r in moves:
        t = (x, y)
        if not t in positions_rotations:
            positions_rotations[t] = r
        else:
            positions_rotations[t] = (positions_rotations[t] + r) % 8

    print("positions_rotations: {}".format(positions_rotations))

    return positions_rotations


def convert_str_base_moves_to_num_base_moves(base_moves_str):
    convert_str_to_num = {'a':'1','b':'2','c':'3'}
    return ["".join([convert_str_to_num[c] for c in list(l)]) for l in base_moves_str]


def find_a_lot_neutral_moves_for_3x3_block():
    field_size = (4, 4) # (y, x)
    max_num = np.multiply.reduce(field_size)
    num_field = np.arange(0, max_num).reshape((field_size))
    print("num_field:\n{}".format(num_field))

    # num_field[:] = 1
    num_field[:3, :3] = 0
    print("num_field:\n{}".format(num_field))

    d = DotMap()
    for v, x, y in [('a', 1, 1), ('b', 2, 1), ('c', 1, 2)]:
        d[v].l = [(x, y, i, 0) for i in range(1, 4)]
        d[v].r = [(x, y, i, 1) for i in range(1, 5)]

    print("d: {}".format(d))
    globals()['d'] = d

    combinations_moves = [('l', i) for i in range(0, 3)]+[('r', i) for i in range(0, 4)]
    print("combinations_moves: {}".format(combinations_moves))

    all_found_neutral_moves = DotMap()
    all_base_moves_not_neutral_moves = []

    def create_non_repeating_next_number_sequence(m, n):
        all_combinations = []

        def non_repeating_next_number(index, nums, last_num):
            if index >= n:
                if nums[0] != 0 and nums[-1] != 0: # for a faster search!
                    all_combinations.append(nums)
                return

            for i2 in range(1, m):
                next_num = (last_num+i2)%m
                non_repeating_next_number(index+1, nums+[next_num], next_num)

        for i in range(0, m):
            non_repeating_next_number(1, [i], i)

        return all_combinations

    # for max_n in range(2, 8):

    base_moves_names = ['a', 'b', 'c']
    all_num_combinations_lst = [create_non_repeating_next_number_sequence(len(base_moves_names), n) for n in range(5, 6)]

    all_num_combinations = [l for li in all_num_combinations_lst for l in li]

    # all_num_combinations = [
    #     # [2,0,1,0,2,0,1],
    #     # [1,0,1],
    #     # [1,0,1,0,1],
    #     # [1,0,1,0,1,0,1],
    #     [1,0,1,0,1,0,1,0,1],
    #     # [2,1,0,1,2,1,0,1],
    #     # [2,1,0,1,2,0,1,0,1],
    #     # [1,0,1,0,1,0,1,0,1,0,1],
    #     # [2,0,2,0,2,0,2,0,2,0,2],
    # ]

    # all_num_combinations = create_non_repeating_next_number_sequence(3, max_n)
    # print("all_num_combinations: {}".format(all_num_combinations))
    print("len(all_num_combinations): {}".format(len(all_num_combinations)))

    for num_combination in all_num_combinations:
        base_moves = [base_moves_names[i] for i in num_combination]
        print("base_moves: {}".format(base_moves))

        found_neutral_moves = []
        def find_neutral_moves(i, length, field, moves):
            p = base_moves[i]
            p_moves = d[p]
            is_last_move = i >= length-1
            if is_last_move:
                for c, r in combinations_moves:
                    field_ = field.copy()
                    move = p_moves[c][r]
                    do_moves(field_, [move])
                    # print("i: {}, field_:\n{}".format(i, field_))
                    if np.sum(field_!=num_field)==0:
                        # print("field_:\n{}".format(field_))
                        moves_ = moves+[move]
                        found_neutral_moves.append(moves_)
                        print("len(found_neutral_moves): {}, moves_: {}".format(len(found_neutral_moves), moves_))
                        # field_ = num_field.copy()
                        # for i in range(0, 4):
                        #     field_[i] = np.arange(4*i, 4*(i+1))
                        # print("before: field_:\n{}".format(field_))
                        # do_moves(field_, moves_)
                        # print("after: field_:\n{}".format(field_))
            else:
                for c, r in combinations_moves:
                    field_ = field.copy()
                    move = p_moves[c][r]
                    do_moves(field_, [move])
                    # print("i: {}, field_:\n{}".format(i, field_))
                    moves_ = moves+[move]
                    find_neutral_moves(i+1, length, field_, moves_)

        find_neutral_moves(0, len(base_moves), num_field, [])
        print("len(found_neutral_moves):\n{}".format(len(found_neutral_moves)))

        base_moves_str = "".join(base_moves)
        if len(found_neutral_moves) > 0:
            all_found_neutral_moves[base_moves_str] = found_neutral_moves
        else:
            all_base_moves_not_neutral_moves.append(base_moves_str)

    lens = sorted([(k, len(k), len(v)) for k, v in all_found_neutral_moves.items()], key=lambda x: (x[2], x[1], x[0]))
    print("lens:\n{}".format(lens))

    for base_moves, len_base_moves, amount in lens:
        print("base_moves: {}, len_base_moves: {}, amount: {}".format(base_moves, len_base_moves, amount))

    # for key, val in all_found_neutral_moves.items():
    #     print("key: {}, len(val): {}".format(key, len(val)))

    la = [l for li in all_found_neutral_moves.values() for l in li]
    print("len(la): {}".format(len(la)))
    python_code = 'neutral_moves = [\n'+",\n".join(["    "+str(l).replace(" ", "") for l in la])+'\n]\n'
    with open('utils_03_1.py', 'w') as f:
        f.write(python_code)

    return all_found_neutral_moves, all_base_moves_not_neutral_moves


if __name__ == "__main__":
    # test_rotation_of_fields()

    # test_rotation_of_fields_nr_2()

    # find_solutions_for_3x3_block()
    all_found_neutral_moves, all_base_moves_not_neutral_moves = find_a_lot_neutral_moves_for_3x3_block()
    sys.exit(-1)

    w = 15
    h = w

    create_moving_table_json_file("field_moving_table_{w}x{h}.json".format(w=w, h=h), w, h)
