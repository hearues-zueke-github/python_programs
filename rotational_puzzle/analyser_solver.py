#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import mmap
import os
import re
import sys
import time

import itertools
import multiprocessing

from multiprocessing import Process, Pipe
# from multiprocessing import shared_memory # in python3.8 available!

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

from copy import copy, deepcopy

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


def get_building_parsin_functions(w, h):
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
                    print("cell: {}".format(cell))
                    if cell in moving_dict:
                        neighs_4_rest.pop(neighs_4_rest.index(cell))
                # print("neighs_4_rest: {}".format(neighs_4_rest))

                for rot_center in rots_centers:
                    rot_moves = rotation_cell_dict[rot_center]
                    for neigh_cell in neighs_4_rest:
                        if rot_moves[0][center_cell] == neigh_cell and not neigh_cell in moving_dict:
                            moving_dict[neigh_cell] = []
                            moving_dict[center_cell].append((neigh_cell, rot_center+(1, False)))
                            
                            possible_moves[center_cell] = moving_dict[center_cell]
                        elif rot_moves[1][center_cell] == neigh_cell and not neigh_cell in moving_dict:
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


    return get_absolute_moves_per_center_cell


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
        moves_inverted.append(move[:-1]+(move[-1]==False, ))
    return moves_inverted


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


if __name__ == "__main__":
    # test_rotation_of_fields()

    w, h = 6, 5
    field_size = (h, w) # (y, x)
    max_num = np.multiply.reduce(field_size)
    num_field = np.arange(0, max_num).reshape((field_size))
    print("num_field:\n{}".format(num_field))
    _, _, _, _, f_rot = get_rot_functions()

    get_absolute_moves_per_center_cell = get_building_parsin_functions(w, h)

    moving_dict_field = {}
    
    start_x = w-1
    start_y = h-1
    moving_dict_start = {(start_x, start_y): []}
    absolute_moving_dict = get_absolute_moves_per_center_cell(start_x, start_y, moving_dict_start)
    # moving_dict = build_best_moves(start_x, start_y, moving_dict_start)
    # absolute_moving_dict = parse_moving_dict(start_x, start_y, moving_dict)

    moving_dict_field[w*h-1] = absolute_moving_dict

    start_x = w-2
    start_y = h-1
    moving_dict_start = {(start_x, start_y): [], (start_x+1, start_y): []}
    absolute_moving_dict = get_absolute_moves_per_center_cell(start_x, start_y, moving_dict_start)
    # moving_dict = build_best_moves(start_x, start_y, moving_dict_start)
    # absolute_moving_dict = parse_moving_dict(start_x, start_y, moving_dict)

    moving_dict_field[w*h-2] = absolute_moving_dict

    print("num_field:\n{}".format(num_field))

    for move in absolute_moving_dict[(0, 0)]:
        f_rot(*((num_field, )+move))

    print("after doing moves for (0, 0): num_field:\n{}".format(num_field))
