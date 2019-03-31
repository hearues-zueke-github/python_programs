#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from dotmap import DotMap

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence

sys.path.append("..")
import utils_all

def create_tree_dict_path(n, next_pos_movess):
    print("n: {}".format(n))

    # U ... up or like north
    # D ... down or like south
    # L ... left or like west
    # R ... right or like east
    dict_path = {(0, 0): DotMap({'dir': 'U', 'point_parent': None, 'points_abs': [], 'points_rel': []})}


    points_now = [(0, 0)]

    def get_dict_next_pos_rel(next_pos_movess):
        moves_pos_rel = {'R': (0, 1), 'L': (0, -1), 'U': (1, 0), 'D': (-1, 0)}
        get_pos_rel_lst = lambda next_pos_moves: [moves_pos_rel[move] for move in next_pos_moves]
        next_pos_rel = [(moves, get_pos_rel_lst(moves)) for moves in next_pos_movess]

        dict_next_pos_rel = {'U': next_pos_rel}

        directions = ['U', 'R', 'D', 'L']
        directions_num = {'U': 0, 'R': 1, 'D': 2, 'L': 3}

        for i in range(1, 4):
            d = directions[i]
            next_pos_rel_new = []

            for pos_rel in next_pos_rel:
                d2 = directions[i]
                dirs = []
                poss = []
                for d1, pos in zip(pos_rel[0], pos_rel[1]):
                    if d2 == 'R':
                        poss.append((-pos[1], pos[0]))
                    elif d2 ==  'D':
                        poss.append((-pos[0], -pos[1]))
                    elif d2 ==  'L':
                        poss.append((pos[1], -pos[0]))
                    else:   
                        poss.append(pos)

                    d3 = directions[(directions_num[d1]+i) % 4]
                    dirs.append(d3)

                next_pos_rel_new.append((dirs, poss))
            dict_next_pos_rel[d] = next_pos_rel_new

        return dict_next_pos_rel

    dicts_next_pos_rel = [get_dict_next_pos_rel(next_pos_moves) for next_pos_moves in next_pos_movess]

    # dict_next_pos_rel_1 = get_dict_next_pos_rel(next_pos_movess_1)
    # dict_next_pos_rel_2 = get_dict_next_pos_rel(next_pos_movess_2)
    # dict_next_pos_rel_3 = get_dict_next_pos_rel(next_pos_movess_3)

    # dicts_next_pos_rel = [dict_next_pos_rel_1, dict_next_pos_rel_2, dict_next_pos_rel_3]
    idx_dict_next_pos_rel = 0

    for it in range(0, n):
        # print("it: {}".format(it))
        # print("dict_path: {}".format(dict_path))
        points_now_new = []
        for point_now in points_now:
            obj_now = dict_path[point_now]

            y, x = point_now

            d = obj_now.dir

            dict_next_pos_rel_now = dicts_next_pos_rel[idx_dict_next_pos_rel]

            is_used_dict = False
            for next_poss_rel in dict_next_pos_rel_now[d]:
                d1 = next_poss_rel[0][-1]
                poss = [point_now]
                add_points = lambda a, b: (a[0]+b[0], a[1]+b[1])
                for pos_rel in next_poss_rel[1]:
                    poss.append(add_points(poss[-1], pos_rel))
                # poss.pop(0)

                is_all_free = True
                for pos in poss[1:]:
                    if pos in dict_path:
                        is_all_free = False
                        break

                if not is_all_free:
                    continue

                is_used_dict = True
                for p1, p2, p2_rel in zip(poss[:-1], poss[1:], next_poss_rel[1]):
                    dict_path[p2] = DotMap({'dir': d1, 'point_parent': p1, 'points_abs': [], 'points_rel': []})
                    obj_p1 = dict_path[p1]
                    obj_p1.points_abs.append(p2)
                    obj_p1.points_rel.append(p2_rel)

                points_now_new.append(poss[-1])

            if is_used_dict:
                idx_dict_next_pos_rel = (idx_dict_next_pos_rel+1) % len(dicts_next_pos_rel)

        points_now = points_now_new

    # print("dict_path:")
    # for key, val in dict_path.items():
    #     print("key: {}, val: {}".format(key, val))

    return dict_path


def convert_dict_path_to_x_y_array(dict_path):
    point_pairs = []
    def walk_dict_path(p):
        # print("p: {}".format(p))
        points_abs = dict_path[p].points_abs
        # print("points_abs: {}".format(points_abs))
        for point in points_abs:
            point_pairs.append(p+point)
            walk_dict_path(point)

    walk_dict_path((0, 0))

    point_pairs = np.array(point_pairs)
    # print("point_pairs:\n{}".format(point_pairs))
    y1, x1, y2, x2 = point_pairs.T
    # print("y1:\n{}".format(y1))
    # print("x1:\n{}".format(x1))
    # print("y2:\n{}".format(y2))
    # print("x2:\n{}".format(x2))
    # print("y1==y2:\n{}".format((y1==y2)+0))
    # print("x1==x2:\n{}".format((x1==x2)+0))

    min_x = np.min((x1, x2))
    min_y = np.min((y1, y2))

    # print("min_x: {}, min_y: {}".format(min_x, min_y))

    x1 -= min_x
    x2 -= min_x
    y1 -= min_y
    y2 -= min_y

    max_x = np.max((x1, x2))
    max_y = np.max((y1, y2))
    # print("max_x: {}".format(max_x))
    # print("max_y: {}".format(max_y))

    arr_x = np.zeros((max_y+1, max_x), dtype=np.int)
    arr_y = np.zeros((max_y, max_x+1), dtype=np.int)

    idx = x1==x2

    # print("idx: {}".format(idx))

    arr_px = point_pairs[~idx].T
    # print("arr_px: {}".format(arr_px))
    arr_py = point_pairs[idx].T
    
    if arr_px.shape[0] > 0:
        px_xs = arr_px[[1, 3]]
        px_x = np.min(px_xs, axis=0)
        
        px_y = arr_px[0]

        # print("1 arr_x: {}".format(arr_x))
        # print("px_xs: {}".format(px_xs))
        # print("px_y: {}".format(px_y))
        # print("px_x: {}".format(px_x))
        arr_x[(px_y, px_x)] = 1
        # print("2 arr_x: {}".format(arr_x))
    else:
        px_x = np.array([])
        px_y = np.array([])

    if arr_py.shape[0] > 0:
        py_x = arr_py[1]
        
        py_ys = arr_py[[0, 2]]
        py_y = np.min(py_ys, axis=0)

        arr_y[(py_y, py_x)] = 1
    else:
        py_x = np.array([])
        py_y = np.array([])

    # print("px_y: {}".format(px_y))
    # print("px_x: {}".format(px_x))
    # print("py_y: {}".format(py_y))
    # print("py_x: {}".format(py_x))

    # print("px_ys:\n{}".format(px_ys))
    # print("py_xs:\n{}".format(py_xs))

    # print("arr_x.shape: {}".format(arr_x.shape))
    # print("arr_y.shape: {}".format(arr_y.shape))

    # print("point_pairs:\n{}".format(point_pairs))

    # arr_x[(px_y, px_x)] = 1
    # arr_y[(py_y, py_x)] = 1

    # print("arr_x:\n{}".format(arr_x))
    # print("arr_y:\n{}".format(arr_y))

    return (-min_y, -min_x), arr_y, arr_x


def create_image_of_y_x_arr(pos_0_offset, arr_y, arr_x):
    # l = 2**n
    if len(arr_y.shape) == 2:
        r1, c1 = arr_y.shape
    else:
        r1, c1 = 0, 0

    if len(arr_x.shape) == 2:
        r2, c2 = arr_x.shape
    else:
        r2, c2 = 0, 0
    
    rows = np.max((r1, r2))
    cols = np.max((c1, c2))
    # cols = arr_x.shape[1]

    print("rows: {}, cols: {}".format(rows, cols))
    # print("pos_0_offset: {}".format(pos_0_offset))
    # return 0

    # print("arr_y:\n{}".format(arr_y))
    # print("arr_x:\n{}".format(arr_x))

    pw = 3 # pixel width of one line
    pix = np.zeros(((rows-1)*(pw+1)+1 if rows > 1 else 1, (cols-1)*(pw+1)+1 if cols > 1 else 1, 3), dtype=np.uint8)

    col1 = (0xFF, 0xFF, 0x00)
    col2 = (0x00, 0xC0, 0x80)
    col_green = (0x00, 0xFF, 0x00)
    col_black = (0x00, 0x00, 0x00)
    col_white = (0xFF, 0xFF, 0xFF)

    for y in range(0, rows*pw+rows-1, pw+1):
        for x in range(0, cols*pw+cols-1, pw+1):
            pix[y, x] = col_white

    y_off, x_off = pos_0_offset
    if y_off >= 0 and x_off >= 0:
        pix[(pw+1)*(y_off), (pw+1)*(x_off)] = col2

    # return pix

    for y in range(0, arr_y.shape[0]):
        for x in range(0, arr_y.shape[1]):
            v = arr_y[y, x]
            py = y*(pw+1)+1
            px = x*(pw+1)
            if v == 1:
                pix[py:py+pw, px] = col1
            elif v == 2:
                pix[py:py+pw, px] = col_green

    for y in range(0, arr_x.shape[0]):
        for x in range(0, arr_x.shape[1]):
            v = arr_x[y, x]
            py = y*(pw+1)
            px = x*(pw+1)+1
            # print("py: {}, px: {}".format(py, px))
            if v == 1:
                pix[py, px:px+pw] = col1
            elif v == 2:
                pix[py, px:px+pw] = col_green

    return pix


if __name__ == '__main__':
    n = 100

    next_pos_movess = [
        # [['R', 'D', 'R', 'D', 'L'], ['L']],
        # [['L', 'D'], ['U', 'L']],
        
        [['R', 'U'], ['U', 'L'], ['L']],
        [['L', 'L'], ['U', 'R', 'D'], ['R', 'U']],
        # [['R'], ['L', 'U']],
        # [['L'], ['R', 'U']],

        # [['R'], ['L'], ['U', 'L']],
        # [['U', 'L'], ['R']],
        # [['R'], ['L']],
        # [['L'], ['R', 'D', 'R']],
        
        # [['L', 'L'], ['R']],
        # [['L'], ['U'], ['R']],
        # [['L'], ['U'], ['R', 'R', 'D']],
        # [['L', 'U'], ['R', 'D']],
    ]

    dict_path = create_tree_dict_path(n, next_pos_movess)
    pos_0_offset, arr_y, arr_x = convert_dict_path_to_x_y_array(dict_path)
    pix = create_image_of_y_x_arr(pos_0_offset, arr_y, arr_x)
    
    resize_factor = 4
    img = Image.fromarray(pix).transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize((img.width*resize_factor, img.height*resize_factor))


    path_img = 'images/tree_pattern_generator/'
    if not os.path.exists(path_img):
        os.makedirs(path_img)
    img.save(path_img+'2d_sequence_n_{}_{}.png'.format(n, utils_all.get_date_time_str_full()))
