#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback
import logging
import random

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

from pprint import pprint
from typing import List, Set, Tuple, Dict

from PIL import Image

import importlib.util
spec = importlib.util.spec_from_file_location("utils", "../utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

import multiprocessing as mp

sys.path.append('..')
from utils_multiprocessing_manager import MultiprocessingManager

if __name__ == '__main__':
    print("Hello World!")

    n: int = 10
    s_cells: Set[Tuple[int, int]] = set([(x, y) for y in range(0, n) for x in range(0, n)])
    s_rects: Set[Tuple[int, int]] = set([(x, y) for x in range(1, n+1) for y in range(x, n+1)])

    d_cell_to_d_rect_size_to_s_cell: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[Set[Tuple[int, int]], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]] = {}
    # d_cell_to_d_rect_size_to_s_cell: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[Set[Tuple[int, int]], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = {}
    l_rect = []
    for y in range(0, n):
        for x in range(0, n):
            d_rect_size_to_s_cell: List[Tuple[Tuple[int, int], Tuple[Set[Tuple[int, int]], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = []
            # d_rect_size_to_s_cell: Dict[Tuple[int, int], Tuple[Set[Tuple[int, int]], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {}
            d_cell_to_d_rect_size_to_s_cell[(x, y)] = d_rect_size_to_s_cell
            for h in range(1, n - y + 1):
                for w in range(1, n - x + 1):
                    cell_top_left = (x, y)
                    cell_bottom_left = (x - 1, y + h)
                    cell_top_right = (x + w, y - 1)
                    d_rect_size_to_s_cell.append((
                        (w, h), (
                            set([(x + i, y + j) for j in range(0, h) for i in range(0, w)]),
                            cell_top_left,
                            cell_bottom_left,
                            cell_top_right,
                        )
                    ))

    s_rects_harmon = set()
    d_rect_size_to_rect_size_harmon = {}
    for w in range(1, n + 1):
        rect_size_square = (w, w)
        d_rect_size_to_rect_size_harmon[(w, w)] = rect_size_square
        s_rects_harmon.add(rect_size_square)

        for h in range(w + 1, n + 1):
            rect_size = (w, h)
            d_rect_size_to_rect_size_harmon[(w, h)] = rect_size
            d_rect_size_to_rect_size_harmon[(h, w)] = rect_size
            s_rects_harmon.add(rect_size)

    s_cells_free = set(s_cells)
    s_rects_harmon_free = set(s_rects_harmon)

    l_parameters = []

    def find_valid_rect_sizes_params(
            n,
            call_i,
            l_found_solutions,
            l_used_cell,
            l_used_rect_size,
            l_next_cell,
            s_cells_free,
            s_rects_harmon_free,
            max_call=1,
    ):
        last_cell = l_next_cell[0]
        d_rect_size_to_s_cell = d_cell_to_d_rect_size_to_s_cell[last_cell]

        for rect_size, (s_cell, cell_top_left, cell_bottom_left, cell_top_right) in d_rect_size_to_s_cell:
            if s_cells_free & s_cell != s_cell:
                continue

            rect_size_harm = d_rect_size_to_rect_size_harmon[rect_size]
            if rect_size_harm not in s_rects_harmon_free:
                continue

            l_used_cell_new = l_used_cell + [cell_top_left]
            l_used_rect_size_new = l_used_rect_size + [rect_size]

            l_next_cell_new = list(l_next_cell[1:])
            if cell_bottom_left not in s_cells_free:
                x1, y1 = cell_bottom_left
                if y1 < n:
                    l_next_cell_new.append((x1 + 1, y1))
            if cell_top_right not in s_cells_free:
                x2, y2 = cell_top_right
                if x2 < n:
                    l_next_cell_new.append((x2, y2 + 1))

            if len(l_next_cell_new) == 0:
                continue

            s_cells_free_new = s_cells_free - s_cell
            s_rects_harmon_free_new = s_rects_harmon_free - set([rect_size_harm])

            if len(s_cells_free_new) == 0:
                print('l_used_rect_size_new: {}'.format(l_used_rect_size_new))
                l_found_solutions.append((l_used_cell_new, l_used_rect_size_new))
                assert sum([w * h for w, h in l_used_rect_size_new]) == n ** 2
                continue

            if call_i >= max_call:
                print("call_i: {}". format(call_i))
                l_parameters.append((
                    n,
                    l_used_cell_new,
                    l_used_rect_size_new,
                    l_next_cell_new,
                    s_cells_free_new,
                    s_rects_harmon_free_new,
                ))

                continue

            # if call_i >= max_call:
            #     continue

            find_valid_rect_sizes_params(
                n,
                call_i + 1,
                l_found_solutions,
                l_used_cell_new,
                l_used_rect_size_new,
                l_next_cell_new,
                s_cells_free_new,
                s_rects_harmon_free_new,
                max_call=max_call,
            )
    l_found_solutions_basic = []
    find_valid_rect_sizes_params(n, 0, l_found_solutions_basic, [], [], [(0, 0)], s_cells_free, s_rects_harmon_free, max_call=2)

    def find_valid_rect_sized_all(args):
        l_found_solutions = []

        try:
            def find_valid_rect_sizes(n, l_found_solutions, l_used_cell, l_used_rect_size, l_next_cell, s_cells_free, s_rects_harmon_free):
                last_cell = l_next_cell[0]

                d_rect_size_to_s_cell = d_cell_to_d_rect_size_to_s_cell[last_cell]

                random.shuffle(d_rect_size_to_s_cell)
                for rect_size, (s_cell, cell_top_left, cell_bottom_left, cell_top_right) in d_rect_size_to_s_cell:
                    # for rect_size, (s_cell, cell_top_left, cell_bottom_left, cell_top_right) in \
                    #       d_rect_size_to_s_cell.items():

                    if s_cells_free & s_cell != s_cell:
                        continue

                    rect_size_harm = d_rect_size_to_rect_size_harmon[rect_size]
                    if rect_size_harm not in s_rects_harmon_free:
                        continue

                    l_used_cell_new = l_used_cell + [cell_top_left]
                    l_used_rect_size_new = l_used_rect_size + [rect_size]

                    l_next_cell_new = list(l_next_cell[1:])
                    if cell_bottom_left not in s_cells_free:
                        x1, y1 = cell_bottom_left
                        if y1 < n:
                            l_next_cell_new.append((x1 + 1, y1))
                    if cell_top_right not in s_cells_free:
                        x2, y2 = cell_top_right
                        if x2 < n:
                            l_next_cell_new.append((x2, y2 + 1))

                    if len(l_next_cell_new) == 0:
                        continue

                    s_cells_free_new = s_cells_free - s_cell
                    s_rects_harmon_free_new = s_rects_harmon_free - set([rect_size_harm])

                    if len(s_cells_free_new) == 0:
                        # print the results!
                        # print('l_used_cell_new: {}'.format(l_used_cell_new))
                        # print('l_used_rect_size_new: {}'.format(l_used_rect_size_new))

                        l_found_solutions.append((l_used_cell_new, l_used_rect_size_new))

                        assert sum([w * h for w, h in l_used_rect_size_new]) == n ** 2
                        if len(l_found_solutions) >= 1000:
                            raise Exception()
                        continue

                    find_valid_rect_sizes(
                        n,
                        l_found_solutions,
                        l_used_cell_new,
                        l_used_rect_size_new,
                        l_next_cell_new,
                        s_cells_free_new,
                        s_rects_harmon_free_new,
                    )

            # find_valid_rect_sizes(n, l_found_solutions, [], [], [(0, 0)], s_cells_free, s_rects_harmon_free)
            l_args = list(args)
            l_args.insert(1, l_found_solutions)
            find_valid_rect_sizes(*l_args)
        except Exception as e:
            logging.error(logging.traceback.format_exc())

        return l_found_solutions


    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_find_valid_rect_sized_all', find_valid_rect_sized_all)

    print('Do the jobs!!')
    # l_arguments = [(args, )for args in l_parameters]
    l_arguments = [(args, )for args in l_parameters[:100]]
    l_ret = mult_proc_mng.do_new_jobs(
        ['func_find_valid_rect_sized_all'] * len(l_arguments),
        l_arguments,
    )
    print("len(l_ret): {}".format(len(l_ret)))

    del mult_proc_mng

    l_t_l_cell_l_rect = [t for l in l_ret for t in l]


    # remove duplicates from l_t_l_cell_l_rect!

    # d_found_dups = {}
    s_t_l_cell_l_rect = set()
    for l_cell, l_rect in l_t_l_cell_l_rect:
        # combine x, y, w, h info
        t_xywh = tuple(sorted([(x, y, w, h) for (x, y), (w, h) in zip(l_cell, l_rect)]))

        if t_xywh in s_t_l_cell_l_rect:
            print('Exists already!')
            continue

        def calc_rot_pos_size(x, y, w, h):
            x1 = x
            y1 = y
            x2 = x + w - 1
            y2 = y + h - 1

            return (min(n-y1-1, n-y2-1), min(x1, x2), h, w)


        t_xywh_rot1 = tuple(sorted([calc_rot_pos_size(x, y, w, h) for x, y, w, h in t_xywh]))
        if t_xywh_rot1 in s_t_l_cell_l_rect:
            # d_found_dups[t_xywh_rot1].add(t_xywh)
            continue

        t_xywh_rot2 = tuple(sorted([calc_rot_pos_size(x, y, w, h) for x, y, w, h in t_xywh_rot1]))
        if t_xywh_rot2 in s_t_l_cell_l_rect:
            # d_found_dups[t_xywh_rot2].add(t_xywh)
            continue

        t_xywh_rot3 = tuple(sorted([calc_rot_pos_size(x, y, w, h) for x, y, w, h in t_xywh_rot2]))
        if t_xywh_rot3 in s_t_l_cell_l_rect:
            # d_found_dups[t_xywh_rot3].add(t_xywh)
            continue

        s_t_l_cell_l_rect.add(t_xywh)
        # d_found_dups[t_xywh] = set()

    rows = 200
    cols = 200
    # image for the little rectangles!
    pix = np.zeros((rows*(n+1)-1, cols*(n+1)-1, 3), dtype=np.uint8)

    l_used_color_per_channel = [0x40, 0x80, 0xC0, 0xFF]

    arr_colors = np.array(
        [
            [r, g, b]
            for r in l_used_color_per_channel
            for g in l_used_color_per_channel
            for b in l_used_color_per_channel
        ],
        dtype=np.uint8,
    )

    l_t_l_cell_l_rect_no_rot_dup = list(s_t_l_cell_l_rect)

    l_len = [len(l) for l in l_t_l_cell_l_rect_no_rot_dup]
    max_len = max(l_len)
    l_t_l_cell_l_rect_no_rot_dup_max = [l for l, length in zip(l_t_l_cell_l_rect_no_rot_dup, l_len) if length == max_len]

    random.shuffle(l_t_l_cell_l_rect_no_rot_dup_max)

    u, c = np.unique(l_len, return_counts=True)
    print("u: {}".format(u))
    print("c: {}".format(c))

    for j, t_xywh in enumerate(l_t_l_cell_l_rect_no_rot_dup_max[:rows*cols], 0):
        j_row = j // cols
        j_col = j % cols

        j_row_add = j_row * (n + 1)
        j_col_add = j_col * (n + 1)

        np.random.shuffle(arr_colors)

        for i, (x, y, w, h) in enumerate(t_xywh, 0):
            pix[j_row_add+y:j_row_add+y+h, j_col_add+x:j_col_add+x+w] = arr_colors[i]

    img = Image.fromarray(pix)
    img.save(os.path.join(TEMP_DIR, 'image_n_{}_square_size.png'.format(n)))

    # l_found_solutions = []
    # for i, params in enumerate(l_parameters[:10], 0):
    #     print("i: {}".format(i))
    #     l_found_solutions_parts = find_valid_rect_sized_all(params)
    #
    #     l_found_solutions.extend(l_found_solutions_parts)
    #
    # # TODO: add rotating, and mirroring for the found solutions!
    # # TODO: create classes for each fields or variables too!
    # # TODO: make it faster with inplace replacing the rect
    # l_solutions_rect_harmon = [tuple(sorted(set(l_rect_harmon))) for _, l_rect_harmon in l_found_solutions]
    # l_solutions_rect_harmon_unique = sorted(set(l_solutions_rect_harmon))
    #
    print('- len(l_parameters): {}'.format(len(l_parameters)))

    print('n: {}'.format(n))
    print('- len(l_t_l_cell_l_rect_no_rot_dup): {}'.format(len(l_t_l_cell_l_rect_no_rot_dup)))
    # print('- len(l_solutions_rect_harmon_unique): {}'.format(len(l_solutions_rect_harmon_unique)))
