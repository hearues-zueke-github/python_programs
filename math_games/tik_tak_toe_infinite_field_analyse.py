#! /usr/bin/python3

# -*- coding: utf-8 -*-

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

if __name__=='__main__':
    s_used_cells = set()
    s_x_cells = set()
    s_o_cells = set()

    # starting with the x at the cell (0, 0)==(y, x)

    s_used_cells.add((0, 0))
    s_x_cells.add((0, 0))
    # s_used_cells.add((0, 1))
    # s_o_cells.add((0, 1))

    # for each move of placing 'o', do a deep analyse search where the most 'x'
    # clusters are blocked at move 1, 2 etc.


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

    # s_free_cells = get_free_cells(s_used_cells)
    # print("s_free_cells: {}".format(s_free_cells))


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


    def find_best_cell_and_add(s_choosen_cells):
        s_choosen_free_cells = get_free_cells(s_used_cells)
        l = []
        for cell in s_choosen_free_cells:
            l.append((cell, get_amount_of_lengths(cell, s_choosen_cells)))
        l_sort = sorted(l, key=lambda x: (max(x[1]), x[0]))
        best_o = l_sort[-1][0]

        s_used_cells.add(best_o)
        s_choosen_cells.add(best_o)


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

    # sys.exit(0)

    def find_best_next_move_for_x_and_o(l_stats, s_used_cells, s_x_cells, s_o_cells, max_depth=3, l_used_o_x_cells=[], l_length_amount=[]):
        assert s_o_cells&s_x_cells==set()
        assert s_used_cells==s_o_cells|s_x_cells

        if max_depth<=0:
            # print("- s_x_cells: {}".format(s_x_cells))
            # print("- s_o_cells: {}".format(s_o_cells))
            d_counts = {}
            for x_cell in s_x_cells:
                t = get_amount_of_lengths(x_cell, s_x_cells)
                arr_u, arr_c = np.unique(t, return_counts=True)
                for u, c in zip(arr_u, arr_c):
                    if not u in d_counts:
                        d_counts[u] = 0
                    d_counts[u] += c
            # print("- d_counts: {}".format(d_counts))
            # l_stats.append((l_used_o_x_cells, tuple(((key, d_counts[key]) for key in sorted(d_counts.keys(), reverse=True)))))
            # l_stats.append((l_used_o_x_cells, d_counts))
            l_stats.append((l_used_o_x_cells, l_length_amount))
            return

        s_o_best_cells, max_length_o, max_amount_o = get_set_of_best_cells(s_used_cells, s_x_cells)
        for o_cell in s_o_best_cells:
            s_used_cells_new = s_used_cells|set([o_cell])
            s_o_cells_new = s_o_cells|set([o_cell])
            s_x_best_cells, max_length_x, max_amount_x = get_set_of_best_cells(s_used_cells_new, s_x_cells)
            # s_x_free_cells = get_free_cells(s_used_cells_new)
            for x_cell in s_x_best_cells:
                s_used_cells_new_2 = s_used_cells_new|set([x_cell])
                s_x_cells_new = s_x_cells|set([x_cell])

                # print("o_cell: {}, x_cell: {}".format(o_cell, x_cell))

                find_best_next_move_for_x_and_o(
                    l_stats,
                    s_used_cells_new_2,
                    s_x_cells_new,
                    s_o_cells_new,
                    max_depth=max_depth-1,
                    l_used_o_x_cells=l_used_o_x_cells+[(o_cell, x_cell)],
                    l_length_amount=l_length_amount+[(max_length_o, max_amount_o), (max_length_x, max_amount_x)]
                )


    max_depth = 4
    # find_best_cell_and_add(s_o_cells)
    for i_round in range(0, 100):
        print("i_round: {}".format(i_round))
        l_stats = []
        find_best_next_move_for_x_and_o(l_stats, s_used_cells, s_x_cells, s_o_cells, max_depth=max_depth)
        l_stats_sorted = sorted(l_stats, key=lambda x: x[1], reverse=True)
        l_cells_o_x, l_length_amount = l_stats_sorted[0]

        o_cell, x_cell = l_cells_o_x[0]
        s_used_cells.add(o_cell)
        s_used_cells.add(x_cell)
        s_x_cells.add(x_cell)
        s_o_cells.add(o_cell)

        # print("s_used_cells: {}".format(s_used_cells))
        # print("s_x_cells: {}".format(s_x_cells))
        # print("s_o_cells: {}".format(s_o_cells))

        print("len(l_stats): {}".format(len(l_stats)))

        print("l_cells_o_x: {}".format(l_cells_o_x))
        print("l_length_amount: {}".format(l_length_amount))

    # create a dirty solution for showing a picture!

    max_length = 0
    for x_cell in s_x_cells:
        t = get_amount_of_lengths(x_cell, s_x_cells)
        length = max(t)
        if max_length<length:
            max_length = length
    print("\nmax_length: {}\n".format(max_length))

    # find the needed size for the picture!
    min_y = 0
    min_x = 0
    max_y = 0
    max_x = 0

    for y, x in s_used_cells:
        if y<min_y:
            min_y = y
        if x<min_x:
            min_x = x
        if y>max_y:
            max_y = y
        if x>max_x:
            max_x = x

    print("min_y: {}".format(min_y))
    print("min_x: {}".format(min_x))
    print("max_y: {}".format(max_y))
    print("max_x: {}".format(max_x))

    # convert s_x_cells and s_o_cells to list first!

    size_w = max_x-min_x+1
    size_h = max_y-min_y+1

    pix = np.zeros((size_h, size_w, 3), dtype=np.uint8)

    for y, x in s_x_cells:
        pix[y-min_y, x-min_x] = (255, 0, 0)
    for y, x in s_o_cells:
        pix[y-min_y, x-min_x] = (0, 0, 255)

    # create a pix with grid
    cell_width = 8
    pix2 = np.zeros((size_h*cell_width+size_h+1, size_w*cell_width+size_w+1, 3), dtype=np.uint8)

    pix2[:] = (128, 128, 128)
    for i in range(0, pix2.shape[1], cell_width+1):
        pix2[:, i] = (0, 0, 0)
    for i in range(0, pix2.shape[0], cell_width+1):
        pix2[i, :] = (0, 0, 0)

    for y, x in s_x_cells:
        y1 = 1+(cell_width+1)*(y-min_y)
        x1 = 1+(cell_width+1)*(x-min_x)
        pix2[y1:y1+cell_width, x1:x1+cell_width] = (255, 0, 0)
    for y, x in s_o_cells:
        y1 = 1+(cell_width+1)*(y-min_y)
        x1 = 1+(cell_width+1)*(x-min_x)
        pix2[y1:y1+cell_width, x1:x1+cell_width] = (0, 0, 255)

    # img = Image.fromarray(pix)
    # resize_factor = 4
    # # resize_factor = 10
    # img = img.resize((size_w*resize_factor, size_h*resize_factor))
    # img.show()
    
    resize_factor = 1
    img2 = Image.fromarray(pix2)
    img2 = img2.resize((img2.width*resize_factor, img2.height*resize_factor))
    img2.show()
