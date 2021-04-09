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

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

from math import factorial as fac

sys.path.append('..')
from utils import mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

class UsedFields(Exception):
    __slots__ = ['i']

    def __init__(self, symbol_nr, l_used_fields, l_pos, l_field, l_field_score):
        pass


if __name__ == '__main__':
    print("Hello World!")

    l = [(9, 9-(i//2)*2-(i%2), (i+1)//2, i//2) for i in range(0, 10)]
    print("l: {}".format(l))

    l_combos = [fac(n)//fac(i1)//fac(i2)//fac(i3) for n, i1, i2, i3 in l]
    print("l_combos: {}".format(l_combos))

    amount_combinations = sum(l_combos)
    print("amount_combinations: {}".format(amount_combinations))

    sys.exit(0)

    n = 3
    l_positions = [(x, y) for y in range(0, n) for x in range(0, n)]
    print("l_positions: {}".format(l_positions))

    s_positions = set(l_positions)

    l_symbols_nr = [1, 2]
    # d_symbols = {1: 'x', 2: 'o'}
    d_next_symbol_nr = {1: 2, 2: 1}

    field = np.zeros((n, n), dtype=np.uint8)

    d_field_to_next_field = {}

    # field[:] = (field.flatten()+list(range(0, 9))).reshape((3, 3))

    d_pos_list_clusters = {}

    # length 1
    d_pos_list_clusters[1] = [([y], [x]) for y in range(0, n) for x in range(0, n)]
    
    d_pos_list_clusters[2] = \
        [([y, y+1], [x, x]) for y in range(0, n-1) for x in range(0, n)] + \
        [([y, y], [x, x+1]) for y in range(0, n) for x in range(0, n-1)] + \
        [([y, y+1], [x, x+1]) for y in range(0, n-1) for x in range(0, n-1)] + \
        [([y, y+1], [x+1, x]) for y in range(0, n-1) for x in range(0, n-1)]

    d_pos_list_clusters[3] = \
        [([y, y+1, y+2], [x, x, x]) for y in range(0, n-2) for x in range(0, n)] + \
        [([y, y, y], [x, x+1, x+2]) for y in range(0, n) for x in range(0, n-2)] + \
        [([y, y+1, y+2], [x, x+1, x+2]) for y in range(0, n-2) for x in range(0, n-2)] + \
        [([y, y+1, y+2], [x+2, x+1, x]) for y in range(0, n-2) for x in range(0, n-2)]

    def combine_list_positions(l_pos_list):
        l_y = []
        l_x = []

        for l_y_next, l_x_next in l_pos_list:
            l_y += l_y_next
            l_x += l_x_next

        return (l_y, l_x)

    for i in range(1, n+1):
        d_pos_list_clusters[i] = combine_list_positions(d_pos_list_clusters[i])

    # sys.exit()

    found_fields = 0
    def add_a_symbol(s_t_field, field, n, s_positions, symbol_nr, depth):
        try:
            l_sums = []
            for symbol_nr_iter in l_symbols_nr:
                l_sums.append(np.sum(field == symbol_nr_iter))

            for sums1 in l_sums:
                for sums2 in l_sums:
                    assert abs(sums1 - sums2) <= 1

            global found_fields
            print("found_fields: {}".format(found_fields))
            found_fields += 1

            t_field = tuple([tuple([v for v in row]) for row in field])

            print("t_field: {}".format(t_field))

            d_pos_to_new_field = {}
            d_field_to_next_field[t_field] = d_pos_to_new_field

            next_symbol_nr = d_next_symbol_nr[symbol_nr]
            for t_pos in s_positions:
                x, y = t_pos

                s_positions_next = set(s_positions)
                s_positions_next.remove(t_pos)
                # s_positions.remove(t_pos)

                # Set the new symbol
                assert field[y, x] == 0
                field[y, x] = symbol_nr

                # check if the board was already set (rotating, mirroring)
                # TODO: later

                # calculate the score of the best possible position
                d_scores = {symbol_nr_iter: {} for symbol_nr_iter in l_symbols_nr}
                for i in range(1, n+1):
                    field_clusters = field[d_pos_list_clusters[i]].reshape((-1, i))
                    for symbol_nr_iter in l_symbols_nr:
                        d_scores[symbol_nr_iter][i] = np.sum(np.all(field_clusters == symbol_nr_iter, axis=1))

                t_field_new = tuple([tuple([v for v in row]) for row in field])
                assert t_pos not in d_pos_to_new_field
                d_pos_to_new_field[t_pos] = (symbol_nr, t_field_new, d_scores)

                if len(s_positions_next) > 0:
                    add_a_symbol(s_t_field, field, n, s_positions_next, next_symbol_nr, depth+1)

                # remove symbol_nr
                assert field[y, x] != 0
                field[y, x] = 0
                # add again the s_positions t_pos back
                # s_positions.add(t_pos)
        except:
            globals()['loc_{}'.format(depth)] = locals()
            sys.exit()

    s_t_field = set()
    add_a_symbol(s_t_field, field, n, s_positions, l_symbols_nr[0], 0)
