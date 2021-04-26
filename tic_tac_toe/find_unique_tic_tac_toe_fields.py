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

from pprint import pprint

from math import factorial as fac

sys.path.append('..')
from utils import mkdirs

from typing import List, Tuple, Dict, Set

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

# amount_symbols = 3 -> for 0, 1 and 2.
# While 0 is an empty cell, 1 is e.g. 'x' and 2 is 'o'
def convert_field_to_number(arr: np.ndarray, amount_symbols: int=3) -> int:
    s: int = 0
    mult: int = 1
    v: np.uint8
    for v in arr.flatten():
        s += v * mult
        mult *= amount_symbols

    return s


def convert_number_to_field(i: int, n: int, amount_symbols: int=3) -> int:
    arr = np.zeros((n**2, ), dtype=np.uint8)

    for pos in range(0, n**2):
        arr[pos] = i % amount_symbols
        i //= amount_symbols

    assert i == 0

    return arr.reshape((n, n))


if __name__ == '__main__':
    print("Hello World!")

    n = 3
    arr = np.zeros((n, n), dtype=np.uint8)

    l_symbols_nr: List[int] = [1, 2]
    d_next_symbol_nr: Dict[int, int] = {sym1: sym2 for sym1, sym2 in zip(l_symbols_nr, l_symbols_nr[1:]+l_symbols_nr[:1])}

    print("l_symbols_nr: {}".format(l_symbols_nr))
    print("d_next_symbol_nr: {}".format(d_next_symbol_nr))

    print("arr:\n{}".format(arr))

    # arr[:] = 2
    # print("arr:\n{}".format(arr))

    s_pos: Set[Tuple[int, int]] = set([(x, y) for y in range(0, n) for x in range(0, n)])

    def convert_to_tuple(arr: np.ndarray) -> Tuple[Tuple[int]]:
        return tuple(tuple(row) for row in arr.tolist())

    # for 2-d only so far
    def get_all_possible_mirror_field(arr: np.ndarray) -> Set[Tuple[Tuple[int]]]:

        s_field: Set[Tuple[Tuple[int]]] = set()
        s_field.add(convert_to_tuple(arr))

        arr = np.flip(arr, 0)
        s_field.add(convert_to_tuple(arr))
        arr = np.flip(arr, 1)
        s_field.add(convert_to_tuple(arr))
        arr = np.flip(arr, 0)
        s_field.add(convert_to_tuple(arr))

        arr = np.transpose(arr, (1, 0))
        s_field.add(convert_to_tuple(arr))

        arr = np.flip(arr, 0)
        s_field.add(convert_to_tuple(arr))
        arr = np.flip(arr, 1)
        s_field.add(convert_to_tuple(arr))
        arr = np.flip(arr, 0)
        s_field.add(convert_to_tuple(arr))

        return s_field

    d_fields = {}
    d_fields_turn_nr = {}

    d_fields_part = {}
    d_fields_part[convert_to_tuple(arr)] = [0, 1, l_symbols_nr[-1], set(), tuple()]

    d_fields_turn_nr[0] = d_fields_part

    for k, v in d_fields_part.items():
        d_fields[k] = v

    turn_nr_iter = 0
    while d_fields_part:
        turn_nr_iter += 1
        print("turn_nr_iter: {}".format(turn_nr_iter))
        print("len(d_fields): {}".format(len(d_fields)))
        print("len(d_fields_part): {}".format(len(d_fields_part)))
        
        d_fields_part_new = {}

        for t_field, (turn_nr, similar_fields, prev_symbol, s_used_pos, t_last_pos) in d_fields_part.items():
            symbol = d_next_symbol_nr[prev_symbol]

            arr = np.array(t_field, dtype=np.uint8)
            s_pos_avail = s_pos - s_used_pos

            for x, y in s_pos_avail:
                assert arr[y, x] == 0
                arr[y, x] = symbol

                s_field = get_all_possible_mirror_field(arr)

                is_in_dict = False
                for field in s_field:
                    if field in  d_fields.keys() or field in d_fields_part_new.keys():
                        is_in_dict = True
                        break

                if is_in_dict:
                    arr[y, x] = 0
                    continue

                field_unique = convert_to_tuple(arr)
                d_fields_part_new[field_unique] = [turn_nr_iter, len(s_field), symbol, s_used_pos|set([(x, y)]), (x, y)]

                arr[y, x] = 0

        d_fields_part = d_fields_part_new
        for k, v in d_fields_part_new.items():
            d_fields[k] = v

        d_fields_turn_nr[turn_nr_iter] = d_fields_part

    df = pd.DataFrame(index=np.arange(0, len(d_fields)), dtype=object)
    df['field'] = None
    df['turn_nr'] = None
    df['similar_fields'] = None
    df['prev_symbol'] = None
    df['s_used_pos'] = None
    df['t_last_pos'] = None

    for i, (t_field, (turn_nr, similar_fields, prev_symbol, s_used_pos, t_last_pos)) in enumerate(d_fields.items(), 0):
        row = df.loc[i]
        row['field'] = t_field
        row['turn_nr'] = turn_nr
        row['similar_fields'] = similar_fields
        row['prev_symbol'] = prev_symbol
        row['s_used_pos'] = s_used_pos
        row['t_last_pos'] = t_last_pos
    # arr[0, 0] = 1
    # arr[0, 2] = 2
    # s_field = get_all_possible_mirror_field(arr)
    # print("s_field:")
    # pprint(s_field)
