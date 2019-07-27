#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import pdb
import sys

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from functools import reduce

from dotmap import DotMap

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table

get_permutation_table

USER_HOME_PATH = os.path.expanduser('~')+'/'
print("USER_HOME_PATH: {}".format(USER_HOME_PATH))

def get_max_cycles_length_for_permutation(n):
    arr = get_permutation_table(n, same_pos=False)

    max_cycles = 0
    cycles_lengths = []
    row_orig = np.arange(0, n)
    for row in arr:
        row_i = row_orig.copy()
        is_full_one_cycle = True
        i = 1
        row_i = row_i[row]
        while not np.all(row_i==row_orig):
            row_i = row_i[row]
            i += 1
        if max_cycles < i:
            max_cycles = i
        cycles_lengths.append(i)

    return max_cycles


def get_landau_values(n_max): # see also: A000793
    print("n: {}, length: {}".format(0, 1))
    print("n: {}, length: {}".format(1, 1))
    lens = [1, 1]
    for n in range(2, n_max+1):
        length = get_max_cycles_length_for_permutation(n)
        print("n: {}, length: {}".format(n, length))
        lens.append(length)

    return lens


if __name__ == "__main__":
    lens = get_landau_values(10)
    print("lens: {}".format(lens))

