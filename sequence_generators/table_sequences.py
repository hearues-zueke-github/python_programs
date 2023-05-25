#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.11 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import itertools
import os
import pdb
import re
import sys
import time
import traceback

import numpy as np # need installation from pip
import pandas as pd # need installation from pip
import multiprocessing as mp

import matplotlib.pyplot as plt # need installation from pip

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap # need installation from pip
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile # need installation from pip
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':

    assert len(sys.argv) > 1
    inp_str = sys.argv[1]
    d_key_val =  dict(kv.split('=') for kv in inp_str.split(','))

    n = int(d_key_val['n'])
    base = int(d_key_val['base'])
    max_iter = int(d_key_val['max_iter'])

    print("n: {}, base: {}".format(n, base))

    max_cycle_len = 0
    best_factors = []
    l_missing_factors = []

    amount = 0

    for a_c in range(0, base):
     for a_l in range(0, base):
      for a_r in range(0, base):
       for a_u in range(0, base):
        for a_d in range(0, base):
            factors = (a_c, a_l, a_r, a_u, a_d, )

            arr = np.zeros((n, n), dtype=np.uint8)
            arr[0, 0] = 1
            
            l_arr = [arr]

            for k in range(1, max_iter):
                arr_l = np.roll(arr, 1, axis=1)
                arr_r = np.roll(arr, -1, axis=1)
                arr_u = np.roll(arr, 1, axis=0)
                arr_d = np.roll(arr, -1, axis=0)
                
                arr_new = (a_c*arr + a_l*arr_l + a_r*arr_r + a_u*arr_u + a_d*arr_d) % base

                l_arr.append(arr_new)
                arr = arr_new

            k_last = len(l_arr)-1
            arr_last = l_arr[-1]
            is_found_same = False
            for k, arr_k in zip(range(k_last-1, -1, -1), reversed(l_arr[:-1])):
                if np.all(arr_last==arr_k):
                    is_found_same = True
                    break

            del l_arr

            if is_found_same:
                cycle_len = k_last-k
                if max_cycle_len < cycle_len:
                    max_cycle_len = cycle_len
                    best_factors = [factors]
                elif max_cycle_len == cycle_len:
                    best_factors.append(factors)
            else:
                l_missing_factors.append(factors)

            amount += 1
            if amount % 10 == 0:
                print("amount: {}, max_cycle_len: {}, len(best_factors): {}, len(l_missing_factors): {}".format(
                    amount, max_cycle_len, len(best_factors), len(l_missing_factors)
                ))

    print()
    print("l_missing_factors: {}".format(l_missing_factors))
    print("len(l_missing_factors): {}".format(len(l_missing_factors)))
    print("best_factors: {}".format(best_factors))
    print()
    print("n: {}, base: {}".format(n, base))
    print("    max_cycle_len: {}".format(max_cycle_len))
    print("    len(best_factors): {}".format(len(best_factors)))
    assert len(l_missing_factors)==0 # this must be fulfilled!

"""
some values:
for the canonical array with the form:
   [[1,0,...,0], [0,0,...,0], ..., [0,0,...0]]

const n = 1
n: 1, base: 1
    max_cycle_len: 1
    len(best_factors): 1
n: 1, base: 2
    max_cycle_len: 1
    len(best_factors): 32
n: 1, base: 3
    max_cycle_len: 2
    len(best_factors): 81
n: 1, base: 4
    max_cycle_len: 2
    len(best_factors): 256
n: 1, base: 5
    max_cycle_len: 4
    len(best_factors): 1250


const n = 2
n: 2, base: 1
    max_cycle_len: 1
    len(best_factors): 1
n: 2, base: 2
    max_cycle_len: 2
    len(best_factors): 12
n: 2, base: 3
    max_cycle_len: 2
    len(best_factors): 189
n: 2, base: 4
    max_cycle_len: 4
    len(best_factors): 128
n: 2, base: 5
    max_cycle_len: 4
    len(best_factors): 2650
n: 2, base: 6
    max_cycle_len: 2
    len(best_factors): 6696

const n = 3
n: 3, base: 2
    max_cycle_len: 3
    len(best_factors): 24
n: 3, base: 3
    max_cycle_len: 6
    len(best_factors): 80
n: 3, base: 4
    max_cycle_len: 6
    len(best_factors): 752
n: 3, base: 5
    max_cycle_len: 24
    len(best_factors): 2720

const base = 2
conjecture: sequence A204983
n: 1, base: 2
    max_cycle_len: 1
    len(best_factors): 32
n: 2, base: 2
    max_cycle_len: 2
    len(best_factors): 12
n: 3, base: 2
    max_cycle_len: 3
    len(best_factors): 24
n: 4, base: 2
    max_cycle_len: 4
    len(best_factors): 12
n: 5, base: 2
    max_cycle_len: 15
    len(best_factors): 20
n: 6, base: 2
    max_cycle_len: 6
    len(best_factors): 24
n: 7, base: 2
    max_cycle_len: 7
    len(best_factors): 30
n: 8, base: 2
    max_cycle_len: 8
    len(best_factors): 12
n: 9, base: 2
    max_cycle_len: 63
    len(best_factors): 20
n: 10, base: 2
    max_cycle_len: 30
    len(best_factors): 20
n: 11, base: 2
    max_cycle_len: 1023
    len(best_factors): 12
n: 12, base: 2
    max_cycle_len: 12
    len(best_factors): 24
n: 13, base: 2
    max_cycle_len: 4095
    len(best_factors): 12

const base = 3
n: 2, base: 3
    max_cycle_len: 2
    len(best_factors): 189
n: 3, base: 3
    max_cycle_len: 6
    len(best_factors): 80
n: 4, base: 3
    max_cycle_len: 8
    len(best_factors): 200
n: 5, base: 3
    max_cycle_len: 80
    len(best_factors): 184
n: 6, base: 3
    max_cycle_len: 6
    len(best_factors): 188
n: 7, base: 3
    max_cycle_len: 728
    len(best_factors): 168
n: 8, base: 3
    max_cycle_len: 8
    len(best_factors): 232
n: 9, base: 3
    max_cycle_len: 18
    len(best_factors): 80
n: 10, base: 3
    max_cycle_len: 80
    len(best_factors): 200
n: 11, base: 3
    max_cycle_len: 242
    len(best_factors): 217
n: 12, base: 3
    max_cycle_len: 24
    len(best_factors): 200
n: 13, base: 3
    max_cycle_len: 26
    len(best_factors): 232
n: 14, base: 3
    max_cycle_len: 728
    len(best_factors): 168
n: 15, base: 3
    max_cycle_len: 240
    len(best_factors): 184

const base = 4
n: 1, base: 4
    max_cycle_len: 2
    len(best_factors): 256
n: 2, base: 4
    max_cycle_len: 4
    len(best_factors): 128
n: 3, base: 4
    max_cycle_len: 6
    len(best_factors): 752
n: 4, base: 4
    max_cycle_len: 8
    len(best_factors): 128
n: 5, base: 4
    max_cycle_len: 30
    len(best_factors): 640

"""
