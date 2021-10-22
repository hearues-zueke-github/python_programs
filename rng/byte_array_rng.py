#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

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
import multiprocessing as mp

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append('..')
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PATH_ROOT_DIR, "../utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PATH_ROOT_DIR, "../utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
    n = 10
    arr = np.zeros((n, ), dtype=np.uint8)
    print("arr:\n{}".format(arr))
    
    # bits: 0 -> +, 1 -> ^
    arr_range = np.arange(0, n)
    arr_range_2d = np.concatenate([np.roll(arr_range, -i-1) for i in range(0, 8)]).reshape((8, n))
    print("arr_range_2d:\n{}".format(arr_range_2d))

    # TODO: create a generator for generating a bit stream of the arr array! later...
    def calc_next_hash(arr, iterations=100):
        arr = arr.copy()
        n = arr.shape[0]

        arr_range = np.arange(0, n)
        arr_range_2d = np.concatenate([np.roll(arr_range, -i-1) for i in range(0, 8)]).reshape((8, n))
        # print("arr_range_2d:\n{}".format(arr_range_2d))

        # copy the current state
        arr_temp = arr.copy()

        for i_round in range(1, iterations+1):
            for idx in range(0, n):
                num = arr[idx]
                for idx_bit in range(0, 8): # can be other numbers too!
                    bit = (num >> (7 - idx_bit)) & 0x1
                    idx_next = arr_range_2d[idx_bit, idx]
                    arr_temp[idx] += idx_next
                    if bit == 0:
                         arr_temp[idx] = arr_temp[idx] + arr[idx_next]
                    elif bit == 1:
                         arr_temp[idx] = arr_temp[idx] ^ arr[idx_next]
                    else:
                        assert False

            arr_xor = arr ^ arr_temp
            arr[:] = arr_temp

            # print("i_round: {:5}, arr: {}, arr_xor: {}".format(
            #     i_round,
            #     '0x'+''.join(['{:02X}'.format(v) for v in arr]),
            #     '0x'+''.join(['{:02X}'.format(v) for v in arr_xor]),
            # ))

        return arr

    arr_1 = calc_next_hash(arr=arr, iterations=100)
    arr_2_1 = calc_next_hash(arr=arr, iterations=99)
    arr_2_2 = calc_next_hash(arr=arr_2_1, iterations=1)

    assert np.all(np.equal(arr_1, arr_2_2))

    arr[0] = 1
    arr_3 = calc_next_hash(arr=arr, iterations=100)

    arr[:] = 0
    arr_a1 = calc_next_hash(arr=arr, iterations=1)
    arr[1] = 1
    arr_a2 = calc_next_hash(arr=arr, iterations=1)
