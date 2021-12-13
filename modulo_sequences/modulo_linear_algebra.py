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

import matplotlib.pyplot as plt

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

from multiprocessing.managers import SharedMemoryManager

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
load_module_dynamically(**dict(var_glob=var_glob, name='utils_graph_theory', path=os.path.join(PYTHON_PROGRAMS_DIR, "graph_theory/utils_graph_theory.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat
get_cycles_of_1_directed_graph = utils_graph_theory.get_cycles_of_1_directed_graph

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
    # smm = SharedMemoryManager()
    # smm.start()

    PKL_GZ_DIR = os.path.join(TEMP_DIR, 'objs/modulo_linear_algebra')
    mkdirs(PKL_GZ_DIR)

    n = 3

    l_m_l_cycle_len_count = []
    l_len_l_cycle_len_count = []
    for m in range(4, 5):
    # for m in range(1, 11):
        arr_combinations = get_all_combinations_repeat(m=m, n=n)
        len_arr_combinations = len(arr_combinations)

        d_comb_tpl_to_idx = {tuple(arr.tolist()): idx for idx, arr in enumerate(arr_combinations, 0)}
        d_idx_to_comb_tpl = {v: k for k, v in d_comb_tpl_to_idx.items()}

        # mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count(), is_print_on=True)
        mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count(), is_print_on=False)
        split_amount = mult_proc_mng.worker_amount
        
        # split evenly if possible into pieces
        arr_split_idx_diff = np.ones((split_amount, ), dtype=np.int32) * (len_arr_combinations // split_amount)
        # print("arr_split_idx_diff: {}".format(arr_split_idx_diff))
        arr_split_idx_diff[:len_arr_combinations % split_amount] += 1

        arr_split_idx = np.hstack(((0, ), np.cumsum(arr_split_idx_diff)))

        # l_arr_comb = [arr_combinations[i1:i2] for i1, i2 in zip(arr_split_idx[:-1], arr_split_idx[1:])]

        # # only for testing the responsivness!
        # mult_proc_mng.test_worker_threads_response()

        arr_x = np.random.randint(0, m, (n, ), dtype=np.uint16)

        # # to get the reference for the chared memory! can be useful later for other projects
        # shm_arr_k = smm.SharedMemory(size=n * np.uint16().itemsize)
        # arr_k = np.ndarray((n, ), dtype=np.uint16, buffer=shm_arr_k.buf)
        # arr_k[:] = np.random.randint(0, m, (n, ), dtype=np.uint16)

        # print("arr_k: {arr_k}")
    # 
        def get_all_combinations_cycles(tpl_arr_k):
            arr_next_comb = np.hstack((arr_combinations[:, 1:], (np.sum(arr_combinations*tpl_arr_k, axis=1) % m).reshape((-1, 1))))

            d_directed_graph = {
                d_comb_tpl_to_idx[tuple(arr1.tolist())]: d_comb_tpl_to_idx[tuple(arr2.tolist())]
                for arr1, arr2 in zip(arr_combinations, arr_next_comb)
            }

            edges_directed = list(d_directed_graph.items())
            l_cycles = get_cycles_of_1_directed_graph(edges_directed)

            try:
                arr_k_idx = d_comb_tpl_to_idx[tpl_arr_k]
            except:
                arr_k_idx = None

            return {'arr_k_idx': arr_k_idx, 'l_cycles': l_cycles}
            # return {'arr_k_idx': arr_k_idx, 'd_directed_graph': d_directed_graph, 'l_cycles': l_cycles}

        mult_proc_mng.define_new_func('func_get_all_combinations_cycles', get_all_combinations_cycles)

        l_arguments = [
            # ((1, ) * n, )
            (tuple(arr_k.tolist()), ) for arr_k in arr_combinations
        ]
        l_ret_l_data = mult_proc_mng.do_new_jobs(
            ['func_get_all_combinations_cycles'] * len(l_arguments),
            l_arguments,
        )
        del mult_proc_mng

        l_cycles_all_orig = [tuple(cycle) for d in l_ret_l_data for cycle in d['l_cycles']]
        l_cycles_all_shift = [(lambda i: t[i:]+t[:i])(t.index(min(t))) for t in l_cycles_all_orig]

        d_cycle_count = defaultdict(int)
        for cycle in l_cycles_all_shift:
            d_cycle_count[cycle] += 1

        d_cycle_count_count = defaultdict(int)
        for cycle_count in d_cycle_count.values():
            d_cycle_count_count[cycle_count] += 1

        d_cycle_len_count = defaultdict(int)
        for cycle in d_cycle_count.keys():
            d_cycle_len_count[len(cycle)] += 1

        l_cycle_len_count = sorted(d_cycle_len_count.items())
        print(f'n: {n}, m: {m}, l_cycle_len_count: {l_cycle_len_count}')

        len_l_cycle_len_count = len(l_cycle_len_count)

        l_m_l_cycle_len_count.append((m, l_cycle_len_count))
        l_len_l_cycle_len_count.append(len_l_cycle_len_count)

    l_max_cycles_for_m = [l[-1][-1][0] for l in l_m_l_cycle_len_count]

    print()
    print(f'n = {n}, max_m: {m}\n- l_max_cycles_for_m = {l_max_cycles_for_m}\n- l_len_l_cycle_len_count: {l_len_l_cycle_len_count}')

    # smm.shutdown()

'''
were arr_k is any combination
n = 1, max_m: 20,
# OEIS A002322
- l_max_cycles_for_m = [1, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4, 4, 16, 6, 18, 4]
# OEIS A066800
- l_len_l_cycle_len_count: [1, 1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 4, 3, 3, 5, 4, 6, 3]

n = 2, max_m: 20
# OEIS A316565
- l_max_cycles_for_m = [1, 3, 8, 6, 24, 24, 48, 12, 24, 60, 120, 24, 168, 48, 60, 24, 288, 24, 360, 60]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 3, 6, 5, 11, 8, 14, 7, 10, 14, 20, 8, 22, 14, 15, 9, 23, 10, 30, 14]

n = 3, max_m: 15
# OEIS not found!
- l_max_cycles_for_m = [1, 7, 26, 14, 124, 182, 342, 28, 78, 868, 1330, 182, 2196, 2394, 1612]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 5, 8, 8, 14, 21, 22, 11, 14, 34, 32, 22, 34, 40, 36]

n = 4, max_m: 10
# OEIS not found!
- l_max_cycles_for_m = [1, 15, 80, 30, 624, 560, 2400, 60, 240, 4368]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 8, 18, 13, 31, 52, 54, 18, 30, 83]

n = 5, max_m: 7
# OEIS not found!
- l_max_cycles_for_m = [1, 31, 242, 62, 3124, 7502, 16806]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 13, 26, 20, 48, 128, 78]

n = 6, max_m: 5
# OEIS not found!
- l_max_cycles_for_m = [1, 63, 728, 126, 15624]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 18, 42, 27, 95]


were arr_k is all 1
n = 1, max_m: 30
# OEIS A000012
- l_max_cycles_for_m = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# OEIS A000012
- l_len_l_cycle_len_count: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

n = 2, max_m: 30
# OEIS A001175
- l_max_cycles_for_m = [1, 3, 8, 6, 20, 24, 16, 12, 24, 60, 10, 24, 28, 48, 40, 24, 36, 24, 18, 60, 16, 30, 48, 24, 100, 84, 72, 48, 14, 120]
# OEIS A015135
- l_len_l_cycle_len_count: [1, 2, 2, 3, 3, 4, 2, 4, 3, 6, 3, 5, 2, 4, 5, 5, 2, 4, 3, 7, 3, 6, 2, 6, 4, 4, 4, 5, 3, 10]

n = 3, max_m: 30
# OEIS A046738
- l_max_cycles_for_m = [1, 4, 13, 8, 31, 52, 48, 16, 39, 124, 110, 104, 168, 48, 403, 32, 96, 156, 360, 248, 624, 220, 553, 208, 155, 168, 117, 48, 140, 1612]
# OEIS A106288
- l_len_l_cycle_len_count: [1, 3, 2, 4, 2, 6, 3, 5, 3, 6, 4, 8, 3, 6, 4, 6, 3, 9, 3, 8, 6, 8, 2, 10, 3, 5, 4, 8, 3, 12]

n = 4, max_m: 30
# OEIS A106295
- l_max_cycles_for_m = [1, 5, 26, 10, 312, 130, 342, 20, 78, 1560, 120, 130, 84, 1710, 312, 40, 4912, 390, 6858, 1560, 4446, 120, 12166, 260, 1560, 420, 234, 1710, 280, 1560]
# OEIS A106289
- l_len_l_cycle_len_count: [1, 2, 2, 3, 2, 4, 4, 4, 4, 4, 3, 5, 3, 8, 3, 5, 3, 8, 3, 5, 7, 4, 4, 7, 3, 6, 6, 9, 4, 6]

n = 5, max_m: 20
# OEIS A106303
- l_max_cycles_for_m = [1, 6, 104, 12, 781, 312, 2801, 24, 312, 4686, 16105, 312, 30941, 16806, 81224, 48, 88741, 312, 13032, 9372]
# OEIS A106290
- l_len_l_cycle_len_count: [1, 3, 4, 4, 2, 9, 2, 6, 7, 6, 2, 11, 2, 6, 8, 8, 2, 9, 3, 8]

n = 6, max_m: 15
# OEIS not found!
- l_max_cycles_for_m = [1, 7, 728, 14, 208, 728, 342, 28, 2184, 1456, 354312, 728, 9520, 2394, 1456]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 2, 2, 3, 3, 3, 3, 4, 3, 6, 2, 4, 3, 6, 5]

n = 7, max_m: 11
# OEIS not found!
- l_max_cycles_for_m = [1, 8, 364, 16, 9372, 728, 137257, 32, 1092, 18744, 161050]
# OEIS not found!
- l_len_l_cycle_len_count: [1, 4, 2, 5, 4, 6, 2, 6, 4, 11, 4]
'''
