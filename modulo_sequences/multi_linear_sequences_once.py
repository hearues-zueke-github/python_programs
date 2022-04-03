#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9 -i

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
    argv = sys.argv

    dim = int(argv[1])
    n = int(argv[2])
    m = int(argv[3])

    assert dim == 1 and n == 3

    # n = int(argv[1])
    # m = int(argv[2])

    # dim = 1
    # n = 1
    # m = 1

    d_found = {
        'max_cycle_length': 0,
        'l_tpl_a_cycle': [],
        'tpl_k': [],
    }
    max_cycle_length = 0
    s_cycle_length = set()
    print_line = False
    for iters in range(0, 500000):
        arr_a = np.random.randint(0, m, (n, ))
        # arr_k = np.random.randint(0, m, (n+1, ))

        arr_k = np.random.randint(0, m, (dim+1, )*n)
        arr_k[0, 1, 1] = 0
        arr_k[1, 0, 0] = 0
        arr_k[1, 0, 1] = 0
        arr_k[1, 1, 1] = 0

        l_tpl_a = [tuple(arr_a.tolist())]
        s_tpl_a = set(l_tpl_a)

        arr_pow = np.arange(0, dim+1)
        while True:
            arr_a_prep = arr_a[0]**arr_pow % m
            for i, a in enumerate(arr_a[1:], 1):
                arr_a_prep = (arr_a_prep * (a**arr_pow).reshape((dim+1, )+(1, )*i)) % m

            a_next = np.sum(arr_k * arr_a_prep) % m
            
            # a_next = (np.sum(arr_k[:-1] * arr_a) + arr_k[-1]) % m

            arr_a[:-1] = arr_a[1:]
            arr_a[-1] = a_next

            tpl_a = tuple(arr_a.tolist())
            if tpl_a in s_tpl_a:
                break

            l_tpl_a.append(tpl_a)
            s_tpl_a.add(tpl_a)

        idx_tpl_a = l_tpl_a.index(tpl_a)
        cycle_length = len(l_tpl_a) - idx_tpl_a

        if cycle_length not in s_cycle_length:
            s_cycle_length.add(cycle_length)

        if max_cycle_length < cycle_length:
            max_cycle_length = cycle_length
            d_found['max_cycle_length'] = max_cycle_length
            d_found['l_tpl_a_cycle'] = l_tpl_a[idx_tpl_a:]
            d_found['tpl_k'] = tuple(arr_k.tolist())
            print_line = True
        
        if iters % 10000 == 0 or print_line:
            # print(f'dim: {dim}, n: {n}, m: {m}, len(s_cycle_length): {len(s_cycle_length)}, max_cycle_length: {max_cycle_length}, iters: {iters}, cycle_length: {cycle_length}')
            print(f'n: {n}, m: {m}, len(s_cycle_length): {len(s_cycle_length)}, m**n: {m**n}, max_cycle_length: {max_cycle_length}, iters: {iters}, cycle_length: {cycle_length}')
            print_line = False

    print(f'n: {n}, m: {m}, len(s_cycle_length): {len(s_cycle_length)}, m**n: {m**n}, max_cycle_length: {max_cycle_length}')
    # print(f'dim: {dim}, n: {n}, m: {m}, len(s_cycle_length): {len(s_cycle_length)}, max_cycle_length: {max_cycle_length}')
    print(f'sorted(s_cycle_length): {sorted(s_cycle_length)}')

    print(f'm**n == max_cycle_length: {m**n == max_cycle_length}')

    # working patterns for:
    # dim = 1, n = 3, m: {1: 1, 2: 8, 7: 343}
