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

def convert_num_to_base_num(n, b, min_len=-1):
    def gen(n):
        while n > 0:
            yield n%b; n //= b
    l = [i for i in gen(n)]
    return l if min_len == -1 else l+[0]*(min_len - len(l) if min_len > len(l) else 0)

def get_num_from_base_lst(l, b):
    n = 0
    mult = 1
    for i, v in enumerate(l, 0):
        n += v*mult
        mult *= b
    return n

if __name__ == '__main__':
    # print('Hello World!')
    argv = sys.argv
    n = int(argv[1])
    m = int(argv[2])

    MAX_CYCLE_LEN = m**n

    def get_missing_tpl_a(d_tpl_a):
        s_all = set(range(0, MAX_CYCLE_LEN))
        for k in d_tpl_a:
            s_all.remove(get_num_from_base_lst(l=k, b=m))
        missing_n = list(s_all)[0]
        # print("- missing_n: {}".format(missing_n))
        l_a_missing = convert_num_to_base_num(n=missing_n, b=m, min_len=n)
        tpl_a_missing = tuple(l_a_missing)
        assert tpl_a_missing not in d_tpl_a
        return tpl_a_missing

    # s_cycle_len = set()
    d_cycle_len = {}
    for iters in range(0, 100000):
        if iters % 1000 == 0:
            print("iters: {}".format(iters))

        arr_a = np.random.randint(0, m, (n, ))
        arr_v_k = np.random.randint(0, m, (n, ))
        arr_m_k = np.random.randint(0, m, (n, n))

        l_a = arr_a.tolist()

        tpl_a = tuple(arr_a.tolist())
        tpl_a_prev = tpl_a
        d_tpl_a = {tpl_a: 0}
        nr_tpl_a = 1
        while True:
            arr_a[:] = (arr_v_k + arr_m_k.dot(arr_a)) % m
            tpl_a_prev = tpl_a
            tpl_a = tuple(arr_a.tolist())
            
            if tpl_a in d_tpl_a:
                break

            d_tpl_a[tpl_a] = nr_tpl_a
            nr_tpl_a += 1
        
        cycle_len = d_tpl_a[tpl_a_prev] - d_tpl_a[tpl_a] + 1
        if cycle_len not in d_cycle_len:
            d_cycle_len[cycle_len] = {
                'l_v_k': arr_v_k.tolist(),
                'l_m_k': arr_m_k.tolist(),
                'l_a': l_a,
                'd_tpl_a': d_tpl_a,
                'tpl_a': tpl_a,
                'tpl_a_prev': tpl_a_prev,
            }
        elif cycle_len == MAX_CYCLE_LEN - 1:
            d = d_cycle_len[cycle_len]
            if 'missing_tpl_a' not in d:
                d['missing_tpl_a'] = get_missing_tpl_a(d_tpl_a=d['d_tpl_a'])
                # print("d['missing_tpl_a']: {}".format(d['missing_tpl_a']))
            else:
                missing_tpl_a = get_missing_tpl_a(d_tpl_a=d_tpl_a)
                # print("maybe? missing_tpl_a: {}".format(missing_tpl_a))
                l11 = list(reversed(d['missing_tpl_a']))
                l12 = list(reversed(missing_tpl_a))

                l21 = d['l_v_k']
                l22 = arr_v_k.tolist()
                if l11 > l12 or l11 == l12 and \
                (l21 > l22 or l21 == l22 and sorted(d['l_m_k']) > sorted(arr_m_k.tolist())):
                    d['l_v_k'] = arr_v_k.tolist()
                    d['l_m_k'] = arr_m_k.tolist()
                    d['l_a'] = l_a
                    d['d_tpl_a'] = d_tpl_a
                    d['tpl_a'] = tpl_a
                    d['tpl_a_prev'] = tpl_a_prev
                    d['missing_tpl_a'] = missing_tpl_a
                    # print("d['missing_tpl_a']: {}".format(d['missing_tpl_a']))
        # d_cycle_len[cycle_len] += 1

    l_cycle_len = sorted(d_cycle_len.keys())
    max_cycle_len = l_cycle_len[-1]
    print(f"n: {n}, m: {m}, l_cycle_len: {l_cycle_len}")

    print(f"d_cycle_len[{max_cycle_len}]: {d_cycle_len[max_cycle_len]}")
