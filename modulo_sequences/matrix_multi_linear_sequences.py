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
    argv = sys.argv
    
    # m = int(argv[1])
    # l_cycle_len = []
    l_key_len = []
    for m in range(1, 5):
    # for m in range(1, 21):
    # for m in range(21, 23):
        MAX_CYCLE_LEN = m

        d_t_d_k_to_l_seq = {}
        d_cycle_len = {}
        s_t_d_k_vals = set()
        for k in range(0, m):
            for d in range(0, m):
                a = 0
                a_prev = a
                d_a = {a: 0}
                l_a = [a]
                nr_tpl_a = 1
                while True:
                    a_next = (d + k * a) % m
                    a_prev = a
                    a = a_next

                    if a in d_a:
                        break

                    d_a[a] = nr_tpl_a
                    nr_tpl_a += 1
                    l_a.append(a)
                
                cycle_len = d_a[a_prev] - d_a[a] + 1

                if cycle_len == MAX_CYCLE_LEN:
                    t_d_k = (d, k)
                    if t_d_k not in s_t_d_k_vals:
                        s_t_d_k_vals.add(t_d_k)
                    d_t_d_k_to_l_seq[t_d_k] = l_a

        l_t_d_k_vals = sorted(s_t_d_k_vals)
        print(f"l_t_d_k_vals: {l_t_d_k_vals}")

        t_d_k = l_t_d_k_vals[0]
        # t_d_k = l_t_d_k_vals[len(l_t_d_k_vals) % 4]
        l_a = d_t_d_k_to_l_seq[t_d_k]
        d_a_next = {a1: a2 for a1, a2 in zip(l_a, l_a[1:]+l_a[:1])}

        print(f"t_d_k: {t_d_k}")
        print(f"- l_a: {l_a}")

        # l_a_len = []

        # d_n_to_d_cycle_len_to_l_t_unique = {}
        d_cycle_len_to_l_t_unique = {}

        # n = m
        for n in range(m**2, m**2+1):
        # for n in range(1, 19):
            # d_cycle_len_to_l_t_unique = {}
            # d_n_to_d_cycle_len_to_l_t_unique[n] = d_cycle_len_to_l_t_unique

        # for n in range(1, 10):
            s_t_avail = set(tuple(l) for l in get_all_combinations_repeat(2, n).tolist())
            # s_t_used = set()
            s_t_unique = set()

            while s_t_avail:
                t = s_t_avail.pop()
                t_inv = tuple((i+1)%2 for i in t)
                t_rev = t[::-1]
                t_rev_inv = t_inv[::-1]

                l_t = [t, t_inv, t_rev, t_rev_inv]
                for _ in range(0, n-1):
                    t = t[1:] + t[:1]
                    t_inv = t_inv[1:] + t_inv[:1]
                    t_rev = t_rev[1:] + t_rev[:1]
                    t_rev_inv = t_rev_inv[1:] + t_rev_inv[:1]
                    l_t.append(t)
                    l_t.append(t_inv)
                    l_t.append(t_rev)
                    l_t.append(t_rev_inv)

                for t in l_t:
                    if t in s_t_avail:
                        s_t_avail.remove(t)

                s_t_unique.add(sorted(l_t)[0])

            print(f"- n: {n}, s_t_unique: {s_t_unique}")
            
            # l_mult_factor = [i for i in range(1, n) if n % i == 0]
            # for t_unique in sorted(s_t_unique):
            #     is_inner_cycle = False
            #     for mult_factor in l_mult_factor:
            #         t_first = t_unique[:mult_factor]
            #         if all([t_first == t_unique[mult_factor*i:mult_factor*(i+1)] for i in range(1, n // mult_factor)]):
            #             is_inner_cycle = True
            #             break
            #     if is_inner_cycle:
            #         s_t_unique.remove(t_unique)

            # print(f"- n: {n}, removed some inner_cycles s_t_unique: {s_t_unique}")

            # l_a_len.append(len(s_t_unique))
            # continue
            
            # print(f"l_a_len: {l_a_len}")

            # sys.exit()

            # l_l_idx = [
            #     [0, 0, 1],
            #     [0, 1, 0],
            #     [0, 1, 1],
            #     [0, 0, 0, 1],
            #     [0, 0, 1, 1],
            #     [0, 1],
            #     [0, 1, 0, 1],
            #     [0, 0, 1, 0, 1, 1],
            #     [0, 0, 0, 0, 1],
            #     [0, 0, 0, 0, 1, 1],
            #     [0, 0, 0, 0, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 1, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 0, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 1],
            #     [0, 0, 0, 0, 0, 0, 1, 0, 1],
            #     [0, 0, 0, 0, 0, 1, 1, 0, 1],
            #     [0, 0, 0, 0, 1, 1, 0, 0, 1],
            #     [0, 0, 0, 0, 0, 1, 0, 1, 1],
            #     [0, 0, 0, 0, 1, 0, 1, 0, 1],
            #     [0, 0, 0, 0, 0, 1, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 0, 1],
            #     [0, 0, 0, 0, 0, 0, 1, 1],
            #     [0, 0, 0, 0, 0, 1, 1, 1],
            #     [0, 0, 0, 0, 1, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            # ]
            for t_idx in s_t_unique:
            # for t_idx in l_l_idx:
                la = [0, 0]
                # t_idx = [0, 0, 1]
                len_l_idx = len(t_idx)
                idx_index = 0

                ta = tuple(la)
                d_ta_to_nr_tpl_a = {ta: 0}
                la_prev = [0, 0]
                d_ta = {ta: 0}
                l_ta = [ta]
                nr_tpl_a = 1

                while True:
                    la_prev[0] = la[0]
                    la_prev[1] = la[1]

                    idx = t_idx[idx_index]
                    idx_index = (idx_index + 1) % len_l_idx

                    a_next = d_a_next[la[idx]]
                    # la_prev[idx] = la[idx]
                    la[idx] = a_next

                    ta = tuple(la)
                    if ta in d_ta:
                        break

                    d_ta[ta] = nr_tpl_a
                    nr_tpl_a += 1
                    l_ta.append(ta)
                
                cycle_len = d_ta[tuple(la_prev)] - d_ta[ta] + 1
                print()
                print(f"t_idx: {t_idx}")
                print(f"- cycle_len: {cycle_len}")
                # print(f"- l_ta: {l_ta}")

                if cycle_len not in d_cycle_len_to_l_t_unique:
                    d_cycle_len_to_l_t_unique[cycle_len] = []

                d_cycle_len_to_l_t_unique[cycle_len].append(t_idx)

        # if cycle_len == MAX_CYCLE_LEN:
        #     t_d_k = (d, k)
        #     if t_d_k not in s_t_d_k_vals:
        #         s_t_d_k_vals.add(t_d_k)
        #     d_t_d_k_to_l_seq[t_d_k] = l_a

        print(f"m: {m}, d_cycle_len_to_l_t_unique.keys(): {d_cycle_len_to_l_t_unique.keys()}")
        l_key_len.append(len(d_cycle_len_to_l_t_unique))

    print(f"l_key_len: {l_key_len}")
