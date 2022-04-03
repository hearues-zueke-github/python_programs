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
    print('Hello World!')

    l_seq = []
    l_seq_2 = []
    l_seq_3 = []

    for n in range(1, 21):
        # n = 4
        m = 2**n
        edges_directed = [(i, ((i >> 1) + ((i % 2) << (n - 1)) + 1) % m) for i in range(0, m)]

        # edges_directed = list(zip(arr_a_idx, np.sum(np.vstack((arr_a[1:], arr_k_a)).T*arr_mult_a, axis=1)))
        # d = dict(edges_directed)

        l_cycles = get_cycles_of_1_directed_graph(edges_directed)
        l_len = list(map(len, l_cycles))
        max_len = max(l_len)

        print(f'n: {n}, max_len: {max_len}')
        print(f'- l_len: {l_len}')
        l_seq.append(max_len)
        l_seq_2.append(len(l_len))
        l_seq_3.append(len(set(l_len)))

    print(f'l_seq: {l_seq}')
    print(f'l_seq_2: {l_seq_2}')
    print(f'l_seq_3: {l_seq_3}')

    # l_seq: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    
    ## oeis A008965
    # l_seq_2: [1, 2, 3, 5, 7, 13, 19, 35, 59, 107, 187, 351, 631, 1181, 2191, 4115, 7711, 14601, 27595, 52487]

    ## oeis A212356
    # l_seq_3: [1, 2, 3, 4, 3, 5, 3, 5, 4, 5, 3, 7, 3, 5, 5, 6, 3, 7, 3, 7]

    # # smm = SharedMemoryManager()
    # # smm.start()

    # PKL_GZ_DIR = os.path.join(TEMP_DIR, 'objs/multi_linear_sequences')
    # mkdirs(PKL_GZ_DIR)

    # l_amount_full_cycle = []
    # l_amount_tk_1_cycle = []

    # # n = 1
    # n = 2
    # # n = 3

    # # TODO: need to make this multiprocessing able!
    # for m in range(1, 16):
    # # for m in range(1, 5):
    # # for m in range(5, 6):
    # # for m in range(1, 101):


    #     # method 2
    #     print('method 2')
    #     arr_a = get_all_combinations_repeat(n=n, m=m).astype(dtype=np.uint32).T
    #     arr_k = get_all_combinations_repeat(n=2**n, m=m).astype(dtype=np.uint32)
    #     arr_mult_a = m**np.arange(arr_a.shape[0]-1, -1, -1, dtype=np.uint32)[::-1]
    #     arr_mult_k = m**np.arange(arr_k.shape[1]-1, -1, -1, dtype=np.uint32)[::-1]

    #     d_t_to_i = {tuple(a.tolist()): i for i, a in enumerate(arr_a.T, 0)}
    #     d_i_to_t = {v: k for k, v in d_t_to_i.items()}

    #     arr_a_idx = np.sum(arr_a.T*arr_mult_a, axis=1, dtype=np.uint32)
    #     arr_k_idx = np.sum(arr_k*arr_mult_k, axis=1, dtype=np.uint32)

    #     arr_temp = np.empty(arr_k.shape, dtype=np.uint32)
    #     arr_sum = np.empty((arr_k.shape[0], ), dtype=np.uint32)

    #     if n == 1:
    #         arr_a_prep = np.vstack((
    #             arr_a[0],
    #             np.ones((arr_a.shape[1], ), dtype=np.uint32),
    #         )).T
    #     elif n == 2:
    #         arr_a_prep = np.vstack((
    #             arr_a[0]*arr_a[1],
    #             arr_a[0],
    #             arr_a[1],
    #             np.ones((arr_a.shape[1], ), dtype=np.uint32),
    #         )).T
    #     elif n == 3:
    #         arr_a_prep = np.vstack((
    #             arr_a[0]*arr_a[1]*arr_a[2],
    #             arr_a[0]*arr_a[1],
    #             arr_a[0]*arr_a[2],
    #             arr_a[1]*arr_a[2],
    #             arr_a[0],
    #             arr_a[1],
    #             arr_a[2],
    #             np.ones((arr_a.shape[1], ), dtype=np.uint32),
    #         )).T

    #     arr_a_prep %= m

    #     arr_k_a_all = np.einsum('ij,kj->ki', arr_a_prep, arr_k) % m

    #     d_k_idx_to_d_a_idx_to_idx_next = {}
    #     d_a_to_l_cycles = {}
    #     l_tk_empty_cycles = []
    #     d_len_l_cycles_to_l_k_idx = {}
    #     for k_idx, arr_k_a in zip(arr_k_idx, arr_k_a_all):
    #         edges_directed = list(zip(arr_a_idx, np.sum(np.vstack((arr_a[1:], arr_k_a)).T*arr_mult_a, axis=1)))
    #         d_k_idx_to_d_a_idx_to_idx_next[k_idx] = dict(edges_directed)

    #         l_cycles = get_cycles_of_1_directed_graph(edges_directed)

    #         d_a_to_l_cycles[k_idx] = l_cycles

    #         len_l_cycles = len(l_cycles)

    #         if len_l_cycles == 1:
    #             l_tk_empty_cycles.append(k_idx)

    #         if len_l_cycles not in d_len_l_cycles_to_l_k_idx:
    #             d_len_l_cycles_to_l_k_idx[len_l_cycles] = []
    #         d_len_l_cycles_to_l_k_idx[len_l_cycles].append(k_idx)

    #         # if k_idx == 60:
    #         #     break


    #     # # method 1
    #     # print('method 1')
    #     # arr_a = get_all_combinations_repeat(n=n, m=m).astype(dtype=np.uint32)
    #     # arr_k = get_all_combinations_repeat(n=2**n, m=m).astype(dtype=np.uint32)
    #     # d_t_to_i = {tuple(a.tolist()): i for i, a in enumerate(arr_a, 0)}
    #     # d_i_to_t = {v: k for k, v in d_t_to_i.items()}

    #     # d_a_to_l_cycles = {}
    #     # l_tk_empty_cycles = []
        
    #     # # for k1, k2, k3, k4, k5, k6, k7, k8 in arr_k:
    #     # #     tk = (k1, k2, k3, k4, k5, k6, k7, k8)

    #     # for k_idx, (k1, k2, k3, k4) in enumerate(arr_k, 0):
    #     #     tk = (k1, k2, k3, k4)

    #     # # for k1, k2 in arr_k:
    #     # #     tk = (k1, k2)
    #     #     # print("tk: {}".format(tk))

    #     #     d = {}
    #     #     # for a1, a2, a3 in arr_a:
    #     #     #     t1 = (a1, a2, a3)
    #     #     #     t2 = (a2, a3, (a1*a2*k1 + a1*k2 + a2*k3 + k4) % m)
            
    #     #     for a1, a2 in arr_a:
    #     #         t1 = (a1, a2)
    #     #         t2 = (a2, (a1*a2*k1 + a1*k2 + a2*k3 + k4) % m)
            
    #     #     # for a1,  in arr_a:
    #     #     #     t1 = (a1, )
    #     #     #     t2 = ((a1*k1 + k2) % m, )

    #     #         d[d_t_to_i[t1]] = d_t_to_i[t2]

    #     #     edges_directed = list(d.items())
    #     #     l_cycles = get_cycles_of_1_directed_graph(edges_directed)

    #     #     d_a_to_l_cycles[tk] = l_cycles

    #     #     if len(l_cycles) == 1:
    #     #         l_tk_empty_cycles.append(tk)

    #     #     # if any([len(l)==m**2-1 for l in l_cycles]):
    #     #     #     print("k_idx: {}".format(k_idx))
    #     #     #     break


    #     l_cycles_all = [l1 for l in d_a_to_l_cycles.values() for l1 in l]
    #     u, c = np.unique(list(map(len, l_cycles_all)), return_counts=True)

    #     l_unique_seq_tpl = [(lambda x: tuple(l[x:]+l[:x]))(l.index(min(l))) for l in l_cycles_all]
    #     amount_unique_tpl = len(set(l_unique_seq_tpl))
    #     print("n: {}, m: {}, amount_unique_tpl: {}, len(u): {}".format(n, m, amount_unique_tpl, len(u)))

    #     l_cycles_all_l_num = [d_i_to_t[l[0]]+tuple(d_i_to_t[i][-1] for i in l[1:]) for l in l_cycles_all]

    #     u_len, c_len = np.unique([len(l) for l in l_cycles_all_l_num], return_counts=True)

    #     l_amount_full_cycle.append((m, c_len[-1]))
    #     l_amount_tk_1_cycle.append((m, len(l_tk_empty_cycles)))

    # print("n: {}".format(n))
    # print("l_amount_full_cycle: {}".format(l_amount_full_cycle))
    # print("l_amount_tk_1_cycle: {}".format(l_amount_tk_1_cycle))

    # l_a_n = [a for _, a in l_amount_full_cycle]
    # print("l_a_n: {}".format(l_a_n))
    # l_tk_1_cycle = [a for _, a in l_amount_tk_1_cycle]
    # print("l_tk_1_cycle: {}".format(l_tk_1_cycle))

