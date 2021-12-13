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

    PKL_GZ_DIR = os.path.join(TEMP_DIR, 'objs/multi_linear_sequences')
    mkdirs(PKL_GZ_DIR)

    l_amount_full_cycle = []
    l_amount_tk_1_cycle = []

    # n = 1
    n = 2
    # n = 3

    # TODO: need to make this multiprocessing able!
    # for m in range(1, 16):
    # for m in range(1, 5):
    for m in range(5, 6):
    # for m in range(1, 101):


        # method 2
        print('method 2')
        arr_a = get_all_combinations_repeat(n=n, m=m).astype(dtype=np.uint32).T
        arr_k = get_all_combinations_repeat(n=2**n, m=m).astype(dtype=np.uint32)
        arr_mult_a = m**np.arange(arr_a.shape[0]-1, -1, -1, dtype=np.uint32)[::-1]
        arr_mult_k = m**np.arange(arr_k.shape[1]-1, -1, -1, dtype=np.uint32)[::-1]

        d_t_to_i = {tuple(a.tolist()): i for i, a in enumerate(arr_a.T, 0)}
        d_i_to_t = {v: k for k, v in d_t_to_i.items()}

        arr_a_idx = np.sum(arr_a.T*arr_mult_a, axis=1, dtype=np.uint32)
        arr_k_idx = np.sum(arr_k*arr_mult_k, axis=1, dtype=np.uint32)

        arr_temp = np.empty(arr_k.shape, dtype=np.uint32)
        arr_sum = np.empty((arr_k.shape[0], ), dtype=np.uint32)

        if n == 1:
            arr_a_prep = np.vstack((
                arr_a[0],
                np.ones((arr_a.shape[1], ), dtype=np.uint32),
            )).T
        elif n == 2:
            arr_a_prep = np.vstack((
                arr_a[0]*arr_a[1],
                arr_a[0],
                arr_a[1],
                np.ones((arr_a.shape[1], ), dtype=np.uint32),
            )).T
        elif n == 3:
            arr_a_prep = np.vstack((
                arr_a[0]*arr_a[1]*arr_a[2],
                arr_a[0]*arr_a[1],
                arr_a[0]*arr_a[2],
                arr_a[1]*arr_a[2],
                arr_a[0],
                arr_a[1],
                arr_a[2],
                np.ones((arr_a.shape[1], ), dtype=np.uint32),
            )).T

        arr_a_prep %= m

        arr_k_a_all = np.einsum('ij,kj->ki', arr_a_prep, arr_k) % m

        d_k_idx_to_d_a_idx_to_idx_next = {}
        d_a_to_l_cycles = {}
        l_tk_empty_cycles = []
        d_len_l_cycles_to_l_k_idx = {}
        for k_idx, arr_k_a in zip(arr_k_idx, arr_k_a_all):
            edges_directed = list(zip(arr_a_idx, np.sum(np.vstack((arr_a[1:], arr_k_a)).T*arr_mult_a, axis=1)))
            d_k_idx_to_d_a_idx_to_idx_next[k_idx] = dict(edges_directed)

            l_cycles = get_cycles_of_1_directed_graph(edges_directed)

            d_a_to_l_cycles[k_idx] = l_cycles

            len_l_cycles = len(l_cycles)

            if len_l_cycles == 1:
                l_tk_empty_cycles.append(k_idx)

            if len_l_cycles not in d_len_l_cycles_to_l_k_idx:
                d_len_l_cycles_to_l_k_idx[len_l_cycles] = []
            d_len_l_cycles_to_l_k_idx[len_l_cycles].append(k_idx)

            # if k_idx == 60:
            #     break


        # # method 1
        # print('method 1')
        # arr_a = get_all_combinations_repeat(n=n, m=m).astype(dtype=np.uint32)
        # arr_k = get_all_combinations_repeat(n=2**n, m=m).astype(dtype=np.uint32)
        # d_t_to_i = {tuple(a.tolist()): i for i, a in enumerate(arr_a, 0)}
        # d_i_to_t = {v: k for k, v in d_t_to_i.items()}

        # d_a_to_l_cycles = {}
        # l_tk_empty_cycles = []
        
        # # for k1, k2, k3, k4, k5, k6, k7, k8 in arr_k:
        # #     tk = (k1, k2, k3, k4, k5, k6, k7, k8)

        # for k_idx, (k1, k2, k3, k4) in enumerate(arr_k, 0):
        #     tk = (k1, k2, k3, k4)

        # # for k1, k2 in arr_k:
        # #     tk = (k1, k2)
        #     # print("tk: {}".format(tk))

        #     d = {}
        #     # for a1, a2, a3 in arr_a:
        #     #     t1 = (a1, a2, a3)
        #     #     t2 = (a2, a3, (a1*a2*k1 + a1*k2 + a2*k3 + k4) % m)
            
        #     for a1, a2 in arr_a:
        #         t1 = (a1, a2)
        #         t2 = (a2, (a1*a2*k1 + a1*k2 + a2*k3 + k4) % m)
            
        #     # for a1,  in arr_a:
        #     #     t1 = (a1, )
        #     #     t2 = ((a1*k1 + k2) % m, )

        #         d[d_t_to_i[t1]] = d_t_to_i[t2]

        #     edges_directed = list(d.items())
        #     l_cycles = get_cycles_of_1_directed_graph(edges_directed)

        #     d_a_to_l_cycles[tk] = l_cycles

        #     if len(l_cycles) == 1:
        #         l_tk_empty_cycles.append(tk)

        #     # if any([len(l)==m**2-1 for l in l_cycles]):
        #     #     print("k_idx: {}".format(k_idx))
        #     #     break


        l_cycles_all = [l1 for l in d_a_to_l_cycles.values() for l1 in l]
        u, c = np.unique(list(map(len, l_cycles_all)), return_counts=True)

        l_unique_seq_tpl = [(lambda x: tuple(l[x:]+l[:x]))(l.index(min(l))) for l in l_cycles_all]
        amount_unique_tpl = len(set(l_unique_seq_tpl))
        print("n: {}, m: {}, amount_unique_tpl: {}, len(u): {}".format(n, m, amount_unique_tpl, len(u)))

        l_cycles_all_l_num = [d_i_to_t[l[0]]+tuple(d_i_to_t[i][-1] for i in l[1:]) for l in l_cycles_all]

        u_len, c_len = np.unique([len(l) for l in l_cycles_all_l_num], return_counts=True)

        l_amount_full_cycle.append((m, c_len[-1]))
        l_amount_tk_1_cycle.append((m, len(l_tk_empty_cycles)))

    print("n: {}".format(n))
    print("l_amount_full_cycle: {}".format(l_amount_full_cycle))
    print("l_amount_tk_1_cycle: {}".format(l_amount_tk_1_cycle))

    l_a_n = [a for _, a in l_amount_full_cycle]
    print("l_a_n: {}".format(l_a_n))
    l_tk_1_cycle = [a for _, a in l_amount_tk_1_cycle]
    print("l_tk_1_cycle: {}".format(l_tk_1_cycle))

    # for n: 1
    # >>> arr=np.array(l_a_n)
    # >>> np.where(arr==np.arange(1, len(arr)+1))[0]+1
    # array([ 1,  8, 18, 36])
    
    # n: 1, l_a_n: [1, 1, 2, 2, 4, 2, 6, 8, 18, 4, 10, 4, 12, 6, 8]
    # n: 2, l_a_n = [1, 1, 6, 24, 20, 24, 56, 768, 1782, 72]

    # n: 2
    # l_a_n: [1, 1, 6, 24, 20, 24, 56, 768, 1782, 72, 176, 2448, 312, 104, 120, 24576, 816, 3780, 912, 480]
    # l_tk_1_cycle: [1, 8, 29, 88, 129, 180, 433, 1280, 999, 752, 1451, 1816, 4081, 2876, 2181, 20480, 10097, 5832, 14401, 7512]

    # n: 2, m: 1, amount_unique_tpl: 1, len(u): 1
    # n: 2, m: 2, amount_unique_tpl: 6, len(u): 4
    # n: 2, m: 3, amount_unique_tpl: 43, len(u): 6
    # n: 2, m: 4, amount_unique_tpl: 102, len(u): 6
    # n: 2, m: 5, amount_unique_tpl: 535, len(u): 13
    # n: 2, m: 6, amount_unique_tpl: 434, len(u): 8
    # n: 2, m: 7, amount_unique_tpl: 3007, len(u): 21
    # n: 2, m: 8, amount_unique_tpl: 3140, len(u): 8
    # n: 2, m: 9, amount_unique_tpl: 8661, len(u): 10
    # n: 2, m: 10, amount_unique_tpl: 5258, len(u): 22
    # n: 2, m: 11, amount_unique_tpl: 27053, len(u): 33

    # n: 3, m: 1, amount_unique_tpl: 1
    
    # n: 3, m: 2, amount_unique_tpl: 19
    
    # amount_unique_tpl: 1425, n=3, m=3
    
    # len(set(l_unique_seq_tpl)) == 5986, n=3, m=4
    
    # n: 3, m: 5, amount_unique_tpl: 289597

    # n: 3, m: 6, amount_unique_tpl: 55315


    # l_m_l_cycle_len_count = []
    # l_len_l_cycle_len_count = []
    # for m in range(4, 5):
    # # for m in range(1, 11):
    #     arr_combinations = get_all_combinations_repeat(m=m, n=n)
    #     len_arr_combinations = len(arr_combinations)

    #     d_comb_tpl_to_idx = {tuple(arr.tolist()): idx for idx, arr in enumerate(arr_combinations, 0)}
    #     d_idx_to_comb_tpl = {v: k for k, v in d_comb_tpl_to_idx.items()}

    #     # mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count(), is_print_on=True)
    #     mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count(), is_print_on=False)
    #     split_amount = mult_proc_mng.worker_amount
        
    #     # split evenly if possible into pieces
    #     arr_split_idx_diff = np.ones((split_amount, ), dtype=np.int32) * (len_arr_combinations // split_amount)
    #     # print("arr_split_idx_diff: {}".format(arr_split_idx_diff))
    #     arr_split_idx_diff[:len_arr_combinations % split_amount] += 1

    #     arr_split_idx = np.hstack(((0, ), np.cumsum(arr_split_idx_diff)))

    #     # l_arr_comb = [arr_combinations[i1:i2] for i1, i2 in zip(arr_split_idx[:-1], arr_split_idx[1:])]

    #     # # only for testing the responsivness!
    #     # mult_proc_mng.test_worker_threads_response()

    #     arr_x = np.random.randint(0, m, (n, ), dtype=np.uint16)

    #     # # to get the reference for the chared memory! can be useful later for other projects
    #     # shm_arr_k = smm.SharedMemory(size=n * np.uint16().itemsize)
    #     # arr_k = np.ndarray((n, ), dtype=np.uint16, buffer=shm_arr_k.buf)
    #     # arr_k[:] = np.random.randint(0, m, (n, ), dtype=np.uint16)

    #     # print("arr_k: {arr_k}")
    # # 
    #     def get_all_combinations_cycles(tpl_arr_k):
    #         arr_next_comb = np.hstack((arr_combinations[:, 1:], (np.sum(arr_combinations*tpl_arr_k, axis=1) % m).reshape((-1, 1))))

    #         d_directed_graph = {
    #             d_comb_tpl_to_idx[tuple(arr1.tolist())]: d_comb_tpl_to_idx[tuple(arr2.tolist())]
    #             for arr1, arr2 in zip(arr_combinations, arr_next_comb)
    #         }

    #         edges_directed = list(d_directed_graph.items())
    #         l_cycles = get_cycles_of_1_directed_graph(edges_directed)

    #         try:
    #             arr_k_idx = d_comb_tpl_to_idx[tpl_arr_k]
    #         except:
    #             arr_k_idx = None

    #         return {'arr_k_idx': arr_k_idx, 'l_cycles': l_cycles}
    #         # return {'arr_k_idx': arr_k_idx, 'd_directed_graph': d_directed_graph, 'l_cycles': l_cycles}

    #     mult_proc_mng.define_new_func('func_get_all_combinations_cycles', get_all_combinations_cycles)

    #     l_arguments = [
    #         # ((1, ) * n, )
    #         (tuple(arr_k.tolist()), ) for arr_k in arr_combinations
    #     ]
    #     l_ret_l_data = mult_proc_mng.do_new_jobs(
    #         ['func_get_all_combinations_cycles'] * len(l_arguments),
    #         l_arguments,
    #     )
    #     del mult_proc_mng

    #     l_cycles_all_orig = [tuple(cycle) for d in l_ret_l_data for cycle in d['l_cycles']]
    #     l_cycles_all_shift = [(lambda i: t[i:]+t[:i])(t.index(min(t))) for t in l_cycles_all_orig]

    #     d_cycle_count = defaultdict(int)
    #     for cycle in l_cycles_all_shift:
    #         d_cycle_count[cycle] += 1

    #     d_cycle_count_count = defaultdict(int)
    #     for cycle_count in d_cycle_count.values():
    #         d_cycle_count_count[cycle_count] += 1

    #     d_cycle_len_count = defaultdict(int)
    #     for cycle in d_cycle_count.keys():
    #         d_cycle_len_count[len(cycle)] += 1

    #     l_cycle_len_count = sorted(d_cycle_len_count.items())
    #     print(f'n: {n}, m: {m}, l_cycle_len_count: {l_cycle_len_count}')

    #     len_l_cycle_len_count = len(l_cycle_len_count)

    #     l_m_l_cycle_len_count.append((m, l_cycle_len_count))
    #     l_len_l_cycle_len_count.append(len_l_cycle_len_count)

    # l_max_cycles_for_m = [l[-1][-1][0] for l in l_m_l_cycle_len_count]

    # print()
    # print(f'n = {n}, max_m: {m}\n- l_max_cycles_for_m = {l_max_cycles_for_m}\n- l_len_l_cycle_len_count: {l_len_l_cycle_len_count}')

    # smm.shutdown()

