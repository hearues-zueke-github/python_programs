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

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
    PKL_GZ_DIR = os.path.join(TEMP_DIR, 'objs/modulo_power_numbers')
    mkdirs(PKL_GZ_DIR)

    N_MAX = 300
    P_MAX = int(N_MAX*1.5)
    if P_MAX % 100 != 0:
        P_MAX += 100 - P_MAX % 100

    STEPS = 100

    def get_l_data(p_start, p_end, n_start, n_end):
        file_name = f'p_{p_start}_{p_end-1}_n_{n_start}_{n_end-1}.pkl.gz'
        file_path = os.path.join(PKL_GZ_DIR, file_name)
        
        # TODO: create temp obj creation and loading as an util function!
        if not os.path.exists(file_path):
            l_data = []
            for p in range(p_start, p_end):
                for n in range(n_start, n_end):
                    l_seq = [pow(i, p, n) for i in range(1, n)]
                    # l_seq = [i**p % n for i in range(1, n)]
                    l_data.append({'p': p, 'n': n, 'l_seq': l_seq})

            with gzip.open(file_path, 'wb') as f:
                dill.dump(l_data, f)
        else:
            with gzip.open(file_path, 'rb') as f:
                l_data = dill.load(f)
            
        return l_data

    def f(x):
        return x**2

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count()-1)

    # # only for testing the responsivness!
    # mult_proc_mng.test_worker_threads_response()

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_get_l_data', get_l_data)

    l_arguments = [
        (p_start, p_start + STEPS, n_start, n_start + STEPS)
            for p_start in range(1, P_MAX, STEPS)
            for n_start in range(1, N_MAX+1, STEPS)
    ]
    l_ret_l_data = mult_proc_mng.do_new_jobs(
        ['func_get_l_data']*len(l_arguments),
        l_arguments,
    )
    del mult_proc_mng

    d_p_to_d_n_to_l_seq = {p: {} for p in range(1, P_MAX + 1)}

    for l_data in l_ret_l_data:
        for d in l_data:
            p_d = d['p']
            n_d = d['n']
            # assert p_d >= p_start and p_d <= p_end
            # assert n_d >= n_start and n_d <= n_end
            l_seq = d['l_seq']
            d_p_to_d_n_to_l_seq[p_d][n_d] = l_seq

    # for p_start in range(1, P_MAX, STEPS):
    #     print("p_start: {}".format(p_start))
        
    #     p_end = p_start + STEPS
        
    #     for n_start in range(1, N_MAX+1, STEPS):
    #         n_end = n_start + STEPS

    #         file_name = f'p_{p_start}_{p_end-1}_n_{n_start}_{n_end-1}.pkl.gz'
    #         file_path = os.path.join(PKL_GZ_DIR, file_name)
            
    #         # TODO: create temp obj creation and loading as an util function!
    #         if not os.path.exists(file_path):
    #             l_data = []
    #             for p in range(p_start, p_end):
    #                 for n in range(n_start, n_end):
    #                     l_seq = [i**p % n for i in range(1, n)]
    #                     l_data.append({'p': p, 'n': n, 'l_seq': l_seq})

    #             with gzip.open(file_path, 'wb') as f:
    #                 dill.dump(l_data, f)
    #         else:
    #             with gzip.open(file_path, 'rb') as f:
    #                 l_data = dill.load(f)
                
            # for d in l_data:
            #     p_d = d['p']
            #     n_d = d['n']
            #     assert p_d >= p_start and p_d <= p_end
            #     assert n_d >= n_start and n_d <= n_end
            #     l_seq = d['l_seq']
            #     d_p_to_d_n_to_l_seq[p_d][n_d] = l_seq

    #         # d_n_to_l_seq[n] = l_seq
    #     # d_p_to_d_n_to_l_seq[p] = d_n_to_l_seq

    d_n_to_d_tpl_to_l_p = {}
    d_n_to_amount_cycles = {}
    # d_n_to_l_tpl_unique_with_zero = {}
    d_n_to_l_tpl_unique = {}
    for n in range(2, N_MAX+1):
        print("n: {}".format(n))
        d_tpl_to_l_p = {}
        d_n_to_d_tpl_to_l_p[n] = d_tpl_to_l_p

        for p in range(1, P_MAX+1):
            l_seq = d_p_to_d_n_to_l_seq[p][n]
            tpl = tuple(l_seq)

            if tpl not in d_tpl_to_l_p:
                d_tpl_to_l_p[tpl] = []

            d_tpl_to_l_p[tpl].append(p)

        amount_cycles = len(d_tpl_to_l_p)
        d_n_to_amount_cycles[n] = amount_cycles

        l_tpl = list(d_tpl_to_l_p.keys())

        # l_tpl_unique_with_zero = []
        l_tpl_unique = []
        len_tpl = len(l_tpl[0])
        for tpl in l_tpl:
            u = np.sort(np.unique(tpl))
            if u.shape[0] == len_tpl:
                # l_tpl_unique_with_zero.append(tpl)
            
                if u[0] != 0:
                    l_tpl_unique.append(tpl)

        # d_n_to_l_tpl_unique_with_zero[n] = l_tpl_unique_with_zero
        d_n_to_l_tpl_unique[n] = l_tpl_unique

    l_seq_oeis_A109746 = [d_n_to_amount_cycles[k] for k in sorted(d_n_to_amount_cycles.keys())]
    
    # maybe an interesting sequence!
    # l_seq_len_of_unique_with_zero_cycles = [len(d_n_to_l_tpl_unique_with_zero[k]) for k in sorted(d_n_to_l_tpl_unique_with_zero.keys())]
    l_seq_len_of_unique_cycles = [len(d_n_to_l_tpl_unique[k]) for k in sorted(d_n_to_l_tpl_unique.keys())]

    # l_seq_n_len_of_unique_with_zero_cycles = [(k, len(d_n_to_l_tpl_unique_with_zero[k])) for k in sorted(d_n_to_l_tpl_unique_with_zero.keys())]
    l_seq_n_len_of_unique_cycles = [(k, len(d_n_to_l_tpl_unique[k])) for k in sorted(d_n_to_l_tpl_unique.keys())]

    l_n, l_a_n = list(zip(*l_seq_n_len_of_unique_cycles))
    arr_n = np.array(l_n)
    arr_a_n = np.array(l_a_n)
    
    arr_n_half_amount_full_cycles = arr_n[(arr_n // arr_a_n) == 2]

    l_seq_oeis_A003627 = list(arr_n_half_amount_full_cycles)

    l_min = []
    l_max = []
    l_mean = []

    for i in range(1, len(arr_n)+1):
        arr_a_n_part = arr_a_n[:i]

        l_min.append(np.min(arr_a_n_part))
        l_mean.append(np.mean(arr_a_n_part))
        l_max.append(np.max(arr_a_n_part))

    plt.figure()

    plt.title('l_seq_n_len_of_unique_cycles')
    plt.plot(arr_n, arr_a_n/arr_n, 'bo')

    plt.xlabel('arr_n')
    plt.xlabel('arr_a_n/arr_n')


    plt.figure()

    plt.title('l_seq_n_len_of_unique_cycles')
    plt.plot(arr_n, arr_a_n, 'bo')

    plt.xlabel('arr_n')
    plt.xlabel('arr_a_n')


    plt.figure()

    plt.title('l_seq_n_len_of_unique_cycles min/mean/max')
    plt.plot(arr_n, l_min, 'b.')
    plt.plot(arr_n, l_mean, 'k.')
    plt.plot(arr_n, l_max, 'r.')

    plt.xlabel('arr_n')
    plt.xlabel('min/mean/max')


    plt.show()
